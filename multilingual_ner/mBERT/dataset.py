import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


class WikiAnnDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer):
        self.sentences = sentences
        self.sentences_tags = tags

        self.tokenizer = tokenizer

        self.ner_tags = ['<PAD>'] + list(set(tag for tag_list in self.sentences_tags for tag in tag_list))
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.ner_tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.ner_tags)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        words = self.sentences[item]
        tags = self.sentences_tags[item]

        word2tag = dict(zip(words, tags))

        tokens = []
        tokenized_tags = []

        for word in words:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                for i in range(len(subtokens)):
                    tokenized_tags.append(word2tag[word])
                tokens.extend(subtokens)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokenized_tags = ['O'] + tokenized_tags + ['O']
        tags_ids = [self.tag2idx[tag] for tag in tokenized_tags]

        return torch.LongTensor(tokens_ids), torch.LongTensor(tags_ids)

    def entities_statistics(self):
        entity_types_counter = Counter()

        for tags_idx in range(len(self.sentences_tags)):
            entity = []

            for idx in range(len(self.sentences_tags[tags_idx])):
                if self.sentences_tags[tags_idx][idx] != 'O':
                    entity.append(self.sentences_tags[tags_idx][idx])
                else:
                    entity_types_counter['O'] += 1

            if entity:
                for idx in range(len(entity)):
                    if entity[idx][0] == 'B':
                        entity_tag = entity[idx][2:]
                        entity_types_counter[entity_tag] += 1

        return entity_types_counter

    def BIO_tags_statistics(self):
        bio_types_counter = Counter()

        for tags_idx in range(len(self.sentences_tags)):
            for idx in range(len(self.sentences_tags[tags_idx])):
                bio_types_counter[self.sentences_tags[tags_idx][idx]] += 1

        return bio_types_counter

    def paddings(self, batch):
        tokens, tags = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.tag2idx['<PAD>'])
        tags = pad_sequence(tags, batch_first=True, padding_value=self.tag2idx['<PAD>'])

        return tokens, tags


def read_data(filename):
    rows = open(filename, 'r').read().strip().split("\n\n")
    sentences, sentences_tags = [], []

    for sentence in rows:
        words = [line.split()[0][3:] for line in sentence.splitlines()]
        tags = [line.split()[-1] for line in sentence.splitlines()]
        sentences.append(words)
        sentences_tags.append(tags)

    return sentences, sentences_tags


def create_dataset_and_dataloader(filename, batch_size, tokenizer):
    sentences, tags = read_data(filename)
    dataset = WikiAnnDataset(sentences, tags, tokenizer)

    return dataset, DataLoader(dataset, batch_size, collate_fn=dataset.paddings)