import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import datasets

class WikiAnnDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer):
        self.sentences = sentences
        self.sentences_tags = tags

        self.tokenizer = tokenizer

        # self.ner_tags = ['<PAD>'] + list(set(tag for tag_list in self.sentences_tags for tag in tag_list))
        #self.tag2idx = {tag: idx for idx, tag in enumerate(self.ner_tags)}
        #self.idx2tag = {idx: tag for idx, tag in enumerate(self.ner_tags)}
        self.idx2tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: '<PAD>'}
        self.tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 7: '<PAD>'}

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

        #tokenized_tags = ['O'] + tokenized_tags + ['O']
        tokenized_tags = [self.tag2idx['O']] + tokenized_tags + [self.tag2idx['O']]
        # tags_ids = [self.tag2idx[tag] for tag in tokenized_tags]

        return torch.LongTensor(tokens_ids), torch.LongTensor(tokenized_tags)

    def paddings(self, batch):
        tokens, tags = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.tag2idx['<PAD>'])
        tags = pad_sequence(tags, batch_first=True, padding_value=self.tag2idx['<PAD>'])

        return tokens, tags


def read_data(filename):
    rows = open(filename, 'r').read().strip().split("\n\n")
    sentences, sentences_tags = [], []

    for sentence in rows:
        words = [line.split()[0] for line in sentence.splitlines()]
        tags = [line.split()[-1] for line in sentence.splitlines()]
        sentences.append(words)
        sentences_tags.append(tags)

    return sentences, sentences_tags

def get_sentences_and_tags(mode, data):
    sentences, tags = [], []
    for element in data[mode]:
        sentences.append(element['tokens'])
        tags.append(element['ner_tags'])

    return sentences, tags

def create_dataset_and_dataloader(mode, lang, batch_size, tokenizer):
    lang_data = datasets.load_dataset("wikiann", lang)

    sentences, tags = get_sentences_and_tags(mode, lang_data)
    dataset = WikiAnnDataset(sentences, tags, tokenizer)

    return dataset, DataLoader(dataset, batch_size, num_workers=4, collate_fn=dataset.paddings)