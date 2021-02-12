from multilingual_ner.dataset import WikiAnnDataset, create_dataset_and_dataloader
from multilingual_ner.train import train_model, test_epoch
from multilingual_ner.model import BertNER

from transformers import BertTokenizer

import torch
import torch.nn as nn
import torch.optim as optim

TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
EPOCHS = 4
BATCH_SIZE = 16
LANGUAGE = 'es'

train_dataset, train_dataloader = create_dataset_and_dataloader('train', LANGUAGE, BATCH_SIZE, TOKENIZER)
eval_dataset, eval_dataloader = create_dataset_and_dataloader('validation', LANGUAGE, BATCH_SIZE, TOKENIZER)
test_dataset, test_dataloader = create_dataset_and_dataloader('test', LANGUAGE, BATCH_SIZE, TOKENIZER)

classes = len(train_dataset.ner_tags)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SpanishBertModel = BertNER(classes, pretrained='bert-base-multilingual-cased').to(device)
optimizer = optim.AdamW(SpanishBertModel.parameters(), lr=2e-6)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
EPOCHS = 4

train_model(SpanishBertModel, criterion, optimizer, train_dataloader, eval_dataloader, train_dataset.tag2idx, train_dataset.idx2tag, device, None, EPOCHS)