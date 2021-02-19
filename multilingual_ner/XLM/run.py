from multilingual_ner.XLM.dataset import create_dataset_and_dataloader
from multilingual_ner.XLM.train import train_model
from multilingual_ner.XLM.model import XLMRoBERTaNER

from transformers import XLMRobertaTokenizer

import torch
import torch.nn as nn
import torch.optim as optim

TOKENIZER = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=False)
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
EPOCHS = 4
BATCH_SIZE = 16

train_dataset, train_dataloader = create_dataset_and_dataloader('train', BATCH_SIZE, TOKENIZER)
eval_dataset, eval_dataloader = create_dataset_and_dataloader('validation', BATCH_SIZE, TOKENIZER)
test_dataset, test_dataloader = create_dataset_and_dataloader('test', BATCH_SIZE, TOKENIZER)

classes = len(train_dataset.ner_tags)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

XLM_Model = XLMRoBERTaNER(classes).to(device)
optimizer = optim.AdamW(XLM_Model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
EPOCHS = 4

train_model(XLM_Model, criterion, optimizer, train_dataloader, eval_dataloader, train_dataset.tag2idx, train_dataset.idx2tag, device, None, EPOCHS)