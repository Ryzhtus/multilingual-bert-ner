from transformers import BertModel
import torch.nn as nn

class BertNER(nn.Module):
    def __init__(self, num_classes, pretrained='bert-base-multilingual-cased'):
        super(BertNER, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(pretrained)
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, tokens):
        embeddings = self.bert(tokens)[0]
        predictions = self.linear(embeddings)

        return predictions