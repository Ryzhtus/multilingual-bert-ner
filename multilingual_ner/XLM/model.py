from transformers import XLMRobertaModel
import torch.nn as nn

class XLMRoBERTaNER(nn.Module):
    def __init__(self, num_classes):
        super(XLMRoBERTaNER, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.RoBERTa = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, tokens):
        embeddings = self.RoBERTa(tokens)[0]
        predictions = self.linear(embeddings)

        return predictions