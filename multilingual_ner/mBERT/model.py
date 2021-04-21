from transformers import BertModel
import torch.nn as nn

class BertNER(nn.Module):
    def __init__(self, num_classes, pretrained='bert-base-multilingual-cased'):
        super(BertNER, self).__init__()
        self.embedding_dim = 768
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(pretrained, output_attentions=True)
        self.linear = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, tokens):
        outputs = self.bert(tokens)
        last_hidden_state = outputs['last_hidden_state']
        predictions = self.linear(last_hidden_state)

        return predictions, outputs