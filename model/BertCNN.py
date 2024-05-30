import torch
import torch.nn as nn
import torch.nn.functional as F

class BertCNN(nn.Module):
    def __init__(self, bert_model, cnn_model):
        super(BertCNN, self).__init__()
        self.bert = bert_model
        self.cnn = cnn_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]  
        logits = self.cnn(cls_output)
        return logits
