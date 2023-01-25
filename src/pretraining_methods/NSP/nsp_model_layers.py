import torch
from torch import nn
from transformers import BertForNextSentencePrediction


class NSPLayers(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        #self.softmax = nn.LogSoftmax(dim=-1) # No need for softmax as it's already included in the CrossEntropyLoss

    def forward(self, x):
        """
        :param x: Pooled output of BERT model
        :return:
        """
        linear_output = self.linear(x)
        return linear_output
