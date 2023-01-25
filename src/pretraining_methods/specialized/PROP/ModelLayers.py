"""PROP"""


import torch
from torch import nn
from transformers import BertForNextSentencePrediction


class PROPLayers(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.output = nn.Linear(hidden, 1) #todo add dropout with p=0.1
        #self.softmax = nn.LogSoftmax(dim=-1) # No need for softmax as it's already included in the CrossEntropyLoss

    def forward(self, x):
        """
        :param x: Pooled output of BERT model
        :return:
        """
        linear_output = self.output(self.dropout(x))
        return linear_output
