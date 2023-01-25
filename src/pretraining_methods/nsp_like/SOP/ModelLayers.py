"""Sentence Order Prediction"""

from torch import nn


class SOPLayers(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.output = nn.Linear(hidden, 2)
        # self.softmax = nn.LogSoftmax(dim=-1) # No need for softmax as it's already included in the CrossEntropyLoss

    def forward(self, x):
        """
        :param x: Pooled output of BERT model
        :return:
        """
        linear_output = self.output(x)
        return linear_output
