"""Sentence Distance Task"""

from torch import nn


class SDTLayers(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.output = nn.Linear(hidden, 3)

    def forward(self, x):
        """
        :param x: Pooled output of BERT model
        :return:
        """
        linear_output = self.output(x)
        return linear_output
