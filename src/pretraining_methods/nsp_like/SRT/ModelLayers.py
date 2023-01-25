"""Sentence Reordering Task (as in ERNIE 2.0)"""

from torch import nn
import nltk


class SRTLayers(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.output = nn.Linear(hidden, 153)  # 1! + 2! + 3! + 4! + 5!
        nltk.download("punkt")

    def forward(self, x):
        """
        :param x: Pooled output of BERT model
        :return:
        """
        linear_output = self.output(x)
        return linear_output
