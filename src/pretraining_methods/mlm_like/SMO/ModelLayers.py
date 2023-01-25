"""Span Masking Objective"""

import torch
from torch import nn
from transformers import activations


class SMOLayers(nn.Module):

    def __init__(self, hidden, vocab_size, activation):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        if isinstance(activation, str):
            self.activation = activations.get_activation(activation)
        else:
            self.activation = activations.get_activation("gelu")
        self.layer_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.output.bias = self.bias

    def forward(self, x):
        hidden = self.dense(x)
        hidden = self.activation(hidden)
        hidden = self.layer_norm(hidden)
        lin_output = self.output(hidden)
        return lin_output
