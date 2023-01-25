"""Span Boundary Objective"""

import torch
from torch import nn
from transformers import activations


class SBOLayers(nn.Module):

    def __init__(self, hidden, vocab_size, position_embedding):
        super().__init__()
        self.hidden = hidden
        self.position_embedding = position_embedding
        self.linear1 = nn.Linear(hidden * 3, hidden * 3)
        self.linear2 = nn.Linear(hidden * 3, hidden * 3)
        self.output = nn.Linear(hidden * 3, vocab_size)
        self.activation = activations.get_activation("gelu")
        self.layer_norm1 = nn.LayerNorm(hidden * 3)
        self.layer_norm2 = nn.LayerNorm(hidden * 3)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.output.bias = self.bias

    def forward(self, x, boundary_dicts):
        # x are token represenations produced by bert
        # concatenate positional encodings with span boundaries
        batched_concatenated_representations = []
        for i, batch in enumerate(x):
            concatenated_representations = []
            boundary_dict = boundary_dicts[i]
            for j, item in batch:
                if j in boundary_dict.keys():
                    left_boundary = boundary_dict.get(j).get("left")
                    right_boundary = boundary_dict.get(j).get("right")
                    left_repr = batch[left_boundary]
                    right_repr = batch[right_boundary]
                    # todo: maybe with torch.no_grad()?
                    relative_pos = self.position_embedding(torch.tensor(j - left_boundary))
                    concatenated = torch.cat((left_repr, right_repr, relative_pos))
                else:
                    concatenated = torch.full((self.hidden*3,), -100)
                if not concatenated_representations:
                    concatenated_representations = concatenated
                else:
                    concatenated_representations = torch.stack((concatenated_representations, concatenated))
            if not batched_concatenated_representations:
                batched_concatenated_representations = concatenated_representations
            else:
                batched_concatenated_representations = torch.stack((batched_concatenated_representations,
                                                                    concatenated_representations))

        hidden1 = self.linear1(batched_concatenated_representations)
        hidden1 = self.activation(hidden1)
        hidden1 = self.layer_norm1(hidden1)
        hidden2 = self.linear2(hidden1)
        hidden2 = self.activation(hidden2)
        hidden2 = self.layer_norm2(hidden2)
        output = self.output(hidden2)
        return output
