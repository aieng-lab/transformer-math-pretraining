"""Sentence Distance Task"""


from src.pretraining_methods.Objectives import Objectives

import torch
from torch.nn import CrossEntropyLoss


def get_sdt_losses(outputs, labels):
    loss_fn = CrossEntropyLoss()
    outputs_transform = outputs.view(-1, 3)
    labels_transform = labels.view(-1)
    loss = loss_fn(outputs_transform, labels_transform)
    return loss


def get_correct_predictions(outputs, targets):
    output = outputs.get(Objectives.SDT.name)
    _, output = torch.max(output, dim=-1)
    labels = targets.get(Objectives.SDT.name)
    labels = labels[:, 0]
    nominator = (torch.sum(output == labels)).item()
    denominator = len(labels)
    return nominator, denominator