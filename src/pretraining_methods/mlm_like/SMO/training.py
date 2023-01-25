"""Span Masking Objective"""


from src.pretraining_methods.Objectives import Objectives

import torch
from torch.nn import CrossEntropyLoss


def get_smo_losses(outputs, targets):
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    outputs_flatten = outputs.view(-1, outputs.size(-1))  # outputs.size(-1) should be the vocab_size
    targets_flatten = targets.view(-1)
    loss = loss_fn(outputs_flatten, targets_flatten)
    return loss


def get_correct_predictions(outputs, targets):
    output = outputs.get(Objectives.SMO.name)
    labels = targets.get(Objectives.SMO.name)
    _, preds = torch.max(output, dim=-1)
    mask = labels != -100
    correct_preds = torch.sum(torch.masked_select(preds == labels, mask))
    nominator = correct_preds.item()
    labels = labels[labels != -100]
    denominator = len(labels)
    return nominator, denominator


