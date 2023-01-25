"""PROP"""


from src.pretraining_methods.Objectives import Objectives

import torch
from torch.nn import MarginRankingLoss


def get_prop_losses(outputs, labels):
    loss_fn = MarginRankingLoss(margin=1.0, reduction='mean')
    outputs_1 = outputs[0]
    outputs_2 = outputs[1]
    outputs_1_transform = outputs_1.view(-1)
    outputs_2_transform = outputs_2.view(-1)
    labels_transform = labels.view(-1)
    loss = loss_fn(outputs_1_transform, outputs_2_transform, labels_transform)
    return loss


def get_correct_predictions(outputs, targets):
    output_list = outputs.get(Objectives.PROP.name)
    output_set_1 = output_list[0]
    output_set_2 = output_list[1]
    output = torch.cat((output_set_1, output_set_2), dim=1)
    _, output = torch.max(output, dim=-1)
    labels = targets.get(Objectives.PROP.name)
    labels = labels[:, 0]
    labels = (labels != 1).long()
    nominator = (torch.sum(output == labels)).item()
    denominator = len(labels)
    return nominator, denominator







