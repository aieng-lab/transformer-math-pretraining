import numpy as np
import torch
from torch.nn import CrossEntropyLoss


def get_losses(outputs, labels):
    real_values = labels[labels < 512]
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(outputs, real_values)
    return loss


def get_correct_predictions(outputs, targets):
    output = outputs.get("NSP")
    _, output = torch.max(output, dim=-1)
    labels = targets.get("NSP")
    labels = labels[:, 0]
    nominator = int(torch.sum(output == labels))
    denominator = int(len(labels))
    return nominator, denominator
