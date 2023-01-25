import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot


# todo: Beware of batching during training
# todo: Are all batches concatenated? or stacked?
# todo: Could we have outputs = [234, 345, 456, 567, 678, 789, 456, 345, 456, 4576]
# todo: And labels = [1, 4, 9, 0, 4, 6] etc
# todo: Then position extraction with torch.gather would be meaningless and would have to be done in the model


def get_losses(output, labels, positions, vocab_size):
    predictions = torch.tensor([])
    real_values = torch.tensor([])
    for i in range(len(output)):
        mask_positions = positions[i][positions[i] < 512]
        real_vals = labels[i][labels[i] > 0]
        preds = output[i][mask_positions, :]
        real_values = torch.cat((real_values, real_vals), 0)
        predictions = torch.cat((predictions, preds))
    #real_values = one_hot(real_values.to(torch.int64), num_classes=vocab_size)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(predictions, real_values.to(torch.long))
    return loss


def get_correct_predictions(outputs, targets):
    output = outputs.get("MLM")
    positions = targets.get("MLM").get("positions")
    labels = targets.get("MLM").get("labels")
    nominator = 0
    denominator = 0
    for i in range(len(output)):
        _, preds = torch.max(output[i], dim=-1)
        mask_pos = positions[i][positions[i] < 512]
        real_values = labels[i][labels[i] > 0]
        preds = torch.gather(preds, 0, mask_pos)
        nominator += int(torch.sum(preds == real_values))
        denominator += int(len(real_values))
    return nominator, denominator
