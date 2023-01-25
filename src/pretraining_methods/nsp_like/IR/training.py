"""Next Sentence Prediction"""


from src.pretraining_methods.Objectives import Objectives

import torch
from torch.nn import CrossEntropyLoss


def get_ir_losses(outputs, labels, objective=Objectives.IR):
    loss_fn = CrossEntropyLoss()
    if isinstance(outputs, dict):
        outputs = outputs[objective.name]

    if isinstance(labels, dict):
        labels = labels[objective.name]

    outputs_transform = outputs.view(-1, 2)
    labels_transform = labels.view(-1)
    loss = loss_fn(outputs_transform, labels_transform)
    return loss

def get_correct_predictions(outputs, targets, objective=Objectives.IR):
    output = outputs.get(objective.name)
    _, output = torch.max(output, dim=-1)
    labels = targets.get(objective.name)
    labels = labels[:, 0]
    nominator = (torch.sum(output == labels)).item()
    denominator = len(labels)
    return nominator, denominator

def get_model_inputs(example, objective=Objectives.IR):
    input_ids = example.get("bert_input")
    attention_mask = example.get("attention_mask")
    token_type_ids = example.get("segment_label")

    labels = get_targets(example)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": token_type_ids,
        "labels": labels
    }

def get_targets(example, objective=Objectives.IR):
    nsp_label = example["bert_label_" + objective.name]
    return {
        objective.name: nsp_label
    }
