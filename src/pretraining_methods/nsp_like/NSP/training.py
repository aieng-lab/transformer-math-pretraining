"""Next Sentence Prediction"""


from src.pretraining_methods.Objectives import Objectives

import torch
from torch.nn import CrossEntropyLoss


def get_nfir_losses(outputs, labels):
    return get_nsp_losses(outputs, labels, objective=Objectives.NFIR)

def get_ffir_losses(outputs, labels):
    return get_nsp_losses(outputs, labels, objective=Objectives.FFIR)

def get_nsp_losses(outputs, labels, objective=Objectives.NSP):
    loss_fn = CrossEntropyLoss()
    if isinstance(outputs, dict):
        outputs = outputs[objective.name]

    if isinstance(labels, dict):
        labels = labels[objective.name]

    outputs_transform = outputs.view(-1, 2)
    labels_transform = labels.view(-1)
    loss = loss_fn(outputs_transform, labels_transform)
    return loss


def get_correct_nfir_predictions(outputs, targets):
    return get_correct_predictions(outputs, targets, objective=Objectives.NFIR)

def get_correct_ffir_predictions(outputs, targets):
    return get_correct_predictions(outputs, targets, objective=Objectives.FFIR)

def get_correct_predictions(outputs, targets, objective=Objectives.NSP):
    output = outputs.get(objective.name)
    _, output = torch.max(output, dim=-1)
    labels = targets.get(objective.name)
    labels = labels[:, 0]
    nominator = (torch.sum(output == labels)).item()
    denominator = len(labels)
    return nominator, denominator

def get_model_inputs_nfir(example):
    return get_model_inputs(example, objective=Objectives.NFIR)

def get_model_inputs_ffir(example):
    return get_model_inputs(example, objective=Objectives.FFIR)

def get_model_inputs(example, objective=Objectives.NSP):
    input_ids = example.get("bert_input")
    attention_mask = example.get("attention_mask")
    token_type_ids = example.get("segment_label")
    if objective == Objectives.NSP:
        get_targets_func = get_targets
    elif objective == Objectives.NFIR:
        get_targets_func = get_targets_nfir
    elif objective == Objectives.FFIR:
        get_targets_func = get_targets_ffir
    else:
        raise ValueError("Unknown objective %s" % objective)

    labels = get_targets_func(example)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": token_type_ids,
        "labels": labels
    }


def get_targets_nfir(example):
    return get_targets(example, Objectives.NFIR)

def get_targets_ffir(example):
    return get_targets(example, Objectives.FFIR)

def get_targets(example, objective=Objectives.NSP):
    nsp_label = example["bert_label_" + objective.name]
    return {
        objective.name: nsp_label
    }
