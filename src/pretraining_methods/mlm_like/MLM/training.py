"""Masked Language Modeling"""


from src.pretraining_methods.Objectives import Objectives

import torch
from torch.nn import CrossEntropyLoss


def get_losses_MLM_MATH(output_dict, labels):
    return get_losses(output_dict, labels, obj=Objectives.MFM)

def get_losses_MLM_MATH_TEXT(output_dict, labels):
    return get_losses(output_dict, labels, obj=Objectives.MTM)

def get_losses(output_dict, labels, obj=Objectives.MLM):
    mlm_output = output_dict.get(obj.name)
    mlm_labels = labels.get(obj.name)
    mlm_loss = get_mlm_losses(mlm_output, mlm_labels)
    return mlm_loss

def get_mlm_losses(outputs, targets):
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    outputs_flatten = outputs.view(-1, outputs.size(-1))  # outputs.size(-1) should be the vocab_size
    targets_flatten = targets.view(-1)
    loss = loss_fn(outputs_flatten, targets_flatten)
    return loss


def get_correct_predictions_MATH(outputs, targets):
    return get_correct_predictions(outputs, targets, obj=Objectives.MFM)

def get_correct_predictions_MATH_TEXT(outputs, targets):
    return get_correct_predictions(outputs, targets, obj=Objectives.MTM)

def get_correct_predictions(outputs, targets, obj=Objectives.MLM):
    output = outputs.get(obj.name)
    labels = targets.get(obj.name)
    _, preds = torch.max(output, dim=-1)
    mask = labels != -100
    correct_preds = torch.sum(torch.masked_select(preds == labels, mask))
    nominator = correct_preds.item()
    labels = labels[labels != -100]
    denominator = len(labels)
    return nominator, denominator


def get_model_inputs_MLM_MATH(example):
    return get_model_inputs(example, obj=Objectives.MFM)

def get_model_inputs_MLM_MATH_TEXT(example):
    return get_model_inputs(example, obj=Objectives.MTM)

def get_model_inputs(example, obj=Objectives.MLM):
    input_ids = example.get("bert_input")
    attention_mask = example.get("attention_mask")
    token_type_ids = example.get("segment_label")
    mlm_labels = example.get("bert_label_%s" % (obj.name))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": token_type_ids,
        "labels": {obj.name: mlm_labels}
    }


def get_targets_MLM_MATH(example):
    return get_targets(example, obj=Objectives.MFM)

def get_targets_MLM_MATH_TEXT(example):
    return get_targets(example, obj=Objectives.MTM)

def get_targets(example, obj=Objectives.MLM):
    mlm_labels = example.get("bert_label_" + obj.name)
    return {
        obj.name: mlm_labels
    }
