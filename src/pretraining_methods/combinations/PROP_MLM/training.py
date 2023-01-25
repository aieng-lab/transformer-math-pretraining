from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.mlm_like.MLM.training import get_mlm_losses
from src.pretraining_methods.specialized.PROP.training import get_prop_losses

import torch


def get_model_inputs(example):
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


def get_targets(example):
    mlm_labels = example.get("bert_label_MLM")
    mlm_labels = torch.cat((mlm_labels[0], mlm_labels[1]), dim=0)
    prop_label = example["bert_label_PROP"]
    return {
        "MLM": mlm_labels,
        "PROP": prop_label
    }


def get_losses(output_dict, labels):
    mlm_output = output_dict.get(Objectives.MLM.name)
    prop_output_list = output_dict.get(Objectives.PROP.name)
    mlm_labels = labels.get(Objectives.MLM.name)
    prop_target = labels.get(Objectives.PROP.name)
    mlm_loss = get_mlm_losses(mlm_output, mlm_labels)
    prop_loss = get_prop_losses(prop_output_list, prop_target)
    # mean of losses do not have to be calculated, as the Cross entropy loss returns the average of a batch by default
    total_loss = mlm_loss + prop_loss
    return total_loss
