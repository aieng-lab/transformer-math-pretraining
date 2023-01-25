from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.mlm_like.MAC.training import get_mac_losses
from src.pretraining_methods.nsp_like.NSP.training import get_nsp_losses


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
    mac_labels = example.get("bert_label_MAC")
    nsp_label = example["bert_label_NSP"]
    return {
        "MAC": mac_labels,
        "NSP": nsp_label
    }


def get_losses(output_dict, labels):
    mac_output = output_dict.get(Objectives.MAC.name)
    nsp_output = output_dict.get(Objectives.NSP.name)
    mac_labels = labels.get(Objectives.MAC.name)
    nsp_target = labels.get(Objectives.NSP.name)
    mac_loss = get_mac_losses(mac_output, mac_labels)
    nsp_loss = get_nsp_losses(nsp_output, nsp_target)
    # mean of losses do not have to be calculated, as the Cross entropy loss returns the average of a batch by default
    total_loss = mac_loss + nsp_loss
    return total_loss
