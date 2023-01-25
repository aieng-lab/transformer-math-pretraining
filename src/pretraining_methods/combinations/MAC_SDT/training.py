from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.mlm_like.MAC.training import get_mac_losses
from src.pretraining_methods.nsp_like.SDT.training import get_sdt_losses


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
    sdt_label = example["bert_label_SDT"]
    return {
        "MAC": mac_labels,
        "SDT": sdt_label
    }


def get_losses(output_dict, labels):
    mac_output = output_dict.get(Objectives.MAC.name)
    sdt_output = output_dict.get(Objectives.SDT.name)
    mac_labels = labels.get(Objectives.MAC.name)
    sdt_target = labels.get(Objectives.SDT.name)
    mac_loss = get_mac_losses(mac_output, mac_labels)
    sdt_loss = get_sdt_losses(sdt_output, sdt_target)
    # mean of losses do not have to be calculated, as the Cross entropy loss returns the average of a batch by default
    total_loss = mac_loss + sdt_loss
    return total_loss