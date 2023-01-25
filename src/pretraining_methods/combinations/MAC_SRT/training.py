from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.mlm_like.MAC.training import get_mac_losses
from src.pretraining_methods.nsp_like.SRT.training import get_srt_losses


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
    srt_label = example["bert_label_SRT"]
    return {
        "MAC": mac_labels,
        "SRT": srt_label
    }


def get_losses(output_dict, labels):
    mac_output = output_dict.get(Objectives.MAC.name)
    srt_output = output_dict.get(Objectives.SRT.name)
    mac_labels = labels.get(Objectives.MAC.name)
    srt_target = labels.get(Objectives.SRT.name)
    mac_loss = get_mac_losses(mac_output, mac_labels)
    srt_loss = get_srt_losses(srt_output, srt_target)
    total_loss = mac_loss + srt_loss
    return total_loss
