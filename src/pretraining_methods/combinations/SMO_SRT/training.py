from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.mlm_like.SMO.training import get_smo_losses
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
    smo_labels = example.get("bert_label_SMO")
    srt_label = example["bert_label_SRT"]
    return {
        "SMO": smo_labels,
        "SRT": srt_label
    }


def get_losses(output_dict, labels):
    smo_output = output_dict.get(Objectives.SMO.name)
    srt_output = output_dict.get(Objectives.SRT.name)
    smo_labels = labels.get(Objectives.SMO.name)
    srt_target = labels.get(Objectives.SRT.name)
    smo_loss = get_smo_losses(smo_output, smo_labels)
    srt_loss = get_srt_losses(srt_output, srt_target)
    # mean of losses do not have to be calculated, as the Cross entropy loss returns the average of a batch by default
    total_loss = smo_loss + srt_loss
    return total_loss
