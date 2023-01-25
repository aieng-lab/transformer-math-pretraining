from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.mlm_like.MLM.training import get_mlm_losses
from src.pretraining_methods.nsp_like.SOP.training import get_sop_losses


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
    sop_label = example["bert_label_SOP"]
    return {
        "MLM": mlm_labels,
        "SOP": sop_label
    }


def get_losses(output_dict, labels):
    mlm_output = output_dict.get(Objectives.MLM.name)
    sop_output = output_dict.get(Objectives.SOP.name)
    mlm_labels = labels.get(Objectives.MLM.name)
    sop_target = labels.get(Objectives.SOP.name)
    mlm_loss = get_mlm_losses(mlm_output, mlm_labels)
    sop_loss = get_sop_losses(sop_output, sop_target)
    # mean of losses do not have to be calculated, as the Cross entropy loss returns the average of a batch by default
    total_loss = mlm_loss + sop_loss
    return total_loss
