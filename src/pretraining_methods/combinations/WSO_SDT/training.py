from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.nsp_like.SDT.training import get_sdt_losses
from src.pretraining_methods.mlm_like.WSO.training import get_wso_losses


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
    wso_label = example.get("bert_label_WSO")
    sdt_label = example["bert_label_SDT"]
    return {
        "WSO": wso_label,
        "SDT": sdt_label
    }


def get_losses(output_dict, labels):
    wso_output = output_dict.get(Objectives.WSO.name)
    sdt_output = output_dict.get(Objectives.SDT.name)
    wso_labels = labels.get(Objectives.WSO.name)
    sdt_target = labels.get(Objectives.SDT.name)
    wso_loss = get_wso_losses(wso_output, wso_labels)
    sdt_loss = get_sdt_losses(sdt_output, sdt_target)
    # mean of losses do not have to be calculated, as the Cross entropy loss returns the average of a batch by default
    total_loss = wso_loss + sdt_loss
    return total_loss
