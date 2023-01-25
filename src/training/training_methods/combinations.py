from src.training.training_methods import mlm, nsp

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np


def mlm_plus_nsp__get_model_inputs(example):
    input_ids = example.get("bert_input")
    attention_mask = example.get("attention_mask")
    token_type_ids = example.get("segment_label")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": token_type_ids
    }


def mlm_plus_nsp__get_targets(example):
    mlm_labels = example.get("bert_label_mlm")
    mask_positions = example["bert_label_mask_positions"]
    nsp_label = example["bert_label_nsp"]
    return {
        "MLM": {"positions": mask_positions, "labels": mlm_labels},
        "NSP": nsp_label
    }


def mlm_plus_nsp__get_optimizer(model):
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, eps=1e-6)
    return optimizer


def mlm_plus_nsp__get_scheduler(optimizer, num_training_steps):
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10_000,
                                                num_training_steps=num_training_steps)
    return scheduler


# todo: beware of batching
def mlm_plus_nsp__get_losses(model_outputs, targets, config):
    mlm_output = model_outputs.get("MLM")
    nsp_output = model_outputs.get("NSP")
    mlm_labels = targets.get("MLM").get("labels")
    mlm_positions = targets.get("MLM").get("positions")
    nsp_target = targets.get("NSP")
    mlm_loss = mlm.get_losses(mlm_output, mlm_labels, mlm_positions, config.vocab_size)
    nsp_loss = nsp.get_losses(nsp_output, nsp_target)
    # mean of losses do not have to be calculated, as the Cross entropy loss returns the average of a batch by default
    total_loss = mlm_loss + nsp_loss
    return total_loss
