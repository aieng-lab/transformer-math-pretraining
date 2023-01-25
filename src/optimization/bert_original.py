from src.config.config import Config

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def get_optimizer(model, config: Config):
    base_lr = config.LR
    epsilon = config.EPSILON
    beta2 = config.BETA2
    effective_batch_size = config.BATCH_SIZE
    if config.NUM_GPUS:
        effective_batch_size = effective_batch_size * config.NUM_GPUS
    lr = base_lr * (effective_batch_size / 256)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=epsilon, betas=(0.9, beta2))

    for name, param in model.named_parameters():
        if 'bias' in name or 'LayerNorm' in name:  # exclude bias and layer norm parameters
            param_group = optimizer.param_groups[0]  # get the first (and only) parameter group
            param_group['weight_decay'] = 0.0  # set weight decay to zero for this parameter group
        else:
            param_group = optimizer.param_groups[0]  # get the first (and only) parameter group
            param_group['weight_decay'] = 0.01
    return optimizer


def get_scheduler(optimizer, num_training_steps, config: Config):
    warmup_steps = int(config.WARMUP)
    if config.NUM_GPUS:
        warmup_steps = warmup_steps * config.NUM_GPUS
        num_training_steps = num_training_steps * config.NUM_GPUS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    if config.NUM_GPUS:
        warmup_steps = warmup_steps / config.NUM_GPUS  # print the original number of warmup steps
    return scheduler, warmup_steps
