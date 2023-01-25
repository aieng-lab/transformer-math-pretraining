import torch
import os
import math


class Config:

    def __init__(self):
        self.DEVICE = torch.device("cpu")
        self.PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.OUTPUTDIR = os.path.join(self.PYTHON_DIR, "output")
        self.DATA_PATH = os.path.join(self.PYTHON_DIR, "data")
        self.MODEL_PATH = os.path.join(self.OUTPUTDIR, "models")
        self.DATA_WORKERS = 2
        self.BATCH_SIZE = 16
        self.EPOCHS = 1
        self.TRIAL_RUN = False
        self.MAX_LEN = 128
        self.FINAL_EVAL = False
        self.VOCAB_SIZE = None  # vocab_size is set after a tokenizer was selected with Tokenizer.get_tokenizer()
        self.ACC_STEPS = 1
        self.NUM_GPUS = None
        self.RUN_NUMBER = None
        self.LR = 1e-4
        self.WARMUP = 10
        self.EPSILON = 1e-6
        self.BETA2 = 0.999
        self.MAX_THREADS = None
        self.OPTIMIZER_STEPS = None
        self.VALIDATE_INTERVAL = None
        self.COMET_API_KEY = None
        self.EPOCH_LEN = None
        self.PRETRAINED_MODEL_PATH = "/models/tokenized"
        self.PRETRAINED_OBJECTIVES = []
        self.RETRAIN = True

    def __str__(self):
        attributes = [f"{attr}: {value}" for attr, value in self.__dict__.items()]
        return "\n".join(attributes)

    def get_total_batches(self, data_len):
        return int(data_len / self.BATCH_SIZE) + 1

    def get_trial_batch_num(self, num_batches):
        percentage = self.TRIAL_RUN / 100
        trial_part = math.ceil(num_batches * percentage)
        return trial_part

    def get_trial_step_and_interval(self, steps, interval_length):
        percentage = self.TRIAL_RUN / 100
        trial_steps = math.ceil(steps * percentage)
        trial_interval = math.ceil(interval_length * percentage)
        if trial_interval < 10:
            # trial_interval = trial_steps
            # todo: remove the following line
            trial_interval = 30
        return trial_steps, trial_interval

    def get_fields_dict(self):
        fields = vars(self)
        fields.pop("COMET_API_KEY")
        return fields

    def determine_batch_logging(self, batch_number, percent_num, index, wait_steps):
        percentage = math.ceil(percent_num / wait_steps)
        log_batches = math.ceil(batch_number * percentage)
        if index % log_batches == 0:
            return True
        else:
            return False

    def determine_train_steps(self, train_loader_len):
        if self.OPTIMIZER_STEPS:
            return self.OPTIMIZER_STEPS
        else:
            return self.EPOCHS * train_loader_len

    def determine_interval_length(self, train_loader_length):
        if not self.VALIDATE_INTERVAL:
            return train_loader_length
        else:
            return self.VALIDATE_INTERVAL

    def determine_epochs(self, train_loader_len):
        if not self.OPTIMIZER_STEPS:
            return self.EPOCHS
        else:
            return math.ceil(self.OPTIMIZER_STEPS / train_loader_len)

    def determine_epoch_length(self, dataset_len):
        if self.NUM_GPUS:
            effective_batch_size = self.BATCH_SIZE * self.NUM_GPUS
        else:
            effective_batch_size = self.BATCH_SIZE

        epoch_len = math.ceil(dataset_len / effective_batch_size)
        self.EPOCH_LEN = epoch_len
