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
        self.TOKENIZER_PATH = os.path.join(self.OUTPUTDIR, "tokenizers")
        self.DATA_WORKERS = 4
        self.BATCH_SIZE = 16
        self.EPOCHS = 1
        self.TRIAL_RUN = False
        self.MAX_LEN = 128
        self.FINAL_EVAL = False

        self.vocab_size = None  # vocab_size is set after a tokenizer was selected with Tokenizer.get_tokenizer()

    def get_total_batches(self, data_len):
        return int(data_len / self.BATCH_SIZE) + 1

    def get_trial_batch_num(self, data_len):
        percentage = self.TRIAL_RUN / 100
        trial_part = math.ceil(data_len * percentage)
        return int(trial_part / self.BATCH_SIZE) + 1

    def get_fields_dict(self):
        return vars(self)
