from src.config.config import Config
from src.helpers.general_helpers import create_path_if_not_exists

import argparse


class ScriptParamHandler:

    def __init__(self, *additional_arguments):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--batch", help="batch size")
        self.parser.add_argument("--epochs", help="number of epochs")
        self.parser.add_argument("--workers", help="number of DataLoader workers")
        self.parser.add_argument("--out", help="dir path for storing output")
        self.parser.add_argument("--trial", help="only train on <trial> percent of training data")
        self.parser.add_argument("--data", help="path to download or look for training data")
        self.parser.add_argument("--maxlen", help="max length for tokenization")
        self.parser.add_argument("--final", help="Evaluate final model as well as best model")
        self.parser.add_argument("--tokenizer", help="Path where tokenizers are saved and loaded")
        self.parser.add_argument("--models", help="Path where models are saved and loaded")
        for arg in additional_arguments:
            self.parser.add_argument(arg)
        self.args = self.parser.parse_args()

    def get_arg_value(self, arg_name):
        args = vars(self.args)
        if args.get(arg_name) is not None:
            return args.get(arg_name)
        else:
            return None

    def set_output_dir(self, config):
        if self.args.out is not None:
            config.OUTPUTDIR = self.args.out
        create_path_if_not_exists(config.OUTPUTDIR)
        print(f"Output directory: {config.OUTPUTDIR}")

    def set_data_workers(self, config):
        if self.args.workers is not None:
            config.DATA_WORKERS = int(self.args.workers)
        print(f"DataLoader workers: {config.DATA_WORKERS}")

    def set_batch_size(self, config):
        if self.args.batch is not None:
            config.BATCH_SIZE = int(self.args.batch)
        print(f"Batch size: {config.BATCH_SIZE}")

    def set_epochs(self, config):
        if self.args.epochs is not None:
            config.EPOCHS = int(self.args.epochs)
        print(f"Number of epochs: {config.EPOCHS}")

    def set_trial_run(self, config):
        if self.args.trial is not None:
            if self.args.trial in ("None", "null", "False", "false"):
                config.TRIAL_RUN = 0
            else:
                config.TRIAL_RUN = float(self.args.trial)
        else:
            config.TRIAL_RUN = 0
        print(f"Trial run: {config.TRIAL_RUN}")

    def set_data_path(self, config):
        if self.args.data is not None:
            config.DATA_PATH = self.args.data
        print(f"Data path: {config.DATA_PATH}")

    def set_max_length(self, config):
        if self.args.maxlen is not None:
            config.MAX_LEN = int(self.args.maxlen)
        print(f"Max length: {config.MAX_LEN}")

    def set_final_eval(self, config):
        if self.args.final is not None:
            if self.args.final in ("true", "True"):
                config.FINAL_EVAL = True
        else:
            config.FINAL_EVAL = False
        print(f"Eval final model: {config.FINAL_EVAL}")

    def set_tokenizer_path(self, config):
        if self.args.tokenizer is not None:
            config.TOKENIZER_PATH = self.args.tokenizer
        print(f"Tokenizer path: {config.TOKENIZER_PATH}")

    def set_model_path(self, config):
        if self.args.models is not None:
            config.MODEL_PATH = self.args.models
        print(f"Model path: {config.MODEL_PATH}")


    def set_config_from_script_params(self, config):
        print("\nSETTING GLOBAL PARAMS:")
        print("-"*40)
        print(f"\nRunning in: {config.PYTHON_DIR}")
        self.set_output_dir(config)
        self.set_model_path(config)
        self.set_tokenizer_path(config)
        self.set_data_path(config)
        self.set_max_length(config)
        self.set_batch_size(config)
        self.set_trial_run(config)
        self.set_epochs(config)
        self.set_data_workers(config)
        self.set_final_eval(config)

