from src.config.config import Config
from src.helpers.general_helpers import create_path_if_not_exists
from src.pretraining_methods.Objectives import Objectives

import argparse


class ParameterError(Exception):
    pass


class ScriptParamHandler:

    def __init__(self, *additional_arguments):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--batch", help="batch size", type=self.check_null)
        self.parser.add_argument("--epochs", help="number of epochs", type=self.check_null)
        self.parser.add_argument("--workers", help="number of DataLoader workers", type=self.check_null)
        self.parser.add_argument("--out", help="dir path for storing output", type=self.check_null)
        self.parser.add_argument("--trial", help="only train on <trial> percent of training data", type=self.check_null)
        self.parser.add_argument("--data", help="path to download or look for training data", type=self.check_null)
        self.parser.add_argument("--maxlen", help="max length for tokenization", type=self.check_null)
        self.parser.add_argument("--models", help="Path where models are saved and loaded", type=self.check_null)
        self.parser.add_argument("--acc_steps", help="Gradient accumulation steps", type=self.check_null)
        self.parser.add_argument("--run_number", help="Which slurm script number is executed", type=self.check_null)
        self.parser.add_argument("--num_gpus", help="If training, on how many GPUs", type=self.check_null)
        self.parser.add_argument("--lr", help="learning rate", type=self.check_null)
        self.parser.add_argument("--warmup", help="percentage of warmup steps (0.1 = 10%)", type=self.check_null)
        self.parser.add_argument("--epsilon", help="AdamW hyperparameter", type=self.check_null)
        self.parser.add_argument("--beta2", help="AdamW hyperparameter", type=self.check_null)
        self.parser.add_argument("--opt_steps", help="Total amount of training steps", type=self.check_null)
        self.parser.add_argument("--interval_len", help="Intervals in which to perform validation",
                                 type=self.check_null)
        self.parser.add_argument("--comet_api_key", help="Api key for comet ML experiment tracking",
                                 type=self.check_null)
        self.parser.add_argument("--pretrained_model_path",
                                 help="for PROP where to find model from which to initialize", type=self.check_null)
        self.parser.add_argument("--pretrained_objectives",
                                 help="for PROP, with which objectives the model from which to initialize was trained",
                                 type=self.check_null)
        self.parser.add_argument("--retrain",
                                 help="Perform retraining of pre-trained model or not",
                                 type=self.check_null)
        for arg in additional_arguments:
            self.parser.add_argument(arg, type=self.check_null)
        self.args = self.parser.parse_args()
        print(self.args)

    def check_null(self, input):
        if input == "null":
            return None
        elif input in ("True", "true"):
            return True
        elif input in ("False", "false"):
            return False
        else:
            return input

    def print_args(self):
        args = vars(self.args)
        for arg in args:
            print(f"{arg}: {args.get(arg)}")

    def get_arg_value(self, arg_name):
        args = vars(self.args)
        if args.get(arg_name) is not None:
            return args.get(arg_name)
        else:
            return None

    def set_output_dir(self, config, printing=True):
        if self.args.out is not None:
            config.OUTPUTDIR = self.args.out
        create_path_if_not_exists(config.OUTPUTDIR)
        if printing:
            print(f"Output directory: {config.OUTPUTDIR}")

    def set_data_workers(self, config, printing=True):
        if self.args.workers is not None:
            config.DATA_WORKERS = int(self.args.workers)
        if printing:
            print(f"DataLoader workers: {config.DATA_WORKERS}")

    def set_batch_size(self, config, printing=True):
        if self.args.batch is not None:
            config.BATCH_SIZE = int(self.args.batch)
        if printing:
            print(f"Batch size: {config.BATCH_SIZE}")

    def set_epochs(self, config, printing=True):
        if self.args.epochs is not None:
            config.EPOCHS = int(self.args.epochs)
        if printing:
            print(f"Number of epochs: {config.EPOCHS}")

    def set_trial_run(self, config, printing=True):
        if self.args.trial is not None:
            if self.args.trial in ("None", "null", "False", "false"):
                config.TRIAL_RUN = 0
            else:
                config.TRIAL_RUN = float(self.args.trial)
        else:
            config.TRIAL_RUN = 0
        if printing:
            print(f"Trial run: {config.TRIAL_RUN}")

    def set_data_path(self, config, printing=True):
        if self.args.data is not None:
            config.DATA_PATH = self.args.data
        if printing:
            print(f"Data path: {config.DATA_PATH}")

    def set_max_length(self, config, printing=True):
        if self.args.maxlen is not None:
            config.MAX_LEN = int(self.args.maxlen)
        if printing:
            print(f"Max length: {config.MAX_LEN}")

    def set_model_path(self, config, printing=True):
        if self.args.models is not None:
            config.MODEL_PATH = self.args.models
        if printing:
            print(f"Model path: {config.MODEL_PATH}")

    def set_acc_steps(self, config, printing=True):
        if self.args.acc_steps is not None:
            config.ACC_STEPS = int(self.args.acc_steps)
        if printing:
            print(f"Gradient accumulation steps: {config.ACC_STEPS}")

    def set_num_gpus(self, config, printing=True):
        if self.args.num_gpus is not None:
            config.NUM_GPUS = int(self.args.num_gpus)
        if printing:
            print(f"GPUs: {config.NUM_GPUS}")

    def set_run_number(self, config, printing=True):
        if self.args.run_number is not None:
            config.RUN_NUMBER = int(self.args.run_number)
        if printing:
            print(f"Run number: {config.RUN_NUMBER}")

    def set_learning_rate(self, config, printing=True):
        if self.args.lr is not None:
            config.LR = float(self.args.lr)
        if printing:
            print(f"Learning rate: {config.LR}")

    def set_warmup(self, config, printing=True):
        if self.args.warmup is not None:
            config.WARMUP = float(self.args.warmup)
        if printing:
            print(f"Warmup steps: {config.WARMUP}")

    def set_epsilon(self, config, printing=True):
        if self.args.epsilon is not None:
            config.EPSILON = float(self.args.epsilon)
        if printing:
            print(f"Epsilon: {config.EPSILON}")

    def set_beta2(self, config, printing=True):
        if self.args.beta2 is not None:
            config.BETA2 = float(self.args.beta2)
        if printing:
            print(f"Beta2: {config.BETA2}")

    def set_opt_steps(self, config, printing=True):
        if self.args.opt_steps is not None:
            config.OPTIMIZER_STEPS = int(self.args.opt_steps)
        if printing:
            print(f"Training steps: {config.OPTIMIZER_STEPS}")

    def set_interval_len(self, config, printing=True):
        if self.args.interval_len is not None:
            config.VALIDATE_INTERVAL = int(self.args.interval_len)
        if printing:
            print(f"Interval length: {config.VALIDATE_INTERVAL}")

    def set_api_key(self, config, printing=True):
        if vars(self.args).get("comet_api_key") is not None:
            config.COMET_API_KEY = self.args.comet_api_key
        if printing:
            print("Set comet api key.")

    def set_pretrained_model_path(self, config, printing=True):
        if vars(self.args).get("pretrained_model_path") is not None:
            config.PRETRAINED_MODEL_PATH = self.args.pretrained_model_path
            if printing:
                print(f"Pretrained model path: {config.PRETRAINED_MODEL_PATH}")

    def set_pretrained_objectives(self, config, printing=True):
        if vars(self.args).get("pretrained_objectives") is not None:
            objectives = []
            given_objectives = self.args.pretrained_objectives
            if not given_objectives:
                raise ParameterError("No pretraining objectives were given as parameter")
            given_objectives = given_objectives.split("_")
            valid_objectives = [member.value for member in Objectives]
            for g_objective in given_objectives:
                if g_objective.upper() not in valid_objectives:
                    raise ParameterError(f"Given objecctive {g_objective} is not valid.")
                for member in Objectives:
                    if g_objective.upper() == member.value:
                        objectives.append(member)
            config.PRETRAINED_OBJECTIVES = objectives

            if printing:
                print(f"Pretrained objectives: {config.PRETRAINED_OBJECTIVES}")

    def set_retrain(self, config, printing=True):
        if vars(self.args).get("retrain") is not None:
            config.RETRAIN = self.args.retrain
            if printing:
                print(f"Re-train: {config.RETRAIN}")


    def set_config_from_script_params(self, config, printing=True):
        if printing:
            print("\nSETTING GLOBAL PARAMS:")
            print("-" * 40)
            print(f"\nRunning in: {config.PYTHON_DIR}")
        self.set_output_dir(config, printing=printing)
        self.set_model_path(config, printing=printing)
        self.set_data_path(config, printing=printing)
        self.set_max_length(config, printing=printing)
        self.set_batch_size(config, printing=printing)
        self.set_trial_run(config, printing=printing)
        self.set_epochs(config, printing=printing)
        self.set_data_workers(config, printing=printing)
        self.set_acc_steps(config, printing=printing)
        self.set_run_number(config, printing=printing)
        self.set_learning_rate(config, printing=printing)
        self.set_num_gpus(config, printing=printing)
        self.set_epsilon(config, printing=printing)
        self.set_beta2(config, printing=printing)
        self.set_warmup(config, printing=printing)
        self.set_opt_steps(config, printing=printing)
        self.set_interval_len(config, printing=printing)
        self.set_api_key(config, printing=printing)
        self.set_pretrained_model_path(config, printing=printing)
        self.set_pretrained_objectives(config, printing=printing)
        self.set_retrain(config, printing=printing)


if __name__ == "__main__":
    config = Config()
    params = ScriptParamHandler()
    params.set_config_from_script_params(config, printing=False)
