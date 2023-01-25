from comet_ml import Experiment

from src.config.config import Config
from src.config.params import ScriptParamHandler
from src.helpers.timer import Timer
from src.pretraining_methods.pretraining_datasets.PretrainingDataset import PretrainingDataset
from src.training.training_data import BooksData, WikipediaData, TrainingDataSet
from src.training.training_data.DataEnum import DataEnum
from src.tokenizer.tokenizer import Tokenizer
from src.pretraining_methods.pretraining_funcs import mlm_plus_nsp
from src.models.pretraining_model import PretrainingModel
from src.training.Trainer import Trainer
from src.training.training_methods import combinations
from src.tracking.comet_ml import create_experiment, get_exp_name
from src.pretraining_methods.Objectives import Objectives

import torch
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from transformers import BertTokenizerFast
import os


class Executor:

    def __init__(self):
        self.config = Config()
        self.timer = Timer()

    def initialise(self, param_handler: ScriptParamHandler):
        param_handler.set_config_from_script_params(self.config)
        self.set_device_to_gpu()
        self.get_cuda_information()

    def get_cuda_information(self):
        print("\nCUDA INFO:")
        print("-" * 40)
        print("Cuda version:")
        print(torch.version.cuda)
        print("\nCuda availabe:")
        print(torch.cuda.is_available())
        print("\nCuda device count:")
        print(torch.cuda.device_count())

    def set_device_to_gpu(self):
        if torch.cuda.is_available():
            self.config.DEVICE = torch.device("cuda")
        print(f"Device: {self.config.DEVICE}")

    def create_data_loader(self, data, tokenizer, max_len, pretrain_method):
        ds = PretrainingDataset(data["text"], tokenizer, max_len, pretrain_method)
        return DataLoader(ds, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.DATA_WORKERS)

    def get_training_data(self, datasets: list['DataEnum'], file_names=None, wikiconfig=None, percentages=None,
                          column_names=None):
        if column_names is None:
            column_names = ["text"]
        train_data_path = os.path.join(self.config.DATA_PATH)
        ds_list = []
        for i, dataset in enumerate(datasets):
            if dataset == DataEnum.BOOKCORPUS:
                ds = BooksData.BooksData(save_dir=train_data_path,
                                         local_file_name=file_names[i] if file_names else None)
                ds.load_data(percentage=percentages[i] if percentages else None, try_local=True)
                ds.data = ds.data.remove_columns([name for name in ds.data.column_names if name not in column_names])
                ds_list.append(ds)
            elif dataset == DataEnum.WIKIPEDIA:
                ds = WikipediaData.WikipediaData(save_dir=train_data_path,
                                                 local_file_name=file_names[i] if file_names else None,
                                                 identifier=wikiconfig)
                ds.load_data(percentage=percentages[i] if percentages else None, try_local=True)
                ds.data = ds.data.remove_columns([name for name in ds.data.column_names if name not in column_names])
                ds_list.append(ds)
        return ds_list

    def prepare_training_data(self, datasets: list['DataEnum'], file_names=None, wikiconfig=None, percentages=None,
                              column_names=None, validation_size=0.1):
        if column_names is None:
            column_names = ["text"]
        ds_list = self.get_training_data(datasets, file_names=file_names, wikiconfig=wikiconfig,
                                         percentages=percentages, column_names=column_names)
        training_data, validation_data = self.create_test_and_val_data(ds_list, validation_size)
        return training_data, validation_data

    def create_test_and_val_data(self, datasets: list['TrainingDataSet.TrainingDataSet'], val_size):
        train_data = []
        val_data = []
        for dataset in datasets:
            split = dataset.data.train_test_split(test_size=val_size, shuffle=False)
            train_data.append(split.get("train"))
            val_data.append(split.get("test"))
        features_type = train_data[0].features.type
        for elem in train_data:
            assert (elem.features.type == features_type)
        training_data = concatenate_datasets(train_data)
        features_type = val_data[0].features.type
        for elem in val_data:
            assert (elem.features.type == features_type)
        validation_data = concatenate_datasets(val_data)
        return training_data, validation_data

    def get_huggingface_tokenizer(self, base_name, save=False):
        tokenizer = Tokenizer(self.config, f"hf_{base_name}")
        tokenizer.load_from_huggingface()
        if save:
            tokenizer_path = os.path.join(self.config.TOKENIZER_PATH, f"hf_{base_name}")
            tokenizer.save(tokenizer_path)
        return tokenizer.get_tokenizer()

    def get_custom_tokenizer(self, tokenizer_name, base_name, vocab_size=32000, local=True, special_tokens=None,
                             train_data=None, batch_size=10000, save=False):
        tokenizer = Tokenizer(self.config, tokenizer_name, base_name)
        tokenizer.set_training_data(train_data)
        tokenizer.load_custom(vocab_size=vocab_size, local=local, batch_size=batch_size)
        if special_tokens:
            tokenizer.customize_special_tokens(special_tokens)
        if save:
            tokenizer_path = os.path.join(self.config.TOKENIZER_PATH, tokenizer_name)
            tokenizer.save(tokenizer_path)
        return tokenizer.get_tokenizer()

    def get_pretraining_func(self, objectives: list['Objectives']):
        if Objectives.MLM in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            return mlm_plus_nsp
        else:
            return None

    def get_input_target_loss_funcs(self, objectives: list['Objectives']):
        if Objectives.MLM in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            input_func = combinations.mlm_plus_nsp__get_model_inputs
            target_func = combinations.mlm_plus_nsp__get_targets
            loss_func = combinations.mlm_plus_nsp__get_losses
            return input_func, target_func, loss_func
        else:
            return None

    def get_optimizer_and_scheduler(self, objectives: list['Objectives'], model, train_steps):
        if Objectives.MLM in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            optimizer = combinations.mlm_plus_nsp__get_optimizer(model)
            scheduler = combinations.mlm_plus_nsp__get_scheduler(optimizer, num_training_steps=train_steps)
            return optimizer, scheduler
        else:
            return None

    def get_comet_project_name(self, objectives):
        project_name = "BERT Pretraining"
        for objective in objectives:
            project_name += f" {objective.name}"
        return project_name

    def prepare_pretraining(self, objectives: list['Objectives'], ds_train, ds_val, tokenizer: BertTokenizerFast):
        if objectives is None:
            objectives = [Objectives.MLM, Objectives.NSP]

        # Method that coordinates calculating loss and preparing inputs for (combination of) pretraining methods
        pretraining_func = self.get_pretraining_func(objectives)

        # Comet-ML experiment creation
        project_name = self.get_comet_project_name(objectives)
        experiment_name = get_exp_name()
        experiment = create_experiment(project_name, experiment_name)
        self.log_hyper_params(experiment)

        n_train = len(ds_train)
        n_val = len(ds_val)
        train_steps = n_train * self.config.EPOCHS
        vocab_size = len(list(tokenizer.get_vocab().keys()))

        model = PretrainingModel(objectives, vocab_size)
        train_loader = self.create_data_loader(ds_train, tokenizer, self.config.MAX_LEN, pretraining_func)
        val_loader = self.create_data_loader(ds_val, tokenizer, self.config.MAX_LEN, pretraining_func)
        optimizer, scheduler = self.get_optimizer_and_scheduler(objectives, model, train_steps)
        input_func, target_func, loss_func = self.get_input_target_loss_funcs(objectives)

        trainer = Trainer(self.config, model, optimizer, scheduler, train_loader, val_loader, experiment)
        trainer.set_model_input_extractor(input_func)
        trainer.set_target_extractor(target_func)
        trainer.set_loss_calculator(
            loss_func)  # todo: put method for getting correct method for calculating predictions and into Trainer class

        return trainer, n_train, n_val

    def log_hyper_params(self, experiment: Experiment):
        hyper_params = self.config.get_fields_dict()
        experiment.log_parameters(hyper_params)


if __name__ == "__main__":
    runner = Executor()
    runner.initialise()
    tokenizer = runner.get_huggingface_tokenizer("bert-base-cased")
    trainer, n_train, n_val = runner.prepare_pretraining(["MLM", "NSP"], tokenizer, "wiki/20220301.en", "books",
                                                         percentage=5)
    trainer.train(n_train, n_val)
