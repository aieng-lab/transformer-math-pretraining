from typing import List

from comet_ml import Experiment

from src.pretraining_methods.nsp_like.NSP.prepare import only_nsp, only_ffir, only_nfir
from src.pretraining_methods.nsp_like.IR.prepare import only_ir
from src.config.config import Config
from src.config.params import ScriptParamHandler
from src.helpers.Timer import Timer
from src.data_sets.PretrainingDataset import PretrainingDataset
from src.data_sets.text_data import BooksData, TrainingDataSet, WikipediaData, OpenWebData
from src.data_sets.text_data.DataEnum import DataEnum
from src.tokenizer.tokenizer import Tokenizer
from src.pretraining_methods.combinations.MLM_NSP.prepare import mlm_plus_nsp
from src.pretraining_methods.combinations.MLM_SOP.prepare import mlm_plus_sop
from src.pretraining_methods.combinations.MLM_SDT.prepare import mlm_plus_sdt
from src.pretraining_methods.combinations.MLM_SRT.prepare import mlm_plus_srt
from src.pretraining_methods.mlm_like.MLM.prepare import only_mlm, only_math_text_mlm, only_math_mlm
from src.pretraining_methods.combinations.SMO_NSP.prepare import smo_plus_nsp
from src.pretraining_methods.combinations.MAC_NSP.prepare import mac_plus_nsp
from src.pretraining_methods.combinations.WSO_NSP.prepare import wso_plus_nsp
from src.pretraining_methods.combinations.SMO_SDT.prepare import smo_plus_sdt
from src.pretraining_methods.combinations.WSO_SDT.prepare import wso_plus_sdt
from src.pretraining_methods.combinations.SMO_SRT.prepare import smo_plus_srt
from src.pretraining_methods.combinations.WSO_SRT.prepare import wso_plus_srt
from src.pretraining_methods.combinations.PROP_MLM.prepare import prop_plus_mlm
from src.pretraining_methods.combinations.MAC_SDT.prepare import mac_plus_sdt
from src.pretraining_methods.combinations.MAC_SRT.prepare import mac_plus_srt
from src.models.pretraining_model import PretrainingModel
from src.training.Trainer import Trainer, IRTrainer
from src.pretraining_methods.nsp_like.IR import training as ir_training
from src.pretraining_methods.combinations.MLM_NSP import training as mlm_nsp_training
from src.pretraining_methods.combinations.MLM_SOP import training as mlm_sop_training
from src.pretraining_methods.combinations.MLM_SDT import training as mlm_sdt_training
from src.pretraining_methods.combinations.MLM_SRT import training as mlm_srt_training
from src.pretraining_methods.mlm_like.MLM import training as mlm_training
from src.pretraining_methods.combinations.SMO_NSP import training as smo_nsp_training
from src.pretraining_methods.combinations.MAC_NSP import training as mac_nsp_training
from src.pretraining_methods.combinations.WSO_NSP import training as wso_nsp_training
from src.pretraining_methods.combinations.SMO_SDT import training as smo_sdt_training
from src.pretraining_methods.combinations.WSO_SDT import training as wso_sdt_training
from src.pretraining_methods.combinations.SMO_SRT import training as smo_srt_training
from src.pretraining_methods.combinations.WSO_SRT import training as wso_srt_training
from src.pretraining_methods.combinations.PROP_MLM import training as prop_mlm_training
from src.pretraining_methods.combinations.MAC_SDT import training as mac_sdt_training
from src.pretraining_methods.combinations.MAC_SRT import training as mac_srt_training
from src.pretraining_methods.nsp_like.NSP import training as nsp_training
from src.tracking.comet_ml import create_experiment, get_exp_name
from src.pretraining_methods.Objectives import Objectives
from src.optimization import bert_original, prop
from execution.training.util import create_model

import torch
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from transformers import BertTokenizerFast, BertModel, BertConfig, AutoTokenizer, AutoModel
import os
from accelerate import Accelerator
from accelerate.utils import set_seed
from datetime import datetime
import pause
import time
import random
import json


class Executor:

    def __init__(self):
        self.config = Config()
        self.timer = Timer()
        self.accelerator: Accelerator = None

    def initialise(self, param_handler: ScriptParamHandler):
        param_handler.set_config_from_script_params(self.config)
        if self.config.MAX_THREADS:
            print(f"Setting threads to {self.config.MAX_THREADS}")
            torch.set_num_threads(self.config.MAX_THREADS)
        set_seed(42)
        if self.config.NUM_GPUS:
            print(f"NCCL version: {torch.cuda.nccl.version()}")

        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"CUDA version: {torch.version.cuda}", flush=True)

        # self.set_device_to_gpu()
        # self.get_cuda_information()

    def initialise_accelerate(self):
        self.accelerator = Accelerator(device_placement=True, gradient_accumulation_steps=self.config.ACC_STEPS,
                                       dispatch_batches=False)
        self.set_accelerate_device(self.accelerator)

    def get_cuda_information(self):
        print("\nCUDA INFO:")
        print("-" * 40)
        print("Cuda version:")
        print(torch.version.cuda)
        print("\nCuda available:")
        print(torch.cuda.is_available())
        print("\nCuda device count:")
        print(torch.cuda.device_count())

    def set_device_to_gpu(self):
        if torch.cuda.is_available():
            self.config.DEVICE = torch.device("cuda")
        print(f"Device: {self.config.DEVICE}")

    def set_accelerate_device(self, accelerator):
        self.config.DEVICE = accelerator.device

    def create_data_loader(self, data, tokenizer, max_len, processing_func, objectives, validation_data=False):
        torch.set_num_threads(1)
        print(f"Creating pretraining dataset")
        ds = PretrainingDataset(objectives, data, tokenizer, max_len, processing_func, batch_size=self.config.BATCH_SIZE, validation_data=validation_data)
        print("created Pretraining Dataset")
        num_workers = self.config.DATA_WORKERS
        if Objectives.MAC in objectives:
            time_out = 3600
        else:
            time_out = 500
        shuffle = len(objectives) == 1
        if num_workers > 1:
            data_loader = DataLoader(ds,
                                     batch_size=self.config.BATCH_SIZE,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     multiprocessing_context=torch.multiprocessing.get_context('spawn'),
                                     timeout=time_out,
                                     pin_memory=False,
                                     drop_last=True)
        else:
            data_loader = DataLoader(ds,
                                     batch_size=self.
                                     config.BATCH_SIZE,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True)
        return data_loader

    def get_training_data(self, datasets: List['DataEnum'], file_names=None, wikiconfig=None, percentages=None,
                          column_names=None, local=True):
        if column_names is None:
            column_names = ["text"]
        train_data_path = os.path.join(self.config.DATA_PATH)
        ds_list = []
        for i, dataset in enumerate(datasets):
            if dataset == DataEnum.BOOKCORPUS:
                base_path = os.path.join(train_data_path, "books")
                ds = BooksData.BooksData(save_dir=base_path,
                                         local_file_name=file_names[i] if file_names else None)
                ds.load_data(percentage=percentages[i] if percentages else None, try_local=True)
                ds.data = ds.data.remove_columns([name for name in ds.data.column_names if name not in column_names])
                ds_list.append(ds)
            elif dataset == DataEnum.WIKIPEDIA:
                base_path = os.path.join(train_data_path, "wiki")
                ds = WikipediaData.WikipediaData(save_dir=base_path,
                                                 local_file_name=file_names[i] if file_names else None,
                                                 identifier=wikiconfig)
                ds.load_data(percentage=percentages[i] if percentages else None, try_local=True)
                ds.data = ds.data.remove_columns([name for name in ds.data.column_names if name not in column_names])
                ds_list.append(ds)
            elif dataset == DataEnum.OPENWEBTEXT:
                base_path = os.path.join(train_data_path, "openwebtext")
                ds = OpenWebData.OpenWebData(save_dir=base_path, local_file_name=file_names[i] if file_names else None,
                                             local=local)
                ds.load_data(percentage=percentages[i] if percentages else None, try_local=True)
                ds.data = ds.data.remove_columns([name for name in ds.data.column_names if name not in column_names])
                ds_list.append(ds)
        return ds_list

    def prepare_training_data(self, datasets: List['DataEnum'], file_names=None, wikiconfig=None, percentages=None,
                              column_names=None, validation_size=0.1, local=True):
        if column_names is None:
            column_names = ["text"]
        ds_list = self.get_training_data(datasets, file_names=file_names, wikiconfig=wikiconfig,
                                         percentages=percentages, column_names=column_names, local=local)
        training_data, validation_data = self.create_test_and_val_data(ds_list, validation_size)
        return training_data, validation_data

    def create_test_and_val_data(self, datasets: List['TrainingDataSet.TrainingDataSet'], val_size):
        self.timer.reset()
        self.timer.start()
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
        self.timer.stop()
        print("\n Time for creating dataset splits:")
        self.timer.print_elapsed()
        return training_data, validation_data

    def get_huggingface_tokenizer(self, base_name):
        tokenizer = Tokenizer(self.config, base_name=base_name)
        tokenizer.load_from_huggingface()
        return tokenizer.get_tokenizer()

    def get_processing_func(self, objectives: List['Objectives'], one_by_one=True):
        if not one_by_one:
            mapping = {
                Objectives.MFM: only_math_mlm,
                Objectives.MTM: only_math_text_mlm,
                Objectives.NFIR: only_nfir,
                Objectives.FFIR: only_ffir
            }

            return {o.name: proc_func for o, proc_func in mapping.items() if o in objectives}


        if Objectives.MLM in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            return mlm_plus_nsp
        elif Objectives.MTM in objectives and len(objectives) == 1:
            return only_math_text_mlm
        elif any(o in objectives for o in [Objectives.MLM, Objectives.MFM, Objectives.MTM]) and len(objectives) == 1:
            if Objectives.MLM in objectives:
                return only_mlm
            elif Objectives.MFM in objectives:
                return only_math_mlm
            else: # MLM_MATH_TEXT
                return only_math_text_mlm
        elif Objectives.NSP in objectives and len(objectives) == 1:
            return only_nsp
        elif Objectives.NFIR in objectives and len(objectives) == 1:
            return only_nfir
        elif Objectives.FFIR in objectives and len(objectives) == 1:
            return only_ffir
        elif Objectives.MLM in objectives and Objectives.SOP in objectives and len(objectives) == 2:
            return mlm_plus_sop
        elif Objectives.MLM in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            return mlm_plus_sdt
        elif Objectives.MLM in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            return mlm_plus_srt
        elif Objectives.SMO in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            return smo_plus_nsp
        elif Objectives.MAC in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            return mac_plus_nsp
        elif Objectives.WSO in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            return wso_plus_nsp
        elif Objectives.SMO in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            return smo_plus_sdt
        elif Objectives.WSO in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            return wso_plus_sdt
        elif Objectives.SMO in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            return smo_plus_srt
        elif Objectives.WSO in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            return wso_plus_srt
        elif Objectives.PROP in objectives and Objectives.MLM in objectives and len(objectives) == 2:
            return prop_plus_mlm
        elif Objectives.MAC in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            return mac_plus_sdt
        elif Objectives.MAC in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            return mac_plus_srt
        else:
            return None

    def get_input_target_funcs(self, objectives: List['Objectives'], one_by_one=False):
        if not one_by_one:
            input_mapping = {
                 Objectives.MFM: mlm_training.get_model_inputs_MLM_MATH,
                 Objectives.MTM: mlm_training.get_model_inputs_MLM_MATH_TEXT,
                 Objectives.NFIR: nsp_training.get_model_inputs_nfir,  # same model inputs as for nsp
                 Objectives.FFIR: nsp_training.get_model_inputs_ffir  # same model inputs as for nsp
             }

            target_mapping = {
                Objectives.MFM: mlm_training.get_targets_MLM_MATH,
                Objectives.MTM: mlm_training.get_model_inputs_MLM_MATH_TEXT,
                Objectives.NFIR: nsp_training.get_targets_nfir,
                Objectives.FFIR: nsp_training.get_targets_ffir
            }

            return {o.name: proc_func for o, proc_func in input_mapping.items() if o in objectives}, {o.name: proc_func for o, proc_func in target_mapping.items() if o in objectives}

        if Objectives.MLM in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            input_func = mlm_nsp_training.get_model_inputs
            target_func = mlm_nsp_training.get_targets
            return input_func, target_func
        elif any(o in objectives for o in [Objectives.MLM, Objectives.MFM, Objectives.MTM]) and len(objectives) == 1:
            if Objectives.MFM in objectives:
                input_func = mlm_training.get_model_inputs_MLM_MATH
                target_func = mlm_training.get_targets_MLM_MATH
            elif Objectives.MTM in objectives:
                input_func = mlm_training.get_model_inputs_MLM_MATH_TEXT
                target_func = mlm_training.get_targets_MLM_MATH_TEXT
            else:
                input_func = mlm_training.get_model_inputs
                target_func = mlm_training.get_targets
            return input_func, target_func
        elif Objectives.MLM in objectives and Objectives.SOP in objectives and len(objectives) == 2:
            input_func = mlm_sop_training.get_model_inputs
            target_func = mlm_sop_training.get_targets
            return input_func, target_func
        elif Objectives.MLM in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            input_func = mlm_sdt_training.get_model_inputs
            target_func = mlm_sdt_training.get_targets
            return input_func, target_func
        elif Objectives.MLM in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            input_func = mlm_srt_training.get_model_inputs
            target_func = mlm_srt_training.get_targets
            return input_func, target_func
        elif Objectives.SMO in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            input_func = smo_nsp_training.get_model_inputs
            target_func = smo_nsp_training.get_targets
            return input_func, target_func
        elif Objectives.MAC in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            input_func = mac_nsp_training.get_model_inputs
            target_func = mac_nsp_training.get_targets
            return input_func, target_func
        elif Objectives.MAC in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            input_func = mac_sdt_training.get_model_inputs
            target_func = mac_sdt_training.get_targets
            return input_func, target_func
        elif Objectives.MAC in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            input_func = mac_srt_training.get_model_inputs
            target_func = mac_srt_training.get_targets
            return input_func, target_func
        elif Objectives.WSO in objectives and Objectives.NSP in objectives and len(objectives) == 2:
            input_func = wso_nsp_training.get_model_inputs
            target_func = wso_nsp_training.get_targets
            return input_func, target_func
        elif Objectives.SMO in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            input_func = smo_sdt_training.get_model_inputs
            target_func = smo_sdt_training.get_targets
            return input_func, target_func
        elif Objectives.WSO in objectives and Objectives.SDT in objectives and len(objectives) == 2:
            input_func = wso_sdt_training.get_model_inputs
            target_func = wso_sdt_training.get_targets
            return input_func, target_func
        elif Objectives.SMO in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            input_func = smo_srt_training.get_model_inputs
            target_func = smo_srt_training.get_targets
            return input_func, target_func
        elif Objectives.WSO in objectives and Objectives.SRT in objectives and len(objectives) == 2:
            input_func = wso_srt_training.get_model_inputs
            target_func = wso_srt_training.get_targets
            return input_func, target_func
        elif Objectives.PROP in objectives and Objectives.MLM in objectives and len(objectives) == 2:
            input_func = prop_mlm_training.get_model_inputs
            target_func = prop_mlm_training.get_targets
            return input_func, target_func
        elif Objectives.NSP in objectives:
            input_func = nsp_training.get_model_inputs
            target_func = nsp_training.get_targets
            return input_func, target_func
        elif Objectives.FFIR in objectives:
            input_func = nsp_training.get_model_inputs_ffir
            target_func = nsp_training.get_targets_ffir
            return input_func, target_func
        elif Objectives.NFIR in objectives:
            input_func = nsp_training.get_model_inputs_nfir
            target_func = nsp_training.get_targets_nfir
            return input_func, target_func
        else:
            raise ValueError("Unknown combinations of objectives: %s" % objectives)

    def get_optimizer_and_scheduler(self, objectives: List['Objectives'], model, train_steps):
        # todo: maybe do not create a different optimizer and scheduler for every combination, maybe restrict to only changing the parameters like learning rate, warmup, ...
        # todo: Maybe also decide that we should always use the same optimizer with the same parameters???
        if set(objectives).issubset(
                {Objectives.MLM, Objectives.NSP, Objectives.MFM, Objectives.MTM, Objectives.NFIR,
                 Objectives.FFIR, Objectives.SOP, Objectives.SDT, Objectives.SRT, Objectives.SMO,
                 Objectives.MAC, Objectives.WSO}) and not self.config.RETRAIN:
            optimizer = bert_original.get_optimizer(model, self.config)
            scheduler, warmup = bert_original.get_scheduler(optimizer, train_steps, self.config)
            return optimizer, scheduler, warmup
        elif Objectives.PROP in objectives or self.config.RETRAIN:
            optimizer = prop.get_optimizer(model, self.config)
            scheduler, warmup = prop.get_scheduler(optimizer, train_steps, self.config)
            return optimizer, scheduler, warmup
        else:
            return None

    def get_comet_project_name(self, objectives, taurus=False, retrain=False):
        if retrain:
            project_name = "BERT Retraining"
        else:
            project_name = "BERT Pretraining"
        if taurus:
            project_name += " TAURUS"
        for objective in objectives:
            project_name += f" {objective.name}"
        return project_name

    def prepare_pretraining(self, objectives: List['Objectives'], ds_train, ds_val, tokenizer: BertTokenizerFast,
                            taurus=False, job_id=None, base_bert=None, path=None, one_by_one=True):
        deberta = False

        if objectives is None:
            objectives = [Objectives.MLM, Objectives.NSP]



        # Method that coordinates calculating loss and preparing inputs for (combination of) pretraining methods
        pretraining_func = self.get_processing_func(objectives, one_by_one=one_by_one)

        # n_train = len(ds_train)
        # n_val = len(ds_val)
        # train_steps = math.ceil((n_train / self.config.BATCH_SIZE) * self.config.EPOCHS)

        vocab_size = len(list(tokenizer.get_vocab().keys()))

        if Objectives.PROP in objectives or self.config.RETRAIN:
            path_to_pretrained_model = self.config.PRETRAINED_MODEL_PATH

            model = create_model(path_to_pretrained_model)
            if base_bert:
                if self.accelerator.is_main_process:
                    print("Use base bert: %s" % base_bert)
                state_dict = torch.load(base_bert)
                state_dict = {k.removeprefix('bert.'): v for k, v in state_dict.items() if not k.startswith('objectives')}
                model.load_state_dict(state_dict)

            model = PretrainingModel(objectives, vocab_size, pretrained_bert=model)
            deberta = 'deberta' in str(model.bert_config)
        else:
            model = PretrainingModel(objectives, vocab_size)

        train_loader = self.create_data_loader(ds_train, tokenizer, self.config.MAX_LEN, pretraining_func, objectives)
        val_loader = self.create_data_loader(ds_val, tokenizer, self.config.MAX_LEN, pretraining_func, objectives, validation_data=True)

        self.config.determine_epoch_length(len(train_loader.dataset))

        train_steps = self.config.determine_train_steps(len(train_loader))
        optimizer, scheduler, warmup_steps = self.get_optimizer_and_scheduler(objectives, model, train_steps)

        input_func, target_func = self.get_input_target_funcs(objectives)

        if self.accelerator.is_main_process:
            # Comet-ML experiment creation
            project_name = self.get_comet_project_name(objectives, taurus=taurus, retrain=self.config.RETRAIN)
            experiment_name = get_exp_name(run_number=self.config.RUN_NUMBER, one_by_one=one_by_one, steps=train_steps, deberta=deberta)
            experiment = create_experiment(project_name, experiment_name, self.config.COMET_API_KEY)
            self.log_hyper_params(experiment)
            if job_id and experiment:
                experiment.log_parameter("JOB_ID", job_id)
        else:
            experiment = None

        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(model, optimizer, train_loader,
                                                                                         val_loader, scheduler)

        if self.accelerator.is_main_process and experiment:
            experiment.log_parameter("WARMUP_STEPS", warmup_steps)

        trainer = Trainer(self.config, model, optimizer, scheduler, train_loader, val_loader, experiment,
                          self.accelerator, path=path)
        trainer.set_model_input_extractor(input_func)
        trainer.set_target_extractor(target_func)

        if self.accelerator.is_main_process:
            print(self.config)

        return trainer

    def prepare_resume_pretraining(self, objectives: List['Objectives'], ds_train, ds_val, tokenizer: BertTokenizerFast,
                                   taurus=False, job_id=None):

        seed = random.randrange(100)
        set_seed(seed)

        base_checkpoint_path = os.path.join(self.config.OUTPUTDIR, "training_checkpoint")
        model_checkpoint_path = os.path.join(base_checkpoint_path, "latest_model")
        optimizer_checkpoint_path = os.path.join(base_checkpoint_path, "optimization")

        if objectives is None:
            objectives = [Objectives.MLM, Objectives.NSP]

        self.config.determine_epoch_length(len(ds_train))

        # Method that coordinates calculating loss and preparing inputs for (combination of) pretraining methods
        pretraining_func = self.get_processing_func(objectives)

        # n_train = len(ds_train)
        # n_val = len(ds_val)
        # train_steps = math.ceil((n_train / self.config.BATCH_SIZE) * self.config.EPOCHS)

        vocab_size = len(list(tokenizer.get_vocab().keys()))

        model = PretrainingModel(objectives, vocab_size)
        state_dict = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        train_loader = self.create_data_loader(ds_train, tokenizer, self.config.MAX_LEN, pretraining_func,
                                               objectives)
        val_loader = self.create_data_loader(ds_val, tokenizer, self.config.MAX_LEN, pretraining_func, objectives, validation_data=True)

        optimizer_states = torch.load(optimizer_checkpoint_path)

        original_train_steps = self.config.determine_train_steps(len(train_loader))
        train_steps = original_train_steps - optimizer_states["completed_steps"]

        optimizer, scheduler, warmup_steps = self.get_optimizer_and_scheduler(objectives, model, train_steps)
        optimizer.load_state_dict(optimizer_states["optimizer"])
        scheduler.load_state_dict(optimizer_states["scheduler"])
        if self.config.NUM_GPUS:
            self.accelerator.scaler.load_state_dict(optimizer_states["scaler"])
        if optimizer_states["completed_steps"] < warmup_steps:
            warmup_steps = warmup_steps - optimizer_states["completed_steps"]
        else:
            warmup_steps = 0

        input_func, target_func = self.get_input_target_funcs(objectives)

        if self.accelerator.is_main_process:
            # Comet-ML experiment creation
            project_name = self.get_comet_project_name(objectives, taurus=taurus)
            experiment_name = get_exp_name(run_number=self.config.RUN_NUMBER)
            experiment = create_experiment(project_name, experiment_name, self.config.COMET_API_KEY)
            self.log_hyper_params(experiment)
            if job_id and experiment:
                experiment.log_parameter("JOB_ID", job_id)
        else:
            experiment = None

        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(model, optimizer, train_loader,
                                                                                         val_loader, scheduler)

        if self.accelerator.is_main_process and experiment:
            experiment.log_parameter("WARMUP_STEPS", warmup_steps)
            experiment.log_parameter("REMAINING_STEPS", train_steps)

        trainer = Trainer(self.config, model, optimizer, scheduler, train_loader, val_loader, experiment,
                          self.accelerator)
        trainer.set_model_input_extractor(input_func)
        trainer.set_target_extractor(target_func)

        return trainer

    def log_hyper_params(self, experiment: Experiment):
        if experiment:
            hyper_params = self.config.get_fields_dict()
            experiment.log_parameters(hyper_params)

    def data_set_test_run(self, dataset):
        # set_seed(42)
        timer = Timer()
        timer.start()
        for i in range(len(dataset)):
            item = dataset[i]
            input = item.get("bert_input")
            if i % 100000 == 0:
                print(f"Finished {i} / {len(dataset) - 1}")
            if i == 100000:
                timer.stop()
                print(f"Took {timer.get_elapsed_minutes()} minutes.")

    def data_loader_test_run(self, dataloader):
        # monitoring_thread = start_monitoring(seconds_frozen=100)
        wait_until = datetime(2023, 3, 19, 22, 15, 0, 0)
        print(datetime.now())
        os.environ["DEBUG_DS"] = "False"
        set_seed(42)
        timer = Timer()
        if self.accelerator.is_main_process:
            print(f"Testing data loader")
        i = 0
        timer.start()
        for d in dataloader:
            model_inputs = d["bert_input"]
            self.accelerator.wait_for_everyone()
            if i % 500 == 0 or i == 1 or i == 10 or i == 20 or i == 40 or i == 60 or i == 93917 or i == 93918 or i == 93919:
                if self.accelerator.is_main_process:
                    print(f"finished batch {i}")
                    if i == 500:
                        timer.stop()
                        print(f"Took {timer.get_elapsed_minutes()} minutes.")
            if i == 80838:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    print(f"About to initiate pause, time: {datetime.now()}")
                pause.until(wait_until)
                if self.accelerator.is_main_process:
                    print("resuming program")
            if i == 93918:
                time.sleep(10)
                print("----------------------------------------------------------", flush=True)
                time.sleep(5)

            i += 1

        # monitoring_thread.stop()

class IRExecutor(Executor):

    def get_processing_func(self, *args, **kwargs):
        return only_ir

    def get_input_target_funcs(self, *args, **kwargs):
        input_func = ir_training.get_model_inputs
        target_func = ir_training.get_targets
        return input_func, target_func

    def get_comet_project_name(self, objectives, taurus=False, retrain=False):
        if retrain:
            project_name = "BERT Finetuning"
        else:
            project_name = "BERT Finetuning"
        if taurus:
            project_name += " TAURUS"
        for objective in objectives:
            project_name += f" {objective.name}"
        return project_name

    def prepare_pretraining(self, objectives, ds_train, ds_val, tokenizer: BertTokenizerFast,
                            taurus=False, job_id=None, base_bert=None, path=None, one_by_one=True, ds_test=None,
                            eval_output=None, run_number=None):
        if ds_test is None:
            raise ValueError

        deberta = False

        objectives = [Objectives.IR]

        # Method that coordinates calculating loss and preparing inputs for (combination of) pretraining methods
        pretraining_func = self.get_processing_func(objectives, one_by_one=one_by_one)

        vocab_size = len(list(tokenizer.get_vocab().keys()))

        if Objectives.PROP in objectives or self.config.RETRAIN:
            path_to_pretrained_model = self.config.PRETRAINED_MODEL_PATH

            model = create_model(path_to_pretrained_model)
            if base_bert:
                if self.accelerator.is_main_process:
                    print("Use base bert: %s" % base_bert)
                state_dict = torch.load(base_bert)
                state_dict = {k.removeprefix('bert.'): v for k, v in state_dict.items() if not k.startswith('objectives')}
                model.load_state_dict(state_dict)

            model = PretrainingModel(objectives, vocab_size, pretrained_bert=model)
            deberta = 'deberta' in str(model.bert_config)
        else:
            model = PretrainingModel(objectives, vocab_size)

        train_loader = self.create_data_loader(ds_train, tokenizer, self.config.MAX_LEN, pretraining_func, objectives)
        val_loader = self.create_data_loader(ds_val, tokenizer, self.config.MAX_LEN, pretraining_func, objectives, validation_data=True)
        test_loader = self.create_data_loader(ds_test, tokenizer, self.config.MAX_LEN, pretraining_func, objectives, validation_data=True)

        self.config.determine_epoch_length(len(train_loader.dataset))

        train_steps = self.config.determine_train_steps(len(train_loader))
        optimizer, scheduler, warmup_steps = self.get_optimizer_and_scheduler(objectives, model, train_steps)

        input_func, target_func = self.get_input_target_funcs(objectives)

        if self.accelerator.is_main_process:
            # Comet-ML experiment creation
            project_name = self.get_comet_project_name(objectives, taurus=taurus, retrain=self.config.RETRAIN)
            experiment_name = get_exp_name(run_number=self.config.RUN_NUMBER, one_by_one=one_by_one, steps=train_steps, deberta=deberta)
            experiment = create_experiment(project_name, experiment_name, self.config.COMET_API_KEY)
            self.log_hyper_params(experiment)
            if job_id and experiment:
                experiment.log_parameter("JOB_ID", job_id)
        else:
            experiment = None

        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(model, optimizer, train_loader,
                                                                                         val_loader, scheduler)

        if self.accelerator.is_main_process and experiment:
            experiment.log_parameter("WARMUP_STEPS", warmup_steps)

        trainer = IRTrainer(self.config, model, optimizer, scheduler, train_loader, val_loader, experiment,
                          self.accelerator, path=path, test_loader=test_loader,
                            eval_output=eval_output, run_number=run_number)
        trainer.set_model_input_extractor(input_func)
        trainer.set_target_extractor(target_func)

        if self.accelerator.is_main_process:
            print(self.config)

        return trainer


if __name__ == "__main__":
    params = ScriptParamHandler()
    runner = Executor()
    runner.initialise(params)
    tokenizer = runner.get_huggingface_tokenizer("bert-base-cased")
    trainer, n_train, n_val = runner.prepare_pretraining(["MLM", "NSP"], "wiki/20220301.en", "books", tokenizer)
    trainer.train(n_train, n_val)
