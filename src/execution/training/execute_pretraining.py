import json
import time
import random

import comet_ml
from datetime import timedelta
from pathlib import Path

import humanize
import torch
from datasets import DatasetDict, Dataset, concatenate_datasets
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel

from execution.training.util import create_model
from src.execution.Executor import Executor
from src.config.params import ScriptParamHandler
from src.pretraining_methods.Objectives import Objectives
from src.config.params import ParameterError
from src.helpers.custom_helpers import extract_datasets
from datasets import load_dataset

import os


def get_pretraining_objectives(params):
    objectives = []
    given_objectives = params.get_arg_value("pretraining_obj")
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
    return objectives

def process_file(dataset_name, objective=None, batch_size=16):
    random.seed(42)
    print("Load file: %s" % dataset_name)
    try:
        result = load_dataset(dataset_name, cache_dir=None)
    except Exception:
        result = DatasetDict.load_from_disk(dataset_name)
    max_size = None
    if max_size:
        print("Truncate dataset to %s" % max_size)
        result = result.filter(lambda example, idx: idx < max_size, with_indices=True)

    # truncate all datasets to a multiple of the batch size
    for split_name, dataset in result.items():
        desired_length = (len(dataset) // batch_size) * batch_size

        # Truncate the dataset to the desired length
        if desired_length < len(dataset):
            print(f"Truncating dataset {dataset_name} {split_name} from {len(dataset)} to {desired_length}")
            truncated_dataset = dataset.select(range(desired_length))

            # Add the truncated dataset to the new DatasetDict
            result[split_name] = truncated_dataset
        else:
            print(f"Dataset {dataset_name} {split_name} is already a multiple of {batch_size}, no truncation needed")
            result[split_name] = dataset

    if objective in [Objectives.NFIR, Objectives.FFIR, Objectives.NSP]:
        new_dict = DatasetDict()
        # Define the number of splits (in this case, 10)

        for key in result:
            # Iterate through the splits
            dataset = result[key]

            trues = dataset.filter(lambda x: x['label'])
            new_len = (len(trues) // batch_size) * batch_size
            trues = trues.select(range(new_len))

            falses = dataset.filter(lambda x: not x['label'])
            n_split = max(1, int(len(falses) / len(trues)))

            end = len(falses)
            integers = list(range(0, end))
            random.shuffle(integers)
            subset_size = max(((end // n_split) // batch_size) * batch_size, batch_size)  # floor division
            subsets = [integers[i:i + subset_size] for i in range(0, len(integers), subset_size) if i + subset_size <= len(integers)]

            datasets = {}
            for i, subset in enumerate(subsets):
                false_ds = falses.select(subset)
                ds = concatenate_datasets([trues, false_ds])
                ds.shuffle(seed=42)
                datasets[i] = ds

            if key == 'test':
                # for test data, always the same data is used
                datasets = datasets[0]

            new_dict[key] = datasets
        result = new_dict
    return result

def run_single_file(param_handler, pretraining_task, file, taurus, tokenizer_name, output, max_len, base_bert=None):
    runner = Executor()
    runner.initialise(param_handler)
    runner.initialise_accelerate()

    runner.config.PRETRAINED_MODEL_PATH = tokenizer_name

    tokenizer = runner.get_huggingface_tokenizer(tokenizer_name)

    if isinstance(pretraining_task, list):
        # we are doing all the given pretraining tasks at once
        runner.config.MAX_LEN = 512
        if runner.config.OPTIMIZER_STEPS is not None:
            runner.config.OPTIMIZER_STEPS = len(pretraining_task) * runner.config.OPTIMIZER_STEPS
        pretraining_tasks = pretraining_task

        result = DatasetDict({'train': DatasetDict(), 'test': DatasetDict()})
        for f, objective in zip(file, pretraining_tasks):
            data = process_file(f, objective=objective)
            for key in ['train', 'test']:
                result[key][objective.name] = data[key]

        training_data = result['train']
        validation_data = result['test']
    else:
        if max_len in [None, 'auto']:
            print("Max Len is None, use default values")
            if pretraining_task == Objectives.MFM:
                runner.config.MAX_LEN = 128
            elif pretraining_task == Objectives.MTM:
                runner.config.MAX_LEN = 512
            elif pretraining_task == Objectives.NSP:
                runner.config.MAX_LEN = 256
            elif pretraining_task == Objectives.NFIR:
                runner.config.MAX_LEN = 256
            elif pretraining_task == Objectives.FFIR:
                runner.config.MAX_LEN = 512
        else:
            runner.config.MAX_LEN = int(max_len)

        result = process_file(file, objective=pretraining_task)
        training_data = result['train']
        validation_data = result['test']
        pretraining_tasks = [pretraining_task]

    trainer = runner.prepare_pretraining(pretraining_tasks, ds_train=training_data, ds_val=validation_data, tokenizer=tokenizer, taurus=taurus, base_bert=base_bert, path=output, one_by_one=one_by_one)
    if runner.accelerator.is_main_process and trainer.comet_ml_experiment:
        url = trainer.comet_ml_experiment.url
    else:
        url = None

    trainer.train(model_name=model_name)


    if model_name:
        return output + '/' + model_name + '_final_model.bin', url
    else:
        return output + '/final_model.bin', url

if __name__ == "__main__":
    print("\nPYTHON OUTPUT")
    print("-" * 60)
    print("\n")

    print(f"ENV HF_DATASETS_IN_MEMORY_MAX_SIZE: {os.environ.get('HF_DATASETS_IN_MEMORY_MAX_SIZE')} (in byte)")

    param_handler = ScriptParamHandler("--tokenizer_name", "--taurus", "--max_len",
                                       "--model_name", "--pretraining_obj", '--files', '--base_path', '--base_bert', '--one_by_one')

    pretraining_tasks = get_pretraining_objectives(param_handler)
    print(pretraining_tasks)
    model_name = param_handler.get_arg_value("model_name")
    print(model_name)

    files = param_handler.get_arg_value("files")
    if files is None or len(files) == 0 or files.lower() == 'auto':
        print("No files given! Use auto mode")
        files = []

        file_mapping = {
            Objectives.MFM: 'ddrg/math_formulas',
            Objectives.MTM: 'ddrg/math_text',
            Objectives.NFIR: 'ddrg/named_math_formulas',
            Objectives.FFIR: 'ddrg/math_formula_retrieval'
        }

        for obj in pretraining_tasks:
            if obj not in file_mapping:
                raise ParameterError(f"Objective {obj} is not valid for auto mode.")
            files.append(file_mapping[obj])
            print(f"File for {obj}: {file_mapping[obj]}")
    else:
        files = files.split(',')
    print(files)
    base_path = param_handler.get_arg_value('base_path')
    if base_path is None:
        base_path = "/root/transformer_pretraining/data"
    print(base_path)
    if not (base_path.endswith('\\') or base_path.endswith('/')):
        base_path += '/'

    taurus = param_handler.get_arg_value("taurus")


    one_by_one = param_handler.get_arg_value('one_by_one')
    if one_by_one:
        print("Run one by one")
    else:
        print("Run all objectives at once")

    base_bert = param_handler.get_arg_value('base_bert')
    print("Starting from %s" % base_bert)

    print(f"Pretraining on: {pretraining_tasks}")
    print(f"Model name: {model_name}")
    tokenizer_name = param_handler.get_arg_value("tokenizer_name")
    if tokenizer_name:
        print(f"Tokenizer: {tokenizer_name}")
    elif base_bert:
        tokenizer_name = base_bert
    else:
        tokenizer_name = f'{base_path}models/tokenized'

        print(f"Tokenizer: {tokenizer_name}")
        print("Tokenizer: from Huggingface")


    max_len = param_handler.get_arg_value('max_len')
    if max_len is not None:
        if ',' in max_len:
            max_len = max_len.split(',')
        else:
            max_len = len(pretraining_tasks) * [max_len]
    else:
        max_len = len(pretraining_tasks) * [None]

    model = None
    model_name = model_name if model_name else 'test'
    total_start = time.time()
    models = f'{base_path}models/'

    urls = {}
    if one_by_one:
        for objective, file, max_l in zip(pretraining_tasks, files, max_len):
            start = time.time()
            print("Start run for file %s and objective %s" % (file, objective))

            output = models + model_name + '/' + file

            Path(output).mkdir(parents=True, exist_ok=True)
            model, url = run_single_file(param_handler, objective, file, taurus, tokenizer_name, output=output, base_bert=model, max_len=max_l)
            duration_seconds = time.time() - start
            if url is not None:
                urls[objective.name] = url
            print("Finished run for file %s and objective %s in %s" % (file, objective, humanize.naturaldelta(timedelta(seconds=duration_seconds))))
        duration_seconds = time.time() - total_start
        print("Finished training of %s in %s" % (pretraining_tasks, humanize.naturaldelta(timedelta(seconds=duration_seconds))))

    else:
        start = time.time()
        print("Start run for files %s and objectives %s" % (files, pretraining_tasks))

        output = models + model_name + '/' + 'mixed'

        Path(output).mkdir(parents=True, exist_ok=True)
        model, url = run_single_file(param_handler, pretraining_tasks, files, taurus, tokenizer_name, output=output,
                                base_bert=model, max_len=max_len[0])
        if url is not None:
            urls["-".join(o.name for o in pretraining_tasks)] = url
        duration_seconds = time.time() - start
        print("Finished run for files %s and objectives %s in %s" % (files, pretraining_tasks, humanize.naturaldelta(timedelta(seconds=duration_seconds))))
        duration_seconds = time.time() - total_start
        print("Finished training of %s in %s" % (pretraining_tasks, humanize.naturaldelta(timedelta(seconds=duration_seconds))))

    output = models + 'best/' + model_name
    save_directory = output + '/' + param_handler.get_arg_value('pretraining_obj')
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {save_directory}")

    if len(urls) > 0:
        print(urls)
        json.dump(urls, open(save_directory + '/comet.json', 'w+'), indent=1)

    print("Waiting for 10 seconds to make sure the model is saved")
    time.sleep(10)
    print("Clearing cache")
    torch.cuda.empty_cache()
    print("Loading model")

    try:
        state_dict = torch.load(model)
    except FileNotFoundError as e:
        model = model.replace('best', 'final')
        state_dict = torch.load(model)

    state_dict = {k.removeprefix('bert.'): v for k, v in state_dict.items() if not k.startswith('objectives')}
    model = create_model(tokenizer_name)
    model.load_state_dict(state_dict)

    model.save_pretrained(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(save_directory)
    print("Saved model to %s" % save_directory)

    stats = {
        'objectives': ",".join([o.name for o in pretraining_tasks]),
        'mixed': not one_by_one,
        'steps': param_handler.get_arg_value('opt_steps')
    }
    json.dump(stats, open(save_directory + '/stats.json', 'w+', ))

    print("Finished training of %s in %s" % (pretraining_tasks, humanize.naturaldelta(timedelta(seconds=duration_seconds))))
