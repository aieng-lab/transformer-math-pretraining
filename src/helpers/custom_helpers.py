from typing import List

from src.config.params import ScriptParamHandler
from src.data_sets.text_data.DataEnum import DataEnum
from src.data_sets.text_data.BooksData import BooksData
from src.data_sets.text_data.WikipediaData import WikipediaData
from src.data_sets.text_data.OpenWebData import OpenWebData
from src.data_sets.text_data.TrainingDataSet import TrainingDataSet
from src.helpers.general_helpers import path_exists

from datasets import concatenate_datasets
from torch.utils.data.dataloader import get_worker_info
import os
import math
import gensim
import gensim.downloader as api


def concat_datasets(ds_list: List['TrainingDataSet']):
    data_list = [ds.data for ds in ds_list]
    return concatenate_datasets(data_list)


def extract_datasets(params: ScriptParamHandler):
    datasets = []
    ds_1 = params.get_arg_value("data1")
    if ds_1:
        datasets.append(ds_1)
    ds_2 = params.get_arg_value("data2")
    if ds_2:
        datasets.append(ds_2)
    ds_3 = params.get_arg_value("data3")
    if ds_3:
        datasets.append(ds_3)
    ds_4 = params.get_arg_value("data4")
    if ds_4:
        datasets.append(ds_4)

    for i, dataset in enumerate(datasets):
        if dataset.lower().startswith("wiki"):
            datasets[i] = DataEnum.WIKIPEDIA
        elif dataset.lower().startswith("book"):
            datasets[i] = DataEnum.BOOKCORPUS
        elif dataset.lower().startswith("openweb"):
            datasets[i] = DataEnum.OPENWEBTEXT

    datasets = [item for item in datasets if item]

    return datasets


def load_dataset(dataset: DataEnum, params: ScriptParamHandler, percentage=None, local=True):
    train_data_path = params.args.data
    if percentage:
        percentage = int(percentage)
    else:
        print("Loading without percentage", flush=True)
    if dataset == DataEnum.BOOKCORPUS:
        book_path = os.path.join(train_data_path, "books")
        book_file = params.get_arg_value("book_file")
        ds = BooksData(save_dir=book_path, local_file_name=book_file)
        ds.load_data(try_local=True, save=True, percentage=percentage, n_proc=params.get_arg_value("n_proc"))
        return ds
    elif dataset == DataEnum.WIKIPEDIA:
        wiki_path = os.path.join(train_data_path, "wiki")
        wiki_file = params.get_arg_value("wiki_file")
        wiki_config = params.get_arg_value("wikiconfig")
        ds = WikipediaData(save_dir=wiki_path, local_file_name=wiki_file, identifier=wiki_config)
        ds.load_data(try_local=True, save=True, percentage=percentage, n_proc=params.get_arg_value("n_proc"))
        return ds
    elif dataset == DataEnum.OPENWEBTEXT:
        open_web_path = os.path.join(train_data_path, "openwebtext")
        open_web_file = params.get_arg_value("open_web_file")
        ds = OpenWebData(save_dir=open_web_path, local_file_name=open_web_file, local=local)
        ds.load_data(try_local=True, save=True, percentage=percentage, n_proc=params.get_arg_value("n_proc"))
        return ds

    else:
        return None


def get_total_batches(data_len, batch_size):
    return int(data_len / batch_size) + 1


def get_trial_batch_num(num_batches, trial):
    percentage = trial / 100
    trial_part = math.ceil(num_batches * percentage)
    return trial_part


def _data_loader_gensim_init_(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    if dataset.w2v_model is None:
        model_name = "glove-wiki-gigaword-100"
        path = os.path.expanduser(f'~/gensim-data-0')
        file_path = os.path.join(path, f"{model_name}.bin")
        if path_exists(file_path):
            model = gensim.models.KeyedVectors.load(file_path)
        else:
            api.BASE_DIR = path
            model = api.load(model_name)
            model.save(file_path)
        dataset.w2v_model = model
