import math

from src.config.params import ScriptParamHandler
from src.training.training_data.DataEnum import DataEnum
from src.training.training_data.BooksData import BooksData
from src.training.training_data.WikipediaData import WikipediaData
from src.training.training_data.TrainingDataSet import TrainingDataSet
from src.helpers.general_helpers import get_max_num_of_workers

import os
from enum import Enum

"""
Module for downloading data and applying data transformations.
Only one transformation can be applied to all given datasets in each run.

Arguments:
--data1 ... --data4: Names of Datasets to be downloaded, e.g. "Wikipedia"
--wikipath: Filename under which the Wikipedia dataset should be saved or has previously been saved
--bookpath: --"--                    BookCorpus dataset --"--
--wikiconfig: identifier of the preprocessed version of the Huggingfacce Wikipedia Dataset, e.g., 20220301.en, if not 
                                            specified, default config is chosen (as in example)
--perc: Percentage of data that should be loaded, e.g. 5
--transformation: Name of the transformation that should be applied to all datasets, e.g. sentencizer (names of 
                                            Transformations enum, if not specified, no transformation will be applied)
                                                                                                
Additional parameters that should be set:
--data: path to the directory where all training data should be stored / has been stored
--maxlen: Max input length that will later be fed into the model (affects length of sequence_packing transformation), 
                                                                                                            e.g. 512
--

"""


class Transformations(Enum):
    SENTENCIZER = "sentencizer"
    SEQUENCE_PACKING = "sequence_packing"


def get_params():
    param_handler = ScriptParamHandler("--data1", "--data2", "--data3", "--data4", "--wikipath", "--bookpath",
                                       "--wikiconfig", "--perc", "--transformation")
    return param_handler


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

    datasets = [item for item in datasets if item]

    return datasets


def load_dataset(dataset: DataEnum, params: ScriptParamHandler, percentage=None):
    train_data_path = params.args.data
    if dataset == DataEnum.BOOKCORPUS:
        book_path = os.path.join(train_data_path, "books")
        book_file = params.get_arg_value("bookpath")
        ds = BooksData(save_dir=book_path, local_file_name=book_file)
        ds.load_data(try_local=True, save=True, percentage=percentage)
        return ds
    elif dataset == DataEnum.WIKIPEDIA:
        wiki_path = os.path.join(train_data_path, "wiki")
        wiki_file = params.get_arg_value("wikipath")
        wiki_config = params.get_arg_value("wikiconfig")
        ds = WikipediaData(save_dir=wiki_path, local_file_name=wiki_file, identifier=wiki_config)
        ds.load_data(try_local=True, save=True, percentage=percentage)
        return ds
    else:
        return None


def apply_transformation(ds: TrainingDataSet, transformation: str, params, tokenizer=None):
    print(f"Transforming dataset {ds.name} ...")
    if isinstance(ds, BooksData):
        if transformation == Transformations.SENTENCIZER.name:
            print(f"{transformation}:")
            print("No transformation to be done.")
            return ds
        elif transformation == Transformations.SEQUENCE_PACKING.name:
            print(f"{transformation}:")
            seq_len = math.floor(params.args.maxlen / 2)
            assert (tokenizer is not None)
            ds.pack_into_longer_sequences(seq_len=seq_len, tokenizer=tokenizer, save=True)
            return ds
        else:
            print("No transformation to be done.")
            return ds
    elif isinstance(ds, WikipediaData):
        if transformation == Transformations.SENTENCIZER.name:
            print(f"{transformation}:")
            ds.transform_to_sentences(save=True)
            return ds
        elif transformation == Transformations.SEQUENCE_PACKING.name:
            print(f"{transformation}:")
            seq_len = math.floor(params.args.maxlen / 2)
            assert (tokenizer is not None)
            ds.transform_to_sentences_pack_longer_sequences(seq_len=seq_len, tokenizer=tokenizer, save=True)
            return ds
        else:
            print("No transformation to be done.")
            return ds
    else:
        print("No transformation to be done")
        return ds


if __name__ == "__main__":
    print("\nPYTHON OUTPUT")
    print("-" * 60)
    print("\n")

    max_workers = get_max_num_of_workers()
    print(f"\nMax available worker threads: {max_workers}\n")

    params = get_params()
    percentage = params.get_arg_value("perc")
    transformation = params.get_arg_value("transformation")
    datasets, wiki_config = extract_datasets(params)
    ds_list = []
    for dataset in datasets:
        ds = load_dataset(dataset, params, float(percentage))
        ds_list.append(ds)

    if transformation:
        for i, ds in ds_list:
            ds = apply_transformation(ds, transformation, params)
