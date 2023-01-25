"""
Module for downloading data and applying data transformations.
Only one transformation can be applied to all given datasets in each run.

Arguments:
--data1 ... --data4: Names of Datasets to be downloaded, e.g. "Wikipedia"
--wikipath: Filename under which the Wikipedia dataset should be saved or has previously been saved
                                                                                            (relative to --data/wiki)
--bookpath: --"--                    BookCorpus dataset --"--                                            --data/books
--wikiconfig: identifier of the preprocessed version of the Huggingfacce Wikipedia Dataset, e.g., 20220301.en, if not
                                            specified, default config is chosen (as in example)
--perc: Percentage of data that should be loaded, e.g. 5
--transformation: Name of the transformation that should be applied to all datasets, e.g. sentencizer (names of
                                            Transformations enum, if not specified, no transformation will be applied)
--tokenizer_name: Filename of local tokenizer (relative to --tokenizer), if not set, Huggingface
                    bert-base-cased tokenizer is used)

Additional parameters that should be set:
--data: path to the directory where all training data should be stored / has been stored
--maxlen: Max input length that will later be fed into the model (affects length of sequence_packing transformation),
            e.g. 512
--n_proc: true / false: use multiprocessing for download or not


If you run this file, the specified datasets will be downloaded (or loaded locally if they were previously downloaded)
to the path given with --data (and --...file) and the specified transformation will be applied to all datasets
"""

import math

from src.config.params import ScriptParamHandler
from src.data_sets.text_data.BooksData import BooksData
from src.data_sets.text_data.WikipediaData import WikipediaData
from src.data_sets.text_data.OpenWebData import OpenWebData
from src.data_sets.text_data.TrainingDataSet import TrainingDataSet
from src.helpers.general_helpers import get_max_num_of_workers, print_size
from src.helpers.custom_helpers import load_dataset, extract_datasets
from src.tokenizer.tokenizer import Tokenizer
from src.config.config import Config

import os
from enum import Enum
import math


class Transformations(Enum):
    """Enum of available data transformations"""
    SENTENCIZER = "sentencizer"
    SEQUENCE_PACKING = "sequence_packing"
    CLEANING = "cleaning"


def get_params():
    """Parse scripts parameters"""
    param_handler = ScriptParamHandler("--data1", "--data2", "--data3", "--data4", "--wikipath", "--bookpath",
                                       "--wikiconfig", "--perc", "--transformation", "--n_proc", "--tokenizer_name",
                                       "--cleaning", "--taurus", "--similar_words", "--word_sets", "--batch_size",
                                       "--num_proc", "--prop_path")
    return param_handler


def apply_transformation(ds: TrainingDataSet, transformation: str, params, tokenizer=None, with_cleaning=False):
    print(f"\nTransforming dataset {ds.name} ...")
    if isinstance(ds, BooksData):
        if transformation.lower() == Transformations.SENTENCIZER.value:
            print(f"{transformation}:")
            print("No transformation to be done.")
            return ds
        elif transformation.lower() == Transformations.SEQUENCE_PACKING.value:
            print(f"{transformation}:")
            seq_len = math.floor(int(params.args.maxlen) / 2)
            assert (tokenizer is not None)
            ds.pack_into_longer_sequences(seq_len=seq_len, tokenizer=tokenizer, save=True)
            return ds
        else:
            print("No transformation to be done.")
            return ds
    elif isinstance(ds, WikipediaData):
        if transformation.lower() == Transformations.SENTENCIZER.value:
            print(f"{transformation}:")
            ds.transform_to_sentences(save=True, clean=with_cleaning)
            return ds
        elif transformation.lower() == Transformations.SEQUENCE_PACKING.value:
            print(f"{transformation}:")
            seq_len = math.floor(int(params.args.maxlen) / 2)
            assert (tokenizer is not None)
            ds.transform_to_sentences_pack_longer_sequences(seq_len=seq_len, tokenizer=tokenizer, save=True,
                                                            clean=with_cleaning)
            return ds
        elif transformation.lower() == Transformations.CLEANING.value:
            print(f"{transformation}:")
            ds.clean_dataset(save=True)
            return ds
        else:
            print("No transformation to be done.")
            return ds
    elif isinstance(ds, OpenWebData):
        if transformation.lower() == Transformations.SENTENCIZER.value:
            print(f"{transformation}:")
            ds.transform_to_sentences(save=True)
            return ds
        elif transformation.lower() == Transformations.SEQUENCE_PACKING.value:
            print(f"{transformation}:")
            seq_len = math.floor(int(params.args.maxlen) / 2)
            assert (tokenizer is not None)
            ds.transform_to_sentences_pack_longer_sequences(seq_len=seq_len, tokenizer=tokenizer, save=True)
            return ds
        else:
            print("No transformation to be done.")
            return ds

    else:
        print("No transformation to be done")
        return ds


def add_similar_words(ds: TrainingDataSet, num_proc=True):
    ds.add_similar_words(save=True, overwrite=False, num_proc=num_proc)
    return ds


def add_representative_words(ds: TrainingDataSet, prop_path, batch_size=None, num_proc=True):
    ds.add_word_sets(prop_path, save=True, overwrite=False, batch_size=batch_size, num_proc=num_proc)


def get_tokenizer(tokenizer_name):
    if not tokenizer_name:
        tokenizer = Tokenizer(config)
        tokenizer.load_from_huggingface()
    else:
        tokenizer = Tokenizer(config, tokenizer_name)
        tokenizer.load_local()
    return tokenizer.get_tokenizer()


def print_train_data_sizes(params):
    base_path = params.args.data
    subdir_lists = [os.path.join(base_path, f) for f in os.listdir(base_path)]
    for subdir in subdir_lists:
        print_size(subdir)


def print_train_data_lens(ds_list):
    for ds in ds_list:
        print(f"\nLength of dataset {ds.name}:")
        print(f"len: {len(ds.data)}, shape: {ds.data.shape}")


if __name__ == "__main__":
    print("\nPYTHON OUTPUT")
    print("-" * 60)
    print("\n")

    max_workers = get_max_num_of_workers()
    print(f"\nMax available worker threads: {max_workers}\n")

    params = get_params()
    params.print_args()
    percentage = params.get_arg_value("perc")
    transformation = params.get_arg_value("transformation")
    cleaning = params.get_arg_value("cleaning")
    taurus = params.get_arg_value("taurus")
    if taurus:
        local = False
    else:
        local = True
    similar_words = params.get_arg_value("similar_words")
    word_sets = params.get_arg_value("word_sets")

    datasets = extract_datasets(params)
    ds_list = []
    for dataset in datasets:
        ds = load_dataset(dataset, params, percentage, local)
        ds_list.append(ds)
        print(f"Downladed dataset {dataset.name}")

    if transformation:
        config = Config()
        params.set_config_from_script_params(config, printing=False)

        tokenizer_name = params.args.tokenizer_name
        tokenizer = get_tokenizer(tokenizer_name)
        for i, ds in enumerate(ds_list):
            ds = apply_transformation(ds, transformation, params, tokenizer=tokenizer, with_cleaning=cleaning)
            print(f"Applied transformation to dataset {ds.name}")

    num_proc = params.get_arg_value("num_proc")

    if similar_words:
        print("Adding similar words")
        for i, ds in enumerate(ds_list):
            ds = add_similar_words(ds, num_proc)
            print(f"Added similar words to dataset {ds.name}")

    if word_sets:
        prop_path = params.get_arg_value("prop_path")
        batch_size = int(params.get_arg_value("batch_size"))

        print(f"Adding representative word sets")
        for i, ds in enumerate(ds_list):
            ds = add_representative_words(ds, prop_path, batch_size, num_proc)

    print_train_data_sizes(params)
    print_train_data_lens(ds_list)
