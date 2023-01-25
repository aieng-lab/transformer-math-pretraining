from helpers.timer import Timer
from helpers.general_helpers import create_path_if_not_exists, path_exists

import datasets
from datasets import load_dataset_builder, load_dataset, load_from_disk, concatenate_datasets
from pprint import pprint
import os.path
import numpy as np
import nltk
import spacy
from spacy_download import load_spacy
import apache_beam
import logging

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TrainingData:

    def __init__(self, *dsetnames):
        self.timer = Timer()
        self.books = None
        self.wiki = None
        self.wiki_config = None
        self.data_path = None
        for name in dsetnames:
            if name.lower().startswith(("books", "boookcorpus", "book")):
                self.books = "bookcorpus"
            elif name.lower().startswith(("wiki", "englishwiki", "enwiki", "english_wiki", "en_wiki")):
                self.wiki = "wikipedia"
                self.wiki_config = name.split("/")[-1]

    def set_data_path(self, path):
        create_path_if_not_exists(path)
        self.data_path = path

    def show_ds_info(self):
        print("\n SELECTED DATASETS:")
        print("-" * 40)
        if self.books:
            print(f"{self.books}:")
            ds_builder = load_dataset_builder(self.books)
            pprint(ds_builder.info.description)
            pprint(ds_builder.info.features)
            pprint(ds_builder.info.download_size)

        if self.wiki:
            print(f"\n{self.wiki}:")
            ds_builder = load_dataset_builder(self.wiki, self.wiki_config)
            pprint(ds_builder.info.description)
            pprint(ds_builder.info.features)
            pprint(ds_builder.info.download_size)

    def load_training_data(self, ds_name, percentage=None, local=True, save=False, overwrite=False):
        if ds_name.startswith("book"):
            print(f"\n Downloading dataset: {self.books}")
            ds_name = self.books
            ds = self.load_dataset(ds_name, percentage, local)
        elif ds_name.startswith(("wiki", "enwiki", "en_wiki", "englishwiki", "english_wiki")):
            print(f"\nDownloading dataset: {self.wiki}")
            ds_name = self.wiki
            ds = self.load_dataset(ds_name, percentage, local)
        else:
            ds = None
        if save:
            self.save_dataset_to_disk(ds_name, ds, overwrite)
        return ds

    def load_dataset(self, ds_name, percentage, local):
        self.timer.reset()
        self.timer.start()
        ds = None
        if local:
            ds = self.load_from_disk_if_exists(ds_name)
        if ds is not None:
            # found data on disk
            self.timer.stop()
            print("\nLoading time:")
            self.timer.print_elapsed()
        else:
            print("No dataset found on disk, loading from Huggingface Hub")
            perc_string = ""
            if percentage:
                perc_string = f"[:{percentage}%]"
            self.timer.reset()
            self.timer.start()
            if ds_name == self.books:
                ds = load_dataset(self.books, split=f"train{perc_string}")
                self.timer.stop()
                print("\nLoading time:")
                self.timer.print_elapsed()
            elif ds_name == self.wiki:
                ds = load_dataset(self.wiki, self.wiki_config, split=f"train{perc_string}", beam_runner="Spark")
                self.timer.stop()
                print("\nLoading and stripping time:")
                self.timer.print_elapsed()
        self.timer.reset()
        return ds

    def save_dataset_to_disk(self, name, ds, overwrite=False):
        print("\nSaving dataset ...")
        if self.data_path is None:
            _logger.warning("No data path was set, use the __.set_data_path method before trying to save a dataset.")
            return
        save_path = os.path.join(self.data_path, f"{name}.hf")
        print(f"Saving dataset {name} to {save_path} ...")
        if not ds:
            print("Nothing to save")
        if path_exists(save_path):
            if not overwrite:
                print(
                    f"File for dataset {name} already exists, aborting. Use overwrite=True to overwrite the existing file.")
                return
            else:
                print(f"File for dataset {name} already exists, overwriting.")
        self.timer.reset()
        self.timer.start()
        ds.save_to_disk(save_path)
        self.timer.stop()
        print("\nSaving time:")
        self.timer.print_elapsed()
        self.timer.reset()

    def load_from_disk_if_exists(self, ds_name):
        if self.data_path is None:
            _logger.warning("No data path was set, use the __.set_data_path method before trying to load from disk.")
            return None
        print(f"Searching for existing file for dataset {ds_name}...")
        path = os.path.join(self.data_path, f"{ds_name}.hf")
        if path_exists(path):
            print("File found, loading data ...")
            ds = load_from_disk(path)
        else:
            ds = None
        return ds

    def concat_datasets(self, *datasets):
        ds_list = list(datasets)
        if ds_list:
            features_type = ds_list[0].features.type
            for elem in ds_list:
                assert elem.features.type == features_type
            dataset_list = concatenate_datasets(ds_list)
            return dataset_list
        else:
            _logger.warning("No datasets provided, cannot concatenate.")
            return None

    def create_splits(self, dataset: datasets.Dataset, test_size=0.1, validation_size=0, shuffle=False):
        split = dataset.train_test_split(test_size=test_size, shuffle=shuffle)
        ds_train = split.get("train")
        ds_test = split.get("test")
        if validation_size > 0:
            split = ds_test.train_test_split(test_size=validation_size, shuffle=shuffle)
            ds_test = split.get("train")
            ds_val = split.get("test")
            return ds_train, ds_test, ds_val
        else:
            return ds_train, ds_test

    def process_wiki_sentence_level(self, ds_wiki: datasets.Dataset, save=True, overwrite=False):
        # nltk.download("punkt")
        nlp = load_spacy("en_core_web_sm", exclude=["parser", "tagger"])

        def transform_example(batch):
            texts = [text for i, text in enumerate(batch["text"])]
            ids = [id for i, id in enumerate(batch["id"])]
            sentences = []
            sentence_ids = []
            for i, text in enumerate(texts):
                text = text.replace("\n\n", " ")
                text = text.replace("\n", " ")
                with nlp.select_pipes(enable=['tok2vec', "parser", "senter"]):
                    doc = nlp(text)
                # tokenized = nltk.tokenize.sent_tokenize(text)
                tokenized = doc.sents
                sentences.extend(tokenized)
                sentence_ids.extend([str(id) for id in [int(ids[i])] * len(tokenized)])

            return {
                "text": sentences,
                "id": sentence_ids
            }

        processed_name = "wikipedia_sentence_splits"
        print(f"\nTrying to load processed dataset from disk ...")
        loaded = None
        if self.data_path:
            loaded = self.load_from_disk_if_exists(processed_name)
        if not loaded:
            print(f"No processed dataset found on disk, starting processing ...")
            ds_wiki = ds_wiki.remove_columns([col for col in ds_wiki.column_names if col not in ("text", "id")])
            ds_wiki = ds_wiki.map(transform_example, batched=True, remove_columns=ds_wiki.column_names)
        if save:
            self.save_dataset_to_disk(processed_name, ds_wiki, overwrite=overwrite)
        if loaded:
            ds_wiki = loaded
        return ds_wiki
