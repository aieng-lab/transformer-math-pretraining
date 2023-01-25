from __future__ import annotations

import random

from src.helpers.Timer import Timer
from src.helpers.general_helpers import create_path_if_not_exists, path_exists, find_file_by_name, \
    get_max_num_of_workers
from src.pretraining_methods.specialized.PROP.prepare import CorpusSampler, generate_word_sets

from abc import abstractmethod
import logging
import os
from datasets import concatenate_datasets
import re
import json
import nltk
from datasets.utils import disable_progress_bar

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TrainingDataSet:

    def __init__(self, save_dir=None, local_file_name=None, local=True):
        self.timer = Timer()
        self.local_data_path = save_dir
        self.data = None
        self.name = None
        self.file_name = local_file_name
        self.init_data_path()
        self.f_extension = ""
        self.timer = Timer()
        self.local = local
        self.similar_words = None
        self.prop_tools = None

    def set_data_path(self, path_to_dir):
        self.local_data_path = path_to_dir
        self.init_data_path()

    def set_file_name(self, name):
        self.file_name = name

    def init_data_path(self):
        if self.local_data_path:
            create_path_if_not_exists(self.local_data_path)

    @abstractmethod
    def show_info(self):
        pass

    @abstractmethod
    def load_data(self, percentage=None, try_local=True, save=True, overwrite=False):
        pass

    def save_to_disk(self, overwrite=False):
        print(f"\nSaving dataset {self.name} ...")
        if self.local_data_path is None:
            _logger.warning("No data path was set, use the __.set_data_path method before trying to save a dataset.")
            return
        print(f"Saving to {self.local_data_path} ...")
        if not self.data:
            print("Nothing to save")
        found = find_file_by_name(self.local_data_path, self.file_name)
        if found:
            if not overwrite:
                print(
                    f"File for dataset {self.name} already exists for the specified path, aborting. Use overwrite=True to overwrite the existing file.")
                return
            else:
                print(f"File for dataset {self.name} already exists at the specified path, overwriting.")
        self.save(os.path.join(self.local_data_path, self.file_name))

    @abstractmethod
    def save(self, path):
        pass

    def load_from_disk_if_exists(self):
        if self.local_data_path is None:
            _logger.warning("No data path was set, use the __.set_data_path method before trying to load from disk.")
            return None
        print(f"\nSearching for existing file for dataset {self.name}...")
        if "." in self.file_name:
            path = os.path.join(self.local_data_path, self.file_name)
        else:
            path = os.path.join(self.local_data_path, f"{self.file_name}.{self.f_extension}")
        if path_exists(path):
            print("File found, loading data ...")
            self.load_from_disk(path)
            return True
        else:
            return False

    @abstractmethod
    def load_from_disk(self, path):
        pass

    def concat_with(self, ds: __class__):
        ds_list = concatenate_datasets([self.data, ds.data])
        return ds_list

    def adapt_names_to_percentage(self, percentage):
        self.name = f"{self.name}_{percentage}perc"
        regex = re.compile("[0-9][0-9]?perc")
        if re.search(regex, self.file_name):
            sub_string = f"{percentage}perc"
            self.file_name = re.sub(regex, sub_string, self.file_name)
        else:
            self.file_name = f"{self.file_name}_{percentage}perc"

    def add_similar_words(self, save=True, overwrite=False, num_proc=True):
        def process_sim_word_batch(batch):
            texts = [text for i, text in enumerate(batch["text"])]
            ids = [id for i, id in enumerate(batch["id"])]

            similar_words_col = []
            random_words_col = []

            all_words = list(self.similar_words.keys())
            common_words = all_words[:200]

            for text in texts:
                similar_words = {}
                words = nltk.word_tokenize(text)
                words = list(set(words))

                random_common_words = random.sample(common_words, 50)
                random_words = random.sample(all_words, 150)
                total_random_words = random_common_words + random_words
                total_random_words = set(total_random_words)
                random_words = " ".join(total_random_words)
                random_words_col.append(random_words)

                for word in words:
                    sim_words = self.similar_words.get(word)
                    if sim_words is not None:
                        similar_words[word] = sim_words
                    else:
                        word = word.lower()
                        sim_words = self.similar_words.get(word)
                        if sim_words is not None:
                            similar_words[word] = sim_words

                similar_words = json.dumps(similar_words)
                similar_words_col.append(similar_words)

            return {
                "id": ids,
                "text": texts,
                "similar_words": similar_words_col,
                "random_words": random_words_col
            }

        path = os.path.expanduser(f"~/gensim-data/precomputed")
        with open(os.path.join(path, "similar_words_small_model.json")) as file:
            sim_words_dict = json.load(file)

        self.similar_words = sim_words_dict

        processed_name = f"{self.file_name}_sim_words"

        self.file_name = processed_name
        if f".{self.f_extension}" not in self.file_name:
            self.file_name = self.file_name.replace(".", "_")
        else:
            name = self.file_name.split(f".{self.f_extension}")[0]
            self.file_name = f"{name.replace('.', '_')}.{self.f_extension}"
        if self.load_from_disk_if_exists():
            return
        else:
            print(f"No processed dataset found on disk, starting processing ...")
            column_names = self.data.column_names
            self.timer.reset()
            self.timer.start()

            #disable_progress_bar()
            if num_proc:
                self.data = self.data.map(process_sim_word_batch, batched=True, remove_columns=column_names,
                                          num_proc=get_max_num_of_workers())
            else:
                self.data = self.data.map(process_sim_word_batch, batched=True, remove_columns=column_names)
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)

    def load_prop_tools(self, path_to_stats):
        normalized_dfs_path = os.path.join(path_to_stats, "normalized_dfs.json")
        with open(normalized_dfs_path) as file:
            normalized_dfs = json.load(file)
        corpus_tfs_path = os.path.join(path_to_stats, "corpus_tfs.json")
        with open(corpus_tfs_path) as file:
            corpus_tfs = json.load(file)
        general_info_path = os.path.join(path_to_stats, "general_info.json")
        with open(general_info_path) as file:
            general_info = json.load(file)
        total_number_of_words = general_info.get("total_number_of_words")
        avg_doc_len = general_info.get("avg_doc_len")
        negative_sampler = CorpusSampler(normalized_dfs)

        prop_tools = {
            "normalized_df": normalized_dfs,
            "corpus_tf": corpus_tfs,
            "total_number_of_words": total_number_of_words,
            "avg_doc_len": avg_doc_len,
            "negative_sampler": negative_sampler
        }
        self.prop_tools = prop_tools

    def add_word_sets(self, path_to_prop_stats, save=True, overwrite=False, batch_size=None, num_proc=True):
        self.load_prop_tools(path_to_prop_stats)

        def generate_word_sets_batch(batch):
            texts = [text for i, text in enumerate(batch["text"])]
            ids = [id for i, id in enumerate(batch["id"])]

            word_sets_col = []

            for text in texts:
                normalized_df = self.prop_tools.get("normalized_df")
                corpus_tf = self.prop_tools.get("corpus_tf")
                negative_sampler = self.prop_tools.get("negative_sampler")
                avg_doc_len = self.prop_tools.get("avg_doc_len")
                total_number_of_words = self.prop_tools.get("total_number_of_words")

                word_set_dict = {}
                for i in range(10):
                    word_sets = generate_word_sets(text, total_number_of_words, avg_doc_len, corpus_tf, normalized_df,
                                                   negative_sampler)
                    score_0 = word_sets[0][0][1]
                    score_1 = word_sets[0][1][1]
                    if score_0 > score_1:
                        true_label = 1  # first set should be ranked higher
                    else:
                        true_label = -1  # second set should be ranked higher
                    word_set_dict[i] = [word_sets[0][0][0], word_sets[0][1][0], true_label]

                word_set_dict = json.dumps(word_set_dict)
                word_sets_col.append(word_set_dict)

            return {
                "id": ids,
                "text": texts,
                "word_sets": word_sets_col
            }

        processed_name = f"{self.file_name}_word_sets"

        self.file_name = processed_name
        if f".{self.f_extension}" not in self.file_name:
            self.file_name = self.file_name.replace(".", "_")
        else:
            name = self.file_name.split(f".{self.f_extension}")[0]
            self.file_name = f"{name.replace('.', '_')}.{self.f_extension}"
        if self.load_from_disk_if_exists():
            return
        else:
            print(f"No processed dataset found on disk, starting processing ...")
            column_names = self.data.column_names
            self.timer.reset()
            self.timer.start()
            #disable_progress_bar()
            if num_proc:
                self.data = self.data.map(generate_word_sets_batch, batched=True, remove_columns=column_names,
                                          num_proc=get_max_num_of_workers(), batch_size=batch_size)
            else:
                self.data = self.data.map(generate_word_sets_batch, batched=True, remove_columns=column_names,
                                          batch_size=batch_size)
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)
