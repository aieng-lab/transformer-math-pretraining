from src.helpers.general_helpers import get_max_num_of_workers, create_path_if_not_exists, path_exists, get_size_in_bytes

from datasets import load_from_disk
import nltk
import json
from pathlib import Path
import shutil
import os.path
from collections import Counter
import math


class DataStats:

    def __init__(self, dataset_path, dataset_name, result_path, multi_processing=True, batch_size=None):
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.dataset_name = dataset_name
        self.ds = None
        self.max_workers = get_max_num_of_workers()
        self.data_len = None
        self.multi_processing = multi_processing
        self.temp_path = os.path.expanduser(f"~/temp/dataset_stats/{self.dataset_name}/")
        self.stats_path = os.path.join(result_path, "statistics")
        self.prop_path = os.path.join(result_path, "prop")
        self.batch_size = batch_size

    def load_data(self):
        ds = load_from_disk(self.dataset_path)
        self.ds = ds

    def get_data_len(self):
        self.data_len = len(self.ds)
        print(f"Length of Dataset: {self.data_len}")

    def get_prop_info(self):
        temp_path = os.path.join(self.temp_path, "prop")
        create_path_if_not_exists(self.prop_path)
        if path_exists(os.path.join(self.prop_path, "corpus_tfs.json")) and path_exists(
                os.path.join(self.prop_path, "corpus_dfs.json")) and path_exists(
            os.path.join(self.prop_path, "general_info.json")):
            print(f"Found computed PROP info at {self.prop_path}, Nothing to do")
        else:
            rm_path = Path(os.path.join(self.prop_path, "corpus_tfs.json"))
            if rm_path.exists() and rm_path.is_dir():
                shutil.rmtree(rm_path)

            rm_path = Path(os.path.join(self.prop_path, "corpus_dfs.json"))
            if rm_path.exists() and rm_path.is_dir():
                shutil.rmtree(rm_path)

            rm_path = Path(os.path.join(self.prop_path, "general_info.json"))
            if rm_path.exists() and rm_path.is_dir():
                shutil.rmtree(rm_path)

            create_path_if_not_exists(temp_path)

            print(f"Starting computation of PROP statistics", flush=True)

            if self.multi_processing:
                self.ds.map(self.compute_prop_info, batched=True, with_rank=True, num_proc=self.max_workers,
                            batch_size=self.batch_size)
            else:
                self.ds.map(self.compute_prop_info, batched=True, with_rank=True, batch_size=self.batch_size)

            # collecting data from all the temp dirs
            if self.multi_processing:
                num_threads = self.max_workers
            else:
                num_threads = 1

            corpus_tfs = Counter()
            corpus_dfs = Counter()
            number_of_words = 0

            print(f"Collecting results ...", flush=True)

            for i in range(num_threads):
                with open(os.path.join(temp_path, f"corpus_tfs_rank_{i}.json")) as tf_file:
                    tfs_i = json.load(tf_file)
                tfs_i = Counter(tfs_i)
                corpus_tfs.update(tfs_i)

                with open(os.path.join(temp_path, f"corpus_dfs_rank_{i}.json")) as df_file:
                    dfs_i = json.load(df_file)
                dfs_i = Counter(dfs_i)
                corpus_dfs.update(dfs_i)

                with open(os.path.join(temp_path, f"general_info_rank_{i}.json")) as general_file:
                    general_info = json.load(general_file)
                word_num_i = general_info.get("total_number_of_words")
                number_of_words += word_num_i

            with open(os.path.join(self.prop_path, "corpus_tfs.json"), "w") as file:
                json.dump(dict(corpus_tfs), file)

            with open(os.path.join(self.prop_path, "corpus_dfs.json"), "w") as file:
                json.dump(dict(corpus_dfs), file)

            total_words = number_of_words
            number_of_docs = self.data_len
            avg_doc_len = round(total_words / number_of_docs, 3)

            general_info = {"total_number_of_words": total_words,
                            "number_of_docs": number_of_docs,
                            "avg_doc_len": avg_doc_len,
                            "unique_words": len(list(corpus_tfs.keys()))}

            with open(os.path.join(self.prop_path, "general_info.json"), "w") as file:
                json.dump(general_info, file)

        print(f"Computing normalized dfs ...")

        self.compute_normalized_dfs()

        rm_path = Path(temp_path)
        if rm_path.exists() and rm_path.is_dir():
            shutil.rmtree(rm_path)

    def compute_normalized_dfs(self):
        if path_exists(os.path.join(self.prop_path, "normalized_dfs.json")):
            print(f"Normalized document frequencies have already been calculated. Nothing to do.")
            return

        with open(os.path.join(self.prop_path, "corpus_dfs.json")) as file:
            corpus_dfs = json.load(file)

        with open(os.path.join(self.prop_path, "corpus_tfs.json")) as file:
            corpus_tfs = json.load(file)

        normalized_df = {}
        total_df = sum(corpus_dfs.values())
        for word, count in corpus_dfs.items():
            if corpus_tfs[word] < 10:
                continue
            normalized_df[word] = count / total_df

        with open(os.path.join(self.prop_path, "normalized_dfs.json"), "w") as file:
            json.dump(normalized_df, file)

    def compute_prop_info(self, batch, rank):
        if rank is None:
            rank = 0

        texts = [text for i, text in enumerate(batch["text"])]

        number_of_words = 0
        total_word_counts = Counter()
        document_counts = {}
        for text in texts:
            text = text.lower()
            words = [word for word in nltk.word_tokenize(text) if word.isalpha()]
            number_of_words += len(words)
            word_counts = Counter(words)
            total_word_counts.update(word_counts)
            for word in list(word_counts.keys()):
                if word in document_counts:
                    document_counts[word] += 1
                else:
                    document_counts[word] = 1

        base_path = os.path.join(self.temp_path, "prop")
        create_path_if_not_exists(base_path)

        tf_file_name = f"corpus_tfs_rank_{rank}.json"
        df_file_name = f"corpus_dfs_rank_{rank}.json"
        general_file_name = f"general_info_rank_{rank}.json"

        if path_exists(os.path.join(base_path, tf_file_name)):
            with open(os.path.join(base_path, tf_file_name)) as file:
                saved_tf = json.load(file)
            saved_tf = Counter(saved_tf)
            saved_tf.update(total_word_counts)
            with open(os.path.join(base_path, tf_file_name), "w") as file:
                json.dump(dict(saved_tf), file)
        else:
            with open(os.path.join(base_path, tf_file_name), "w") as file:
                json.dump(dict(total_word_counts), file)

        if path_exists(os.path.join(base_path, df_file_name)):
            with open(os.path.join(base_path, df_file_name)) as file:
                saved_df = json.load(file)
            saved_df = Counter(saved_df)
            saved_df.update(document_counts)
            with open(os.path.join(base_path, df_file_name), "w") as file:
                json.dump(dict(saved_df), file)
        else:
            with open(os.path.join(base_path, df_file_name), "w") as file:
                json.dump(document_counts, file)

        if path_exists(os.path.join(base_path, general_file_name)):
            with open(os.path.join(base_path, general_file_name)) as file:
                saved_general_info = json.load(file)
            new_number_of_words = saved_general_info.get("total_number_of_words") + number_of_words
            with open(os.path.join(base_path, general_file_name), "w") as file:
                json.dump({"total_number_of_words": new_number_of_words}, file)
        else:
            with open(os.path.join(base_path, general_file_name), "w") as file:
                json.dump({"total_number_of_words": number_of_words}, file)

    def word_and_sentence_counts(self, batch, rank):
        texts = [text for i, text in enumerate(batch["text"])]
        total_word_count = 0
        total_sentence_count = 0
        for text in texts:
            words = [word for word in nltk.word_tokenize(text) if word.isalpha()]
            total_word_count += len(words)

            sentences = nltk.tokenize.sent_tokenize(text)
            total_sentence_count += len(sentences)

        tmp_path = os.path.expanduser(f"~/temp/dataset_stats/{self.dataset_name}/general_stats")
        create_path_if_not_exists(tmp_path)
        file_name = f"general_stats_rank_{rank}.json"
        file_path = os.path.join(tmp_path, file_name)
        if path_exists(file_path):
            with open(file_path) as file:
                json_dic = json.load(file)

            previous_word_count = json_dic.get("total_word_count")
            new_word_count = previous_word_count + total_word_count
            json_dic["total_word_count"] = new_word_count

            previous_sentence_count = json_dic.get("total_sentence_count")
            new_sentence_count = previous_sentence_count + total_sentence_count
            json_dic["total_sentence_count"] = new_sentence_count
            with open(file_path, "w") as file:
                json.dump(json_dic, file)

        else:
            json_dic = {"total_word_count": total_word_count, "total_sentence_count": total_sentence_count}
            with open(file_path, "w") as file:
                json.dump(json_dic, file)

    def get_general_stats(self):
        nltk.download("punkt")
        self.get_data_len()
        temp_path = os.path.join(self.temp_path, "general_stats")
        result_path = f"{self.result_path}/{self.dataset_name}"
        create_path_if_not_exists(result_path)
        result_name = "general_stats.json"

        if not path_exists(os.path.join(result_path, result_name)):
            if self.multi_processing:
                self.ds.map(self.word_and_sentence_counts, batched=True, with_rank=True, num_proc=self.max_workers,
                        batch_size=self.batch_size)
            else:
                self.ds.map(self.word_and_sentence_counts, batched=True, with_rank=True, batch_size=self.batch_size)

            total_word_count = 0
            total_sentence_count = 0
            for i in range(self.max_workers):
                file_name = f"general_stats_rank_{i}.json"
                file_path = os.path.join(temp_path, file_name)
                with open(file_path) as file:
                    data = json.load(file)
                    total_word_count += data.get("total_word_count")
                    total_sentence_count += data.get("total_sentence_count")

            print(f"Total word count: {total_word_count}")
            print(f"Total sentence count: {total_sentence_count}")

            avg_word_count = round(total_word_count / self.data_len, 2)
            avg_sentence_count = round(total_sentence_count / self.data_len, 2)

            byte_size = get_size_in_bytes(self.dataset_path)
            mb_size = byte_size / (1000 ** 2)
            gb_size = mb_size / 1000

            byte_size = round(byte_size, 2)
            mb_size = round(mb_size, 2)
            gb_size = round(gb_size, 2)

            print(f"Average word count per document: {avg_word_count}")
            print(f"Average sentence count per document: {avg_sentence_count}")
            print(f"Dataset length: {self.data_len}")
            print("Dataset size:")
            print(f"{byte_size} Bytes")
            print(f"{mb_size} MB")
            print(f"{gb_size} GB")

            with open(os.path.join(result_path, result_name), "w") as file:
                json.dump({"total_word_count": total_word_count, "total_sentence_count": total_sentence_count,
                           "avg_word_count": avg_word_count, "avg_sentence_count": avg_sentence_count,
                           "dataset_len": self.data_len, "bytes": byte_size, "MB": mb_size, "GB": gb_size}, file)

        else:
            with open(os.path.join(result_path, result_name)) as file:
                data = json.load(file)
            total_word_count = data.get("total_word_count")
            print(f"Total word count: {total_word_count}")
            total_sentence_count = data.get("total_sentence_count")
            print(f"Total sentence count: {total_sentence_count}")
            avg_word_count = data.get("avg_word_count")
            print(f"Average word count per document: {avg_word_count}")
            avg_sentence_count = data.get("avg_sentence_count")
            print(f"Average sentence count per document: {avg_sentence_count}")
            ds_len = data.get("dataset_len")
            print(f"Dataset length: {ds_len}")

        rm_path = Path(temp_path)
        if rm_path.exists() and rm_path.is_dir():
            shutil.rmtree(rm_path)


if __name__ == "__main__":
    dataset_path = "/home/katja/singularity/python-images/transformer_pretraining/python/train_data/wiki/Wikipedia_20220301_en_5perc_sentence_splits_packed_64_cleaned.hf"
    dataset_name = "Wikipedia_20220301_en_5perc_sentence_splits_packed_64_cleaned"
    result_path = "/home/katja/singularity/python-images/transformer_pretraining/python/output/dataset_stats"

    stat_retriever = DataStats(dataset_path, dataset_name, result_path)
    print(f"Result path: {stat_retriever.result_path}")

    stat_retriever.load_data()

    stat_retriever.get_data_len()

    # stat_retriever.count_words()
