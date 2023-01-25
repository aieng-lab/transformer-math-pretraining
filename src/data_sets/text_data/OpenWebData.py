import math

from src.data_sets.text_data.TrainingDataSet import TrainingDataSet
from src.helpers.general_helpers import get_max_num_of_workers

from datasets import load_dataset_builder, load_dataset, load_from_disk, Dataset
from datasets.utils import disable_progress_bar
import nltk
import re

from transformers import BertTokenizerFast


class OpenWebData(TrainingDataSet):

    def __init__(self, save_dir=None, local_file_name=None, local=True):
        super().__init__(save_dir, local_file_name, local)
        self.f_extension = "hf"
        self.name = "OpenWebText"
        if not self.file_name:
            self.file_name = self.name
        if self.file_name.endswith(self.f_extension):
            self.file_name = self.file_name.rstrip(f".{self.f_extension}")
        self.file_name = self.file_name.replace(".", "_")
        self.original_len = 8013769  # use this number (provided by HF dataset card) for total length of original
        # dataset since we do not want to download the entire dataset (that would cost too much time during training,
        # so no need for downloading this entire massive set)
        # We only want to have something equivalent to the BookCorpus (at first)

    def show_info(self):
        print("OpenWebText INFO:")
        print("-" * 40)
        ds_builder = load_dataset_builder("openwebtext")
        print(f"Description: {ds_builder.info.description}")
        print(f"Features: {ds_builder.info.features}")
        print(f"Download Size: {ds_builder.info.download_size}")

    def create_arrow_set_from_streaming_sample(self, data, take_num):
        print(f"Creating Arrow Dataset from subset of streamed dataset ...")
        texts = []
        indices = []
        idx = 0
        iterator = iter(data)

        threshold = 10

        while True:
            try:
                elem = next(iterator)
            except StopIteration:
                break
            texts.append(elem["text"])
            indices.append(idx)
            percent = math.floor((idx / take_num) * 100)
            if percent >= threshold:
                print(f"Finished {percent}%")
                threshold += 10
            idx += 1

        print(f"Finished 100%")

        dataset = Dataset.from_dict({"id": indices, "text": texts})
        return dataset

    def remove_garbage(self, ds: Dataset):
        print(f"Removing garbage ...")

        def filtering(example_batch):
            texts = example_batch["text"]
            ids = example_batch["id"]

            filtered_texts = []
            filtered_ids = []

            for i, text in enumerate(texts):
                # remove short texts
                length = len(text)
                if length < 2000:
                    continue

                # remove consecutive characters:
                # '-'
                pattern = r"-{5,}"
                text = re.sub(pattern, "", text)
                # '.'
                pattern = r"\.{5,}"
                text = re.sub(pattern, "", text)
                # '_'
                pattern = r"_{4,}"
                text = re.sub(pattern, "", text)
                # '~'
                pattern = r"~{2,}"
                text = re.sub(pattern, "", text)
                # '*'
                pattern = r"\*{10,}"
                text = re.sub(pattern, "", text)
                # '='
                pattern = r"={5,}"
                text = re.sub(pattern, "", text)

                # remove texts that contain special characters
                # '|'
                count = text.count("|")
                if count / length > 0.005:
                    continue
                # '#'
                count = text.count("#")
                if count / length > 0.005:
                    continue
                # '/'
                count = text.count("/")
                if count / length > 0.01:
                    continue
                # '\'
                count = text.count("\\")
                if count / length > 0.005:
                    continue
                # '+'
                count = text.count("+")
                if count / length > 0.005:
                    continue
                # '['
                count = text.count("[")
                if count / length > 0.01:
                    continue
                # '{'
                count = text.count("{")
                if count / length > 0.005:
                    continue
                # '='
                count = text.count("=")
                if count / length > 0.005:
                    continue
                # '.'
                count = text.count(".")
                if count / length > 0.027:
                    continue
                # '@'
                count = text.count("@")
                if count / length > 0.008:
                    continue
                # '<'
                count = text.count("<")
                if count / length > 0.005:
                    continue
                # '::'
                count = text.count("::")
                if count / length > 0.005:
                    continue
                # '-'
                count = text.count("-")
                if count / length > 0.02:
                    continue

                filtered_texts.append(text)
                filtered_ids.append(ids[i])

            return {
                "id": filtered_ids,
                "text": filtered_texts
            }

        column_names = ds.column_names
        ds = ds.map(filtering, batched=True, remove_columns=column_names, num_proc=get_max_num_of_workers())
        return ds

    def stream_and_filter(self, take_num):
        print(f"Streaming Dataset {self.name} from huggingface ...")
        data = load_dataset("openwebtext", split="train", streaming=True)
        if self.local:
            buffer_size = 10000  # todo: test this
        else:
            buffer_size = 1000000  # todo: test this
        data = data.shuffle(seed=824, buffer_size=buffer_size)
        data = data.take(take_num)

        ds = self.create_arrow_set_from_streaming_sample(data, take_num)
        ds = self.remove_garbage(ds)
        return ds

    def load_data(self, percentage=None, try_local=True, save=True, overwrite=False, n_proc=False):
        print(f"Loading dataset {self.name} ...")
        if percentage:
            self.adapt_names_to_percentage(percentage)
            percentage = percentage / 100
            take_num = int(self.original_len * percentage)
        else:
            take_num = int(self.original_len * 0.135)
        if try_local:
            self.load_from_disk_if_exists()
        if not self.data:
            self.timer.reset()
            self.timer.start()
            disable_progress_bar()
            self.data = self.stream_and_filter(take_num)
            self.timer.stop()
            print("\nDataset loading time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)

    def load_from_disk(self, path):
        self.data = load_from_disk(path)

    def save(self, path):
        if f".{self.f_extension}" not in path:
            path = f"{path}.{self.f_extension}"
        self.data.save_to_disk(path)

    def transform_to_sentences(self, save=True, overwrite=False, clean=False):
        nltk.download("punkt")

        def transform(batch):
            sentence_batch = self.split_into_sentences(batch, clean=clean)
            return self.transform_ids(sentence_batch)

        processed_name = f"{self.name}_sentence_splits"
        if clean:
            processed_name = f"{processed_name}_cleaned"
        self.file_name = processed_name.replace(".", "_")
        if self.load_from_disk_if_exists():
            return
        else:
            self.timer.reset()
            self.timer.start()
            print(f"No processed dataset found on disk, starting processing ...")
            column_names = self.data.column_names
            disable_progress_bar()
            self.data = self.data.map(transform, batched=True, remove_columns=column_names,
                                      num_proc=get_max_num_of_workers())
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)

    def transform_to_sentences_pack_longer_sequences(self, seq_len, tokenizer: BertTokenizerFast, save=True,
                                                     overwrite=False):
        nltk.download("punkt")

        def transform(batch):
            sentence_batch = self.split_into_sentences(batch)
            sequence_batch = self.pack_into_longer_sequences(seq_len, tokenizer, sentence_batch)
            return self.transform_ids(sequence_batch)

        processed_name = f"{self.name}_sentence_splits_packed_{seq_len}"
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
            disable_progress_bar()
            self.data = self.data.map(transform, batched=True, remove_columns=column_names,
                                      num_proc=get_max_num_of_workers())
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)

    def split_into_sentences(self, batch):
        texts = [text for i, text in enumerate(batch["text"])]
        ids = [id for i, id in enumerate(batch["id"])]
        sentences = []
        sentence_ids = []
        for i, text in enumerate(texts):
            text = text.replace("\n\n", " ")
            text = text.replace("\n", " ")
            text = text.replace("  ", " ")
            tokenized = nltk.tokenize.sent_tokenize(text)
            sentences.extend(tokenized)
            sentence_ids.extend([str(id) for id in [int(ids[i])] * len(tokenized)])

        return {
            "text": sentences,
            "id": sentence_ids
        }

    def pack_into_longer_sequences(self, seq_len, tokenizer, sentence_batch):
        sentences = sentence_batch.get("text")
        sentence_ids = sentence_batch.get("id")

        sequences = []
        sequence = ""
        current_len = 0
        current_id = sentence_ids[0]
        sequence_ids = []
        for i, sentence in enumerate(sentences):
            sentence_len = len(tokenizer.tokenize(sentence))

            if sentence_ids[i] != current_id:
                # New document, create new sequence
                sequence_ids.append(current_id)
                current_id = sentence_ids[i]
                sequences.append(sequence)
                sequence = ""
                current_len = sentence_len

            current_len += sentence_len

            if current_len > seq_len:
                sequences.append(sequence)
                sequence_ids.append(current_id)
                sequence = ""
                current_len = sentence_len

            sequence += f"{sentence} "
        sequences.append(sequence)
        sequence_ids.append(current_id)

        final_sequences = []
        final_ids = []
        for index, sequence in enumerate(sequences):
            if self.check_sequence(sequence, tokenizer):
                final_sequences.append(sequence)
                final_ids.append(sequence_ids[index])

        return {
            "text": final_sequences,
            "id": final_ids
        }

    def transform_ids(self, batch):
        ids = [f"owt_{id}" for i, id in enumerate(batch["id"])]
        texts = [text for i, text in enumerate(batch["text"])]

        return {
            "id": ids,
            "text": texts
        }

    def check_sequence(self, sequence, tokenizer):
        # todo: check if 60 or 75% of the text consist only of special characters (find out decent estimate for %)
        special_chars = [".", "-", "|", "/", "\\", "@", "_", "[", "]", "(", ")", "{", "}", "<", ">", "+", "#", "~", "*",
                         "=", ";", ":"]
        special_char_count = 0
        for item in special_chars:
            count = sequence.count(item)
            special_char_count += count

        digit_count = 0
        for char in sequence:
            if char.isdigit():
                digit_count += 1

        special_count = special_char_count + digit_count
        length = len(sequence)
        if length == 0:
            return False
        if special_count / length > 0.33:
            return False

        tokenized = tokenizer.tokenize(sequence)
        length = len(tokenized)
        unk_occurrences = tokenized.count(tokenizer.unk_token)
        if unk_occurrences >= length / 2:
            return False
        else:
            return True
