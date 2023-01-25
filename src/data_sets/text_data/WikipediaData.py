from src.data_sets.text_data.TrainingDataSet import TrainingDataSet
from src.helpers.general_helpers import get_max_num_of_workers

from datasets import load_dataset_builder, load_dataset, load_from_disk
from transformers import BertTokenizerFast
import nltk
from datasets.utils import disable_progress_bar
import re


class WikipediaData(TrainingDataSet):

    def __init__(self, save_dir=None, local_file_name=None, identifier=None, local=True):
        super().__init__(save_dir, local_file_name, local)
        self.identifier = identifier
        if not self.identifier:
            self.identifier = "20220301.en"
        self.name = f"Wikipedia_{self.identifier.replace('.', '_')}"
        if not self.file_name:
            self.file_name = self.name
        self.f_extension = "hf"
        if self.file_name.endswith(self.f_extension):
            self.file_name = self.file_name.rstrip(f".{self.f_extension}")
        self.file_name = self.file_name.replace(".", "_")

    def show_info(self):
        print("WIKIPEDIA INFO:")
        print("-" * 40)
        ds_builder = load_dataset_builder("wikipedia", self.identifier)
        print(f"Description: {ds_builder.info.description}")
        print(f"Features: {ds_builder.info.features}")
        print(f"Download Size: {ds_builder.info.download_size}")

    def load_data(self, percentage=None, try_local=True, save=True, overwrite=False, n_proc=False):
        print(f"Loading dataset {self.name} ...")
        perc_string = ""
        if percentage:
            self.adapt_names_to_percentage(percentage)
            perc_string = f"[:{percentage}%]"
        if try_local:
            self.load_from_disk_if_exists()
        if not self.data:
            self.timer.reset()
            self.timer.start()
            disable_progress_bar()
            if n_proc:
                self.data = load_dataset("wikipedia", self.identifier, split=f"train{perc_string}",
                                         num_proc=get_max_num_of_workers())  # , beam_runner="DirectRunner",
                # num_proc=get_max_num_of_workers())
            else:
                self.data = load_dataset("wikipedia", self.identifier,
                                         split=f"train{perc_string}")  # , beam_runner="DirectRunner") #beam_runner="Spark"
            self.timer.stop()
            print("\nDataset loading time:")
            self.timer.print_elapsed()

        # self.data = self.data.with_format("np")

        if save:
            self.save_to_disk(overwrite)

    def load_from_disk(self, path):
        self.data = load_from_disk(path)

    def save(self, path):
        if f".{self.f_extension}" not in path:
            path = f"{path}.{self.f_extension}"
        self.data.save_to_disk(path)

    def transform_to_sentences(self, save=True, overwrite=False, clean=False):
        disable_progress_bar()
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
            self.data = self.data.map(transform, batched=True, remove_columns=column_names,
                                      num_proc=get_max_num_of_workers())
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)

    def clean_dataset(self, save=True, overwrite=False):
        #disable_progress_bar()
        nltk.download("punkt")

        def clean_batch(batch):
            texts = [text for i, text in enumerate(batch["text"])]
            ids = [id for i, id in enumerate(batch["id"])]

            new_texts = []

            for i, text in enumerate(texts):
                length = len(text)
                count = text.count("|")
                if count / length > 0.01:
                    tables = True
                else:
                    tables = False
                text = self.remove_meaningless_paragraphs(text)
                text = self.remove_headings(text)
                if tables:
                    text = self.filter_tables(text)
                text = text.replace("\n\n", " ")
                text = text.replace("\n", " ")
                new_texts.append(text)

            return {
                "text": new_texts,
                "id": ids
            }

        processed_name = f"{self.name}_cleaned"
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
            self.data = self.data.map(clean_batch, batched=True, remove_columns=column_names,
                                      num_proc=get_max_num_of_workers())
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)


    def transform_to_sentences_pack_longer_sequences(self, seq_len, tokenizer: BertTokenizerFast, save=True,
                                                     overwrite=False, clean=False):
        disable_progress_bar()
        nltk.download("punkt")

        def transform(batch):
            sentence_batch = self.split_into_sentences(batch, clean=clean)
            sequence_batch = self.pack_into_longer_sequences(seq_len, tokenizer, sentence_batch)
            return self.transform_ids(sequence_batch)

        processed_name = f"{self.name}_sentence_splits_packed_{seq_len}"
        if clean:
            processed_name = f"{processed_name}_cleaned"
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
            self.data = self.data.map(transform, batched=True, remove_columns=column_names,
                                      num_proc=get_max_num_of_workers())
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)

    def split_into_sentences(self, batch, clean=False):
        texts = [text for i, text in enumerate(batch["text"])]
        ids = [id for i, id in enumerate(batch["id"])]
        sentences = []
        sentence_ids = []
        for i, text in enumerate(texts):
            if clean:
                length = len(text)
                count = text.count("|")
                if count / length > 0.01:
                    tables = True
                else:
                    tables = False
                text = self.remove_meaningless_paragraphs(text)
                text = self.remove_headings(text)
                if tables:
                    text = self.filter_tables(text)
            text = text.replace("\n\n", " ")
            text = text.replace("\n", " ")
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
        ids = [f"wiki_{id}" for i, id in enumerate(batch["id"])]
        texts = [text for i, text in enumerate(batch["text"])]

        return {
            "id": ids,
            "text": texts
        }

    def check_sequence(self, sequence, tokenizer):
        tokenized = tokenizer.tokenize(sequence)
        length = len(tokenized)
        unk_occurrences = tokenized.count(tokenizer.unk_token)
        if unk_occurrences >= length / 2:
            return False
        else:
            return True

    def strip_all_but_text_column(self):
        column_names = [name for name in self.data.column_names if name != "text"]
        self.data = self.data.remove_columns(column_names)

    def strip_irrelevant_column(self):
        column_names = [name for name in self.data.column_names if name not in ("text", "id")]
        self.data = self.data.remove_columns(column_names)

    def remove_meaningless_paragraphs(self, text):
        paragraphs = ["See also", "References", "External links", "Footnotes", "Further reading", "Bibliography",
                      "Further references", "Gallery"]
        for paragraph in paragraphs:
            parts = text.split(f"\n\n{paragraph}\n\n")
            if len(parts) == 1:
                parts = text.split(f"\n\n{paragraph} \n\n")
            if len(parts) == 1:
                parts = text.split(f"\n\n {paragraph}\n\n")
            if len(parts) == 1:
                parts = text.split(f"\n\n{paragraph}\n")
            if len(parts) == 1:
                parts = text.split(f"\n\n{paragraph} \n")
            if len(parts) == 1:
                parts = text.split(f"\n\n {paragraph} \n\n")
            if len(parts) == 1 and paragraph == "Footnotes":
                parts = text.split(f"\n\n{paragraph}<")

            relevant_part = parts[0]
            if relevant_part:
                text = relevant_part
        return text

    def remove_headings(self, text):
        relevant_text = []
        parts = text.split("\n\n")
        for part in parts:
            words = part.split(" ")
            if len(words) > 6:
                relevant_text.append(part)

        text = "\n\n".join(relevant_text)

        pattern = r"\n\n(.*?)\n"
        possible_titles = re.findall(pattern, text, re.DOTALL)

        for candidate in possible_titles:
            words = candidate.split(" ")
            if len(words) <= 6:
                text = text.replace(f"\n\n{candidate}\n", '\n')

        return text

    def filter_tables(self, text):
        table_pattern = r"\{\|.*?\|\}"
        text = re.sub(table_pattern, "", text, flags=re.DOTALL)
        table_pattern = r"(\|\-.+?\n)|(\|.+\n)"
        text = re.sub(table_pattern, "", text, flags=re.DOTALL)
        return text


if __name__ == "__main__":

    local_dir = "/train_data/wiki"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    ds = WikipediaData(local_dir)
    ds.show_info()
    ds.load_data(percentage=5, try_local=True, save=True)
    ds.transform_to_sentences_pack_longer_sequences(64, tokenizer, save=True)
    for i in range(10):
        print(ds.data[i])

    for i in range(283, 290):
        print(ds.data[i])
