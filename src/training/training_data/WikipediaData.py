from src.training.training_data.TrainingDataSet import TrainingDataSet
from src.helpers.general_helpers import get_max_num_of_workers

from datasets import load_dataset_builder, load_dataset, load_from_disk
from transformers import BertTokenizerFast
import nltk
from datasets.utils import disable_progress_bar


class WikipediaData(TrainingDataSet):

    def __init__(self, save_dir=None, local_file_name=None, identifier=None):
        super().__init__(save_dir, local_file_name)
        self.identifier = identifier
        if not self.identifier:
            self.identifier = "20220301.en"
        self.name = f"Wikipedia_{self.identifier}"
        if not self.file_name:
            self.file_name = self.name
        self.f_extension = "hf"
        if f".{self.f_extension}" not in self.file_name:
            self.file_name = self.file_name.replace(".", "_")
        else:
            name = self.file_name.split(f".{self.f_extension}")[0]
            self.file_name = f"{name.replace(',', '_')}.{self.f_extension}"

    def show_info(self):
        print("WIKIPEDIA INFO:")
        print("-" * 40)
        ds_builder = load_dataset_builder("wikipedia", self.identifier)
        print(f"Description: {ds_builder.info.description}")
        print(f"Features: {ds_builder.info.features}")
        print(f"Download Size: {ds_builder.info.download_size}")

    def load_data(self, percentage=None, try_local=True, save=True, overwrite=False):
        print(f"Loading dataset {self.name} ...")
        perc_string = ""
        if percentage:
            if self.file_name == self.name:
                self.name = f"{self.name}_{percentage}perc"
                self.file_name = self.name
            else:
                self.name = f"{self.name}_{percentage}perc"
            perc_string = f"[:{percentage}%]"
        if try_local:
            self.load_from_disk_if_exists()
        if not self.data:
            self.timer.reset()
            self.timer.start()
            self.data = load_dataset("wikipedia", self.identifier, split=f"train{perc_string}", beam_runner="Spark")
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

    def transform_to_sentences(self, save=True, overwrite=False):
        self.timer.reset()
        self.timer.start()
        disable_progress_bar()

        def transform(batch):
            return self.split_into_sentences(batch)

        processed_name = f"{self.name}_sentence_splits"
        self.file_name = processed_name.replace(".", "_")
        if self.load_from_disk_if_exists():
            return
        else:
            print(f"No processed dataset found on disk, starting processing ...")
            column_names = self.data.column_names
            self.data = self.data.map(transform, batched=True, remove_columns=column_names,
                                      num_proc=get_max_num_of_workers())
        self.timer.stop()
        print("\n Time for transformation:")
        self.timer.print_elapsed()

        if save:
            self.save_to_disk(overwrite)

    def transform_to_sentences_pack_longer_sequences(self, seq_len, tokenizer: BertTokenizerFast, save=True,
                                                     overwrite=False):

        disable_progress_bar()

        def transform(batch):
            sentence_batch = self.split_into_sentences(batch)
            return self.pack_into_longer_sequences(seq_len, tokenizer, sentence_batch)

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
            self.data = self.data.map(transform, batched=True, remove_columns=column_names, num_proc=get_max_num_of_workers())
            self.timer.stop()
            print("\nDataset transformation time:")
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

        return {
            "text": sequences,
            "id": sequence_ids
        }

    def strip_all_but_text_column(self):
        column_names = [name for name in self.data.column_names if name != "text"]
        self.data = self.data.remove_columns(column_names)


if __name__ == "__main__":

    local_dir = "/home/katja/singularity/python-images/transformer_pretraining/python/train_data/wiki"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    ds = WikipediaData(local_dir)
    ds.show_info()
    ds.load_data(percentage=5, try_local=True, save=True)
    ds.transform_to_sentences_pack_longer_sequences(64, tokenizer, save=True)
    for i in range(10):
        print(ds.data[i])

    for i in range(283, 290):
        print(ds.data[i])
