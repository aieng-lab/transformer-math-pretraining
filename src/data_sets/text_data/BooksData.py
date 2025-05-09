from src.data_sets.text_data.TrainingDataSet import TrainingDataSet
from src.helpers.general_helpers import get_max_num_of_workers

from datasets import load_dataset_builder, load_dataset, load_from_disk
from datasets.utils import disable_progress_bar
from transformers import BertTokenizerFast


class BooksData(TrainingDataSet):
    """Downloaded data from Huggingface: 1.18G"""

    def __init__(self, save_dir=None, local_file_name=None, local=True):
        super().__init__(save_dir, local_file_name, local)
        self.f_extension = "hf"
        self.name = "BooksCorpus"
        if not self.file_name:
            self.file_name = self.name
        if self.file_name.endswith(self.f_extension):
            self.file_name = self.file_name.rstrip(f".{self.f_extension}")
        self.file_name = self.file_name.replace(".", "_")

    def show_info(self):
        print("BOOKSCORPUS INFO:")
        print("-" * 40)
        ds_builder = load_dataset_builder("bookcorpus")
        print(f"Description: {ds_builder.info.description}")
        print(f"Features: {ds_builder.info.features}")
        print(f"Download Size: {ds_builder.info.download_size}")

    def load_data(self, percentage=None, try_local=True, save=True, overwrite=False, n_proc=False):
        print(f"\nLoading dataset {self.name} ...")
        print(f"Percentage: {percentage}")
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
            # set_verbosity_error()
            print(f"perc_string: {perc_string}")
            if n_proc:
                self.data = load_dataset("bookcorpus", split=f"train{perc_string}",
                                         num_proc=get_max_num_of_workers())  # Multi-processing makes no difference here?
            else:
                self.data = load_dataset("bookcorpus",
                                         split=f"train{perc_string}")  # Multi-processing makes no difference here?
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

    def pack_into_longer_sequences(self, seq_len, tokenizer: BertTokenizerFast, save=True, overwrite=False):
        def transform(batch):
            sentences = [text for i, text in enumerate(batch["text"])]
            sequences = []
            sequence = ""
            current_len = 0
            for sentence in sentences:
                sentence_len = len(tokenizer.tokenize(sentence))
                current_len += sentence_len
                if current_len > seq_len:
                    sequences.append(sequence)
                    sequence = ""
                    current_len = sentence_len
                sequence += f"{sentence} "
            sequences.append(sequence)

            final_sequences = []
            for sequence in sequences:
                if self.check_sequence(sequence, tokenizer):
                    final_sequences.append(sequence)

            return {
                "text": final_sequences
            }


        processed_name = f"{self.name}_packed_{seq_len}"
        self.file_name = processed_name
        if self.load_from_disk_if_exists():
            return
        else:
            print(f"No processed dataset found on disk, starting processing ...")
            disable_progress_bar()
            self.timer.reset()
            self.timer.start()
            self.data = self.data.map(transform, batched=True, remove_columns="text", num_proc=get_max_num_of_workers())
            self.timer.stop()
            print("\nTransformation time:")
            self.timer.print_elapsed()
        if save:
            self.save_to_disk(overwrite)

    def check_sequence(self, sequence, tokenizer):
        tokenized = tokenizer.tokenize(sequence)
        length = len(tokenized)
        unk_occurrences = tokenized.count(tokenizer.unk_token)
        if unk_occurrences >= length / 2:
            return False
        else:
            return True


if __name__ == "__main__":
    ds = BooksData()
    ds.show_info()
    local_path = "/train_data/books"
    ds.set_data_path(local_path)
    ds.load_data(percentage=5, try_local=True, save=True)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # set_verbosity_error()
    ds.pack_into_longer_sequences(64, tokenizer, save=True)
    for i in range(10):
        print(ds.data[i])

    length = len(ds.data)
    for i in range(length - 6, length - 1):
        print(ds.data[i])
