import torch.multiprocessing

from src.config.params import ScriptParamHandler
from src.tokenizer.tokenizer import Tokenizer
from src.config.config import Config
from src.data_sets.text_data.BooksData import BooksData
from src.data_sets.text_data.WikipediaData import WikipediaData
from src.data_sets.text_data.OpenWebData import OpenWebData
from src.helpers.Timer import Timer
from src.helpers.general_helpers import print_size

import os
from torch.utils.data import DataLoader, Dataset
import random
from transformers import BertTokenizerFast
import nltk
from pprint import pprint


class SimpleDataset1(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        return {
            "text": text
        }


class SimpleDataset2(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        is_last = False
        if index == len(self.data) - 1:
            is_last = True

        sentence_a = self.data[index]["text"]

        if is_last:
            sentence_b = None
            while sentence_b is None or sentence_b == sentence_a:
                sentence_b = self.data[random.randrange(len(self.data))]["text"]

        else:
            prob = random.random()

            if prob < 0.5:
                sentence_b = self.data[index + 1]["text"]
            else:
                sentence_b = self.data[random.randrange(len(self.data))]["text"]

        sentence_a = self.tokenizer.tokenize(sentence_a)
        sentence_b = self.tokenizer.tokenize(sentence_b)
        sequence = [self.tokenizer.cls_token] + sentence_a + [self.tokenizer.sep_token] + sentence_b + [
            self.tokenizer.sep_token]
        sequence = self.tokenizer.convert_tokens_to_ids(sequence)

        if len(sequence) >= 512:
            sequence = sequence[:512]
        sequence = sequence + ([0] * (512 - len(sequence)))

        return {
            "text": sequence
        }


def get_params():
    params = ScriptParamHandler("--perc", "--tokenizer_name", "--dataloaderfile", "--datasetnum", "--taurus")
    return params


def check_datasets(params, percentage=None, local=True):
    base_path = params.args.data
    subdir_lists = [os.path.join(base_path, f) for f in os.listdir(base_path)]
    for subdir in subdir_lists:
        sub_dirs = [f for f in os.listdir(subdir)]
        if subdir.endswith("wiki"):
            for sub_dir in sub_dirs:
                if "packed" in sub_dir.lower() and not "perc" in sub_dir.lower():
                    ds = WikipediaData(save_dir=subdir, local_file_name=sub_dir)
                    ds.load_data(percentage=percentage)
                    print(f"\n Dataset {sub_dir}:")
                    for i in range(10):
                        print(ds.data[i])
        if subdir.endswith("books"):
            for sub_dir in sub_dirs:
                ds = BooksData(save_dir=subdir, local_file_name=sub_dir)
                ds.load_data(percentage=percentage)
                print(f"\n Dataset {sub_dir}:")
                for i in range(10):
                    print(ds.data[i])
        if subdir.endswith("openwebtext"):
            for sub_dir in sub_dirs:
                if "packed" in sub_dir.lower():
                    ds = OpenWebData(save_dir=subdir, local_file_name=sub_dir, local=local)
                    ds.load_data(percentage=percentage)
                    print(f"\n Dataset {sub_dir}:")
                    for i in range(10):
                        print(ds.data[i])


def check_dataset_lengths(params, percentage=None, local=True):
    print("\n\nDataset sizes:\n")
    base_path = params.args.data
    subdir_lists = [os.path.join(base_path, f) for f in os.listdir(base_path)]
    for subdir in subdir_lists:
        sub_dirs = [f for f in os.listdir(subdir)]
        if subdir.endswith("wiki"):
            for sub_dir in sub_dirs:
                if "packed" in sub_dir.lower() and not "perc" in sub_dir.lower():
                    ds = WikipediaData(save_dir=subdir, local_file_name=sub_dir)
                    ds.load_data(percentage=percentage)
                    total_path = os.path.join(os.path.join(base_path, subdir), sub_dir)
                    print(f"Dataset {sub_dir}:")
                    print(f"Length: {len(ds.data)}")
                    print(f"{print_size(total_path)} \n")
        if subdir.endswith("books"):
            for sub_dir in sub_dirs:
                ds = BooksData(save_dir=subdir, local_file_name=sub_dir)
                ds.load_data(percentage=percentage)
                total_path = os.path.join(os.path.join(base_path, subdir), sub_dir)
                print(f"Dataset {sub_dir}:")
                print(f"Length: {len(ds.data)}")
                print(f"{print_size(total_path)}\n")
        if subdir.endswith("openwebtext"):
            for sub_dir in sub_dirs:
                ds = OpenWebData(save_dir=subdir, local_file_name=sub_dir, local=local)
                ds.load_data(percentage=percentage)
                total_path = os.path.join(os.path.join(base_path, subdir), sub_dir)
                print(f"Dataset {sub_dir}:")
                print(f"Length: {len(ds.data)}")
                print(f"{print_size(total_path)}\n")


def data_loader_check(file_name, data_set_num, config: Config):
    torch.manual_seed(42)
    random.seed(23)
    print("Entering dataloader check function")
    print(f"File name: {file_name}")
    base_path = config.DATA_PATH
    print(f"Base path: {base_path}")
    if "books" in file_name.lower() or "book" in file_name.lower():
        print(f"Should load bookscorpus")
        ds = BooksData(save_dir=os.path.join(base_path, "books"), local_file_name=file_name)
    elif "wiki" in file_name.lower():
        print(f"Should load wikipedia")
        ds = WikipediaData(save_dir=os.path.join(base_path, "wiki"), local_file_name=file_name)
    else:
        ds = None
    if isinstance(ds, WikipediaData):
        print("Dataset is wikipedia")
    if isinstance(ds, BooksData):
        print(f"Dataset is bookscorpus")
    ds.load_data()
    ds = ds.data

    print(ds[0])

    if data_set_num == 1:
        ds = SimpleDataset1(ds)
    elif data_set_num == 2:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        ds = SimpleDataset2(ds, tokenizer)

    data_loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, timeout=60,
                             multiprocessing_context=torch.multiprocessing.get_context('spawn'))

    print(f"Iterating over file {file_name} (dataset: {data_set_num}):")
    i = 0
    total = len(data_loader) - 1
    timer = Timer()
    timer.start()
    for d in data_loader:
        text = d["text"]
        if i % 500 == 0 or i == total:
            print(f"finished batch {i} / {total}", flush=True)
        if i == 500:
            timer.stop()
            print(f"Took {timer.get_elapsed_minutes()} minutes", flush=True)
            timer.reset()
            timer.start()
        if i == 1000:
            timer.stop()
            print(f"Took {timer.get_elapsed_minutes()}")
        if i % 5000 == 0 or i == total:
            print(text[0], flush=True)

        i += 1

    print(f"Finished iteration", flush=True)
    print(f"\n{ds[100]}")


def check_tokenizer(config, tokenizer_name=None, text=None):
    if not text:
        text = "This is a sample text for checking the tokenizer output. This is the second sentence."

    if not tokenizer_name:
        tokenizer = Tokenizer(config)
        tokenizer.load_from_huggingface()
        vocab_size = tokenizer.get_vocab_size()
        tokenizer = tokenizer.get_tokenizer()
        result = tokenizer.tokenize(text)
        print("\nTokenized output of tokenizer (huggingface):")
        print(result)
        print(f"vocab_size: {vocab_size}")
    else:
        tokenizer = Tokenizer(config, tokenizer_name=tokenizer_name)
        tokenizer.load_local()
        vocab_size = tokenizer.get_vocab_size()
        tokenizer = tokenizer.get_tokenizer()
        result = tokenizer.tokenize(text)
        print(f"\nTokenized output of tokenizer ({tokenizer_name})")
        print(result)
        print(f"vocab_size: {vocab_size}")


def check_sentence_numbers(data, dir, file, local=True):
    if "openwebtext" in data.lower():
        ds = OpenWebData(save_dir=dir, local_file_name=file, local=local)
        ds.load_data()
        print(f"\nChecking amount of sentences in dataset {ds.name} (file {file})\n")
    else:
        ds = None

    length_dict = {}
    most_sentences = 0
    nltk.download("punkt")

    for i in range(len(ds.data)):
        text = ds.data[i]["text"]
        sentences = nltk.tokenize.sent_tokenize(text)
        length = len(sentences)
        if length > most_sentences:
            most_sentences = length
        if str(length) in length_dict:
            length_dict[str(length)] = length_dict.get(str(length)) + 1
        else:
            length_dict[str(length)] = 1

    print(f"Most sentences in a paragraph: {most_sentences}")
    print(f"Distribution of lengths:")
    pprint(length_dict)


if __name__ == '__main__':
    params = get_params()
    config = Config()
    params.set_config_from_script_params(config, printing=False)

    taurus = params.get_arg_value("taurus")
    if taurus:
        local = False
    else:
        local = True

    check_datasets(params, params.get_arg_value("perc"), local)

    check_dataset_lengths(params, params.get_arg_value("perc"))

    #check_sentence_numbers("openwebtext",
                           #"/home/katja/singularity/python-images/transformer_pretraining/python/train_data/openwebtext",
                           #"OpenWebText_1perc_sentence_splits_packed_512.hf")

    # check_tokenizer(config)
    # check_tokenizer(config, params.get_arg_value("tokenizer_name"))

    # dataloader_file = params.get_arg_value("dataloaderfile")
    # dataset_num = 1
    '''if params.get_arg_value("datasetnum"):
        dataset_num = int(params.args.datasetnum)
    if dataloader_file:
        data_loader_check(dataloader_file, dataset_num, config)
    print("FINISH")'''
