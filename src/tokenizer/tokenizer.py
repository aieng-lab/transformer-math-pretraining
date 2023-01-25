from src.helpers.general_helpers import create_path_if_not_exists, path_exists
from src.config.config import Config

from transformers import BertTokenizerFast
from tqdm import tqdm
import os
from pathlib import Path
import logging

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TokenizerException(Exception):
    pass


class Tokenizer:

    def __init__(self, config: Config, tokenizer_name, base_name="bert-base-cased",):
        self.tokenizer_name = tokenizer_name
        self.base_name = base_name
        self.config = config
        self.tokenizer = None
        self.data = None

    def set_training_data(self, data):
        self.data = data

    def customize_special_tokens(self, tokens: list):
        if self.tokenizer is None:
            raise TokenizerException("No tokenizer was loaded. Cannot customize special tokens.")
        new_tokens = []
        for token in tokens:
            if token in self.tokenizer.all_special_tokens:
                continue
            else:
                new_tokens.append(token)
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    def get_batch_iterator(self, batch_size=10000):
        for i in tqdm(range(0, len(self.data), batch_size)):
            yield self.data[i: i + batch_size]["text"]

    def train(self, batch_size=10000, vocab_size=32000):
        trained_tokenizer = self.tokenizer.train_new_from_iterator(self.get_batch_iterator(batch_size),
                                                                   vocab_size=vocab_size)
        self.tokenizer = trained_tokenizer

    def save(self, path):
        create_path_if_not_exists(path)
        self.tokenizer.save_pretrained(path)

    def load_local(self, path):
        path = Path(path)
        self.tokenizer = BertTokenizerFast.from_pretrained(path)

    def load_custom(self, vocab_size=32000, local=True, batch_size=10000):
        print("Loading custom tokenizer ...")
        if local:
            print("Searching for local instance of tokenizer...")
            tokenizer_path = os.path.join(self.config.TOKENIZER_PATH, self.tokenizer_name)
            if path_exists(tokenizer_path):
                print(f"Found local tokenizer {self.tokenizer_name}. Loading from disk.")
                self.load_local(tokenizer_path)
        if self.tokenizer is None:
            if self.data is None:
                raise TokenizerException("No local tokenizer was found, but no training data was provided either.")
            if local:
                print("No local instance of tokenizer found.")
            print("Training custom tokenizer...")
            self.tokenizer = BertTokenizerFast.from_pretrained(self.base_name)
            self.train(vocab_size=vocab_size, batch_size=batch_size)

    def load_from_huggingface(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(self.base_name)

    def get_tokenizer(self):
        # set global vocab size according to the selected tokenizer
        self.config.vocab_size = len(list(self.tokenizer.get_vocab().keys()))
        return self.tokenizer



