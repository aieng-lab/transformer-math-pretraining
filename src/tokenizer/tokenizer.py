from src.config.config import Config

from transformers import BertTokenizerFast, AutoTokenizer


class TokenizerException(Exception):
    pass


class Tokenizer:

    def __init__(self, config: Config, tokenizer_name="BertTokenizerFast", base_name="bert-base-cased"):
        self.tokenizer_name = tokenizer_name
        self.base_name = base_name
        self.config = config
        self.tokenizer: BertTokenizerFast = None

    def get_vocab_size(self):
        print(f"Len of list of keys: {len(list(self.tokenizer.get_vocab()))}", flush=True)
        print(f"Direct vocab size: {self.tokenizer.vocab_size}", flush=True)
        print(f"Len of tokenizer: {len(self.tokenizer)}", flush=True)
        return len(list(self.tokenizer.get_vocab().keys()))

    def get_vocab(self):
        return self.tokenizer.get_vocab()

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

    def load_from_huggingface(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_name)

    def get_tokenizer(self):
        # set global vocab size according to the selected tokenizer
        self.config.VOCAB_SIZE = len(list(self.tokenizer.get_vocab().keys()))
        return self.tokenizer

    def test_tokenizer(self, text=None):
        print(self.tokenizer.special_tokens_map)
        if not text:
            text = "This is an example to see if the tokenizer is working properly. We would like to compare the " \
                   "current tokenizer to the original huggingface tokenizer. I'm having difficulty thinking of a " \
                   "better example"
        hf_tokenizer = BertTokenizerFast.from_pretrained(self.base_name)
        hf_tokenized = hf_tokenizer.tokenize(text)
        tokenized = self.tokenizer.tokenize(text)
        print(text)
        print(f"HF: {hf_tokenized}")
        print(f"Custom: {tokenized}")
