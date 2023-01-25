from src.helpers.general_helpers import flatten

import torch
from torch.utils.data import Dataset
from typing import Callable
from transformers import BertTokenizer
from transformers import BertForMaskedLM


class PretrainingDataset(Dataset):

    def __init__(self, text_corpus, tokenizer, max_len,
                 processing_func: Callable):
        """
        :param text_corpus:
        :param tokenizer:
        :param max_len:
        :param processing_func: Function that returns dict of the form
        {"bert_input": tokens, "bert_label": {"label1: tokens, "label2": tokens, ...}, "segment_label": tokens}
        """
        self.tokenizer = tokenizer
        self.vocab = list(tokenizer.get_vocab().keys())
        self.max_len = max_len
        self.data = text_corpus
        self.proc_func = processing_func
        self.label_pad = 600

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        bert_dict = self.proc_func(self.data, index, self.tokenizer, self.vocab, self.max_len)

        bert_input = bert_dict.get("bert_input")[:self.max_len]
        bert_input_pad = bert_input + [self.tokenizer.pad_token_id] * (self.max_len - len(bert_input))

        attention_mask = ([1] * len(bert_input)) + ([0] * (self.max_len - len(bert_input)))

        segment_label = bert_dict.get("segment_label")[:self.max_len]
        segment_label_pad = segment_label + [segment_label[-1]] * (self.max_len - len(segment_label))

        bert_labels = bert_dict.get("bert_label")
        for key, value in bert_labels.items():
            bert_labels[key] = torch.tensor(value)


        #todo: pad mlm and nsp labels to the same length as well

        return flatten({
            "bert_input": torch.tensor(bert_input_pad),
            "bert_label": bert_labels,
            "attention_mask": torch.tensor(attention_mask),
            "segment_label": torch.tensor(segment_label_pad)
        })
