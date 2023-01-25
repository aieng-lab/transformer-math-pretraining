"""Next Sentence Prediction"""


import random
from transformers import BertTokenizerFast
import os

from src.pretraining_methods.Objectives import Objectives


def add_special_tokens(first_sent_tokens, second_sent_tokens, tokenizer: BertTokenizerFast):
    first_sentence = [tokenizer.cls_token] + first_sent_tokens + [tokenizer.sep_token]
    second_sentence = second_sent_tokens + [tokenizer.sep_token]
    return first_sentence, second_sentence

def only_ir(text_corpus, index, tokenizer, vocab, max_len, is_last=False, objective=Objectives.IR):
    target_len = max_len
    max_tokens = target_len - 3

    d = text_corpus[index]

    sentence_a = d['query']
    sentence_b = d['document']
    is_next = d['label']

    sent_a_tokens = tokenizer.tokenize(sentence_a, truncation=True)[:255] # truncate to first 255 tokens, in case of NMF this should never be above anyways
    sent_b_tokens = tokenizer.tokenize(sentence_b, truncation=True)
    sent_b_max_len = max_len - len(sent_a_tokens) - 3
    sent_b_tokens = sent_b_tokens[:sent_b_max_len]

    tokens = [tokenizer.cls_token] + sent_a_tokens + [tokenizer.sep_token] + sent_b_tokens + [tokenizer.sep_token]

    nsp_label = [int(is_next)]

    segment_ids = ([0] * (len(sent_a_tokens) + 2)) + ([1] * (len(sent_b_tokens) + 1))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return {
        "bert_input": input_ids,
        "bert_label": {objective.name: nsp_label},
        "segment_label": segment_ids,
    }

