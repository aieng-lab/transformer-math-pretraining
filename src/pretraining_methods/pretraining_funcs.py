import math

from src.pretraining_methods.MLM import mlm_prep
from src.pretraining_methods.NSP import nsp_prep

import random


def mlm_plus_nsp(text_corpus, index, tokenizer, vocab, max_len):
    target_len = max_len
    max_tokens = target_len - 3

    sentence_a, sentence_b, is_next = nsp_prep.get_sentence_pair(index, text_corpus)

    sent_a_tokens = tokenizer.tokenize(sentence_a)
    sent_b_tokens = tokenizer.tokenize(sentence_b)

    seq_len = len(sent_a_tokens) + len(sent_b_tokens)
    if seq_len > max_tokens:
        first_crop_prob = random.random()
        if first_crop_prob < 0.5:
            sent_a_tokens = sent_a_tokens[:(math.floor(max_tokens / 2))]
            sent_b_tokens = sent_b_tokens[:(max_tokens - len(sent_a_tokens))]
        else:
            sent_b_tokens = sent_b_tokens[:(math.floor(max_tokens / 2))]
            sent_a_tokens = sent_a_tokens[:(max_tokens - len(sent_b_tokens))]
    seq_len = len(sent_a_tokens) + len(sent_b_tokens)
    assert seq_len <= max_tokens

    tokens = [tokenizer.cls_token] + sent_a_tokens + [tokenizer.sep_token] + sent_b_tokens + [tokenizer.sep_token]

    masked_tokens, masked_positions, labels = mlm_prep.mask_tokens(tokens, tokenizer, vocab, seq_len)

    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    mlm_labels = tokenizer.convert_tokens_to_ids(labels)
    mlm_labels = mlm_labels + ([0] * (max_len - len(mlm_labels)))
    masked_positions = masked_positions + ([600] * (max_len - len(masked_positions)))
    nsp_label = [int(is_next)] + ([600] * (max_len - 1))

    segment_ids = ([0] * (len(sent_a_tokens) + 2)) + ([1] * (len(sent_b_tokens) + 1))
    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids)
    original_text = f"{sentence_a} | {sentence_b}"

    return {
        "original_text": original_text,
        "tokenized_text": tokenized_text,
        "bert_input": input_ids,
        "bert_label": {"mlm": mlm_labels, "nsp": nsp_label, "mask_positions": masked_positions},
        "segment_label": segment_ids,
    }
