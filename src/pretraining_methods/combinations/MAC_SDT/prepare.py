"""MAC + SDT"""

from src.pretraining_methods.mlm_like.MAC import prepare as mac_prep
from src.pretraining_methods.nsp_like.SDT import prepare as sdt_prep

import random
import math
import json


def mac_plus_sdt(text_corpus, index, tokenizer, vocab, max_len, is_last=False):
    target_len = max_len
    max_tokens = target_len - 3

    sentence_a, sentence_b, label, sim_words_a, sim_words_b, rand_words_a, rand_words_b = sdt_prep.get_sentence_pair(
        index, text_corpus, is_last, with_sim_words=True)

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

    sim_words_a = json.loads(sim_words_a)
    sim_words_b = json.loads(sim_words_b)
    sim_words_dict = {**sim_words_a, **sim_words_b}

    rand_words = f"{rand_words_a} {rand_words_b}"
    rand_words_list = rand_words.split(" ")

    corrupted_tokens, labels, label_positions = mac_prep.corrupt_tokens(tokens, tokenizer, vocab, seq_len,
                                                                        sim_words_dict, rand_words_list)

    input_ids = tokenizer.convert_tokens_to_ids(corrupted_tokens)

    mac_label_ids = tokenizer.convert_tokens_to_ids(labels)
    mac_labels = [-100 if index not in label_positions else mac_label_ids.pop(0) for index, id in enumerate(input_ids)]
    mac_labels = mac_labels + ([-100] * (max_len - len(mac_labels)))

    sdt_label = [int(label)]

    segment_ids = ([0] * (len(sent_a_tokens) + 2)) + ([1] * (len(sent_b_tokens) + 1))

    return {
        "bert_input": input_ids,
        "bert_label": {"MAC": mac_labels, "SDT": sdt_label},
        "segment_label": segment_ids,
    }
