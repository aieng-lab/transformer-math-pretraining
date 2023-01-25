from src.pretraining_methods.mlm_like.MLM import prepare as mlm_prep
from src.pretraining_methods.specialized.PROP import prepare as prop_prep

import random
import math


def prop_plus_mlm(text_corpus, index, tokenizer, vocab, max_len, epoch_num, is_last=False):
    target_len = max_len
    max_tokens = target_len - 3

    document, word_sets, rop_label = prop_prep.get_doc_word_pairs(index, text_corpus, epoch_num)

    document_tokens = tokenizer.tokenize(document)

    input_id_tuple = []
    mlm_label_tuple = []
    segment_label_tuple = []

    for i in range(2):
        word_set = word_sets[i]
        word_set_tokens = tokenizer.tokenize(word_set)

        document_tokens_i = document_tokens[:]  # make a copy of the document tokens for each iteration

        seq_len = len(document_tokens_i) + len(word_set_tokens)
        while seq_len > max_tokens:
            document_tokens_i.pop()
            seq_len = len(document_tokens_i) + len(word_set_tokens)
        assert seq_len <= max_tokens

        masked_tokens, labels, label_positions = mlm_prep.mask_tokens(document_tokens_i, tokenizer, vocab,
                                                                      len(document_tokens_i))

        tokens = [tokenizer.cls_token] + word_set_tokens + [tokenizer.sep_token] + masked_tokens + [tokenizer.sep_token]
        offset = len(word_set_tokens) + 2

        label_positions = [position + offset for position in label_positions]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        mlm_label_ids = tokenizer.convert_tokens_to_ids(labels)
        mlm_labels = [-100 if index not in label_positions else mlm_label_ids.pop(0) for index, id in
                      enumerate(input_ids)]
        mlm_labels = mlm_labels + ([-100] * (max_len - len(mlm_labels)))

        segment_ids = ([0] * (len(word_set_tokens) + 2)) + ([1] * (len(document_tokens) + 1))

        input_id_tuple.append(input_ids)
        mlm_label_tuple.append(mlm_labels)
        segment_label_tuple.append(segment_ids)

    rop_label = [int(rop_label)]

    return {
        "bert_input": tuple(input_id_tuple),
        "bert_label": {"MLM": tuple(mlm_label_tuple), "PROP": rop_label},
        "segment_label": tuple(segment_label_tuple),
    }
