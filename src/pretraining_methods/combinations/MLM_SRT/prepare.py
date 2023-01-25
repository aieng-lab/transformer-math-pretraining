from src.pretraining_methods.mlm_like.MLM import prepare as mlm_prep
from src.pretraining_methods.nsp_like.SRT import prepare as srt_prep

import random


def mlm_plus_srt(text_corpus, index, tokenizer, vocab, max_len, is_last=False):
    permuted_text, srt_label = srt_prep.get_permuted_text(index, text_corpus)

    length = len(permuted_text)
    special_token_num = length + 1

    tokenized_perm_text = []
    token_length = 0
    for segment in permuted_text:
        tokenized = tokenizer.tokenize(segment)
        tokenized_perm_text.append(tokenized)
        token_length += len(tokenized)

    while token_length + special_token_num > 512:
        # crop the text in some way
        random_num = random.randrange(length)
        tokenized_perm_text = [t if i != random_num else (t[:-5]) for i, t in enumerate(tokenized_perm_text)]
        token_length = sum(len(t) for t in tokenized_perm_text)

    tokens = [tokenizer.cls_token]
    for t in tokenized_perm_text:
        tokens += t
        tokens += [tokenizer.sep_token]

    if len(tokens) > max_len:
        tokens = tokens[:max_len - 1]
        tokens = tokens + [tokenizer.sep_token]

    masked_tokens, labels, label_positions = mlm_prep.mask_tokens(tokens, tokenizer, vocab, token_length)

    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    mlm_label_ids = tokenizer.convert_tokens_to_ids(labels)
    mlm_labels = [-100 if index not in label_positions else mlm_label_ids.pop(0) for index, id in enumerate(input_ids)]
    mlm_labels = mlm_labels + ([-100] * (max_len - len(mlm_labels)))

    srt_label = [int(srt_label)]

    segment_ids = ([0] * max_len)

    return {
        "bert_input": input_ids,
        "bert_label": {"MLM": mlm_labels, "SRT": srt_label},
        "segment_label": segment_ids,
    }
