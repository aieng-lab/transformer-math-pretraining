"""Sentence Order Prediction"""

import random


def get_sentence_pair(index, text_corpus, is_last=False):
    sentence_a = text_corpus[index]
    index_a = sentence_a["id"]
    if not is_last:
        next_index = text_corpus[index + 1]["id"]
    else:
        next_index = None

    if next_index == index_a:
        # next sentence in document exists
        sentence_b = text_corpus[index + 1]
        prob = random.random()
        if prob > 0.5:
            first = sentence_a["text"]
            second = sentence_b["text"]
            is_swapped = False
        else:
            first = sentence_b["text"]
            second = sentence_a["text"]
            is_swapped = True

    else:
        # next sentence in document does not exist, take previous
        sentence_b = text_corpus[index - 1]
        prob = random.random()
        if prob > 0.5:
            first = sentence_b["text"]
            second = sentence_a["text"]
            is_swapped = False
        else:
            first = sentence_a["text"]
            second = sentence_b["text"]
            is_swapped = True

    return first, second, is_swapped
