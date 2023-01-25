"""Sentence Reordering Task (ERNIE 2.0)"""

import random
from itertools import permutations
import nltk
import math


def get_permuted_text(index, text_corpus, with_sim_words=False):
    item = text_corpus[index]
    text = item["text"]
    og_sentences = nltk.tokenize.sent_tokenize(text)
    length = len(og_sentences)
    if length >= 5:
        segments = 5
    else:
        segments = length

    sentences = og_sentences

    merge_len = math.ceil(len(sentences) / segments)

    new_sentences = []
    merged_len = 0
    merged = ""

    for i, sentence in enumerate(sentences):
        if merged_len < merge_len:
            merged += f" {sentence}"
            merged_len += 1
        if merged_len == merge_len:
            new_sentences.append(merged)
            merged = ""
            merged_len = 0

    if merged_len > 0:
        new_sentences.append(merged)

    sentences = new_sentences
    segments = len(sentences)

    # find permutation

    perms = [p for p in permutations(sentences)]

    rand_ind = random.randrange(len(perms))

    permutation = perms[rand_ind]

    if segments > 1:
        label_add = 0
        pred = segments - 1
        while pred != 0:
            label_add += math.factorial(pred)
            pred -= 1
        label = label_add + rand_ind
    else:
        label = rand_ind

    if with_sim_words:
        sim_words = item["similar_words"]
        rand_words = item["random_words"]
        return permutation, label, sim_words, rand_words

    return permutation, label

