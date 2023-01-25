"""Sentence Distance Task"""

import random


def get_sentence_pair(index, text_corpus, is_last=False, with_sim_words=False):
    # todo: do this in a more structured way: each case degenerates to the next if something goes wrong:
    sentence_a = text_corpus[index]
    index_a = sentence_a["id"]

    prob = random.random()
    sentence_b = None

    if prob > (2 / 3):
        # take next sentence
        try:
            sentence_b = text_corpus[index + 1]
        except IndexError:
            sentence_b = None
        label = 0

    if (1 / 3) < prob <= (2 / 3) or (prob > (2/3) and sentence_b is None):
        # take sentence from same document
        already_tried = []

        i = 0
        while len(already_tried) < 20 and i < 100:
            rand_idx = random.randrange(1, 20)
            if rand_idx in already_tried:
                i += 1
                continue

            try:
                if not rand_idx == 1:
                    cand_plus = text_corpus[index + rand_idx]
                else:
                    cand_plus = None
            except IndexError:
                cand_plus = None

            try:
                cand_min = text_corpus[index - rand_idx]
            except IndexError:
                cand_min = None

            if cand_plus:
                if cand_plus["id"] == index_a and not (cand_plus["text"] == sentence_a["text"]):
                    sentence_b = cand_plus
                    break
            if cand_min and not sentence_b:
                if cand_min["id"] == index_a and not (cand_min["text"] == sentence_a["text"]):
                    sentence_b = cand_min
                    break

            already_tried.append(rand_idx)
            i += 1

        label = 1

    if 0 <= prob <= (1 / 3) or (prob > (1/3) and not sentence_b):
        # take sentence from other document
        sentence_b = text_corpus[random.randrange(len(text_corpus))]
        while sentence_b == sentence_a or index_a == sentence_b["id"]:
            sentence_b = text_corpus[random.randrange(len(text_corpus))]
        label = 2

    if with_sim_words:
        sim_words_a = sentence_a["similar_words"]
        sim_words_b = sentence_b["similar_words"]
        rand_words_a = sentence_a["random_words"]
        rand_words_b = sentence_b["random_words"]
        return sentence_a["text"], sentence_b["text"], label, sim_words_a, sim_words_b, rand_words_a, rand_words_b

    return sentence_a["text"], sentence_b["text"], label
