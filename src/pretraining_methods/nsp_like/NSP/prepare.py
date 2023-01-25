"""Next Sentence Prediction"""


import random
from transformers import BertTokenizerFast
import os

from src.pretraining_methods.Objectives import Objectives


def get_sentence_pair(index, text_corpus, is_last=False, with_sim_words=False):
    sentence_a = text_corpus[index]
    index_a = sentence_a["id"]
    if not is_last:
        next_index = text_corpus[index + 1]["id"]
    else:
        next_index = None
    prob = random.random()
    if prob > 0.5 and not is_last and index_a == next_index:
        is_next = True
        sentence_b = text_corpus[index + 1]
    else:
        is_next = False
        sentence_b = text_corpus[random.randrange(len(text_corpus))]
        i = 0
        while sentence_b == sentence_a or (not is_last and sentence_b == text_corpus[index + 1]):
            sentence_b = text_corpus[random.randrange(len(text_corpus))]
            i += 1
            if i > 2:
                print(f"i = {i}")

    if with_sim_words:
        sim_words_a = sentence_a["similar_words"]
        sim_words_b = sentence_b["similar_words"]
        rand_words_a = sentence_a["random_words"]
        rand_words_b = sentence_b["random_words"]
        return sentence_a["text"], sentence_b["text"], is_next, sim_words_a, sim_words_b, rand_words_a, rand_words_b

    else:
        return sentence_a["text"], sentence_b["text"], is_next


def add_special_tokens(first_sent_tokens, second_sent_tokens, tokenizer: BertTokenizerFast):
    first_sentence = [tokenizer.cls_token] + first_sent_tokens + [tokenizer.sep_token]
    second_sentence = second_sent_tokens + [tokenizer.sep_token]
    return first_sentence, second_sentence

def only_nfir(text_corpus, index, tokenizer, vocab, max_len, is_last=False):
    return only_nsp(text_corpus, index, tokenizer, vocab, max_len, is_last, objective=Objectives.NFIR)

def only_ffir(text_corpus, index, tokenizer, vocab, max_len, is_last=False):
    return only_nsp(text_corpus, index, tokenizer, vocab, max_len, is_last, objective=Objectives.FFIR)

def only_nsp(text_corpus, index, tokenizer, vocab, max_len, is_last=False, objective=Objectives.NSP):
    target_len = max_len
    max_tokens = target_len - 3

    d = text_corpus[index]
    keys = d.keys()
    if all(key in keys for key in ['name', 'formula', 'label']):
        sentence_a = d['name']
        sentence_b = d['formula']
        is_next = d['label']
    elif all(key in keys for key in ['formula1', 'formula2', 'label']):
        sentence_a = d['formula1']
        sentence_b = d['formula2']
        is_next = d['label']
    else:
        raise ValueError("Got unknown key combination: %s" % keys)

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


if __name__ == "__main__":
    text_corp = ["This is the first sentence.", "And this is the second sentence.",
                 "After the second comes the third sentence.",
                 "And after that the fourth ome.", "Here we have the fifth sentence.", "And finally the sixth one."]
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    sent_a, sent_b, is_next = get_sentence_pair(2, text_corp)
    print(sent_a)
    print(sent_b)
    print(is_next)
    sent_a = tokenizer.tokenize(sent_a)
    sent_b = tokenizer.tokenize(sent_b)
    special_a, special_b = add_special_tokens(sent_a, sent_b, tokenizer)
    print(special_a)
    print(special_b)
