import math

from transformers import BertTokenizerFast, AutoModelForMaskedLM
import random


# todo: loss function over masked tokens only


def mask_tokens(tokens, tokenizer: BertTokenizerFast, vocabulary: list, seq_len):
    """
    :param tokens:
    :param tokenizer:
    :param vocabulary:
    :param seq_len: Number of tokens that are not special tokens
    :return:
    """
    rand_positions = []
    visited = []
    special_tokens = tokenizer.all_special_tokens
    already_masked = 0

    while True:
        index = random.randint(0, len(tokens) - 1)
        if index not in visited and tokens[index] not in special_tokens:
            # position is chosen for masking
            already_masked += 1
            masked_percent = already_masked / seq_len
            if masked_percent > 0.15:
                # abort when more than 15% of tokens would be masked
                break

            visited.append(index)
            rand_positions.append((index, tokens[index]))

            prob = random.random()
            if prob < 0.8:
                # replace token with MASK
                tokens[index] = tokenizer.mask_token

            elif prob < 0.9:
                # replace with random token
                rand_token = vocabulary[random.randrange(len(vocabulary))]
                tokens[index] = rand_token

            else:
                # keep original token
                pass

    rand_positions.sort(key=lambda x: x[0])
    mask_positions = [item[0] for item in rand_positions]
    labels = [item[1] for item in rand_positions]

    return tokens, mask_positions, labels


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    vocab = list(tokenizer.get_vocab().keys())
    sentence_a = "This is the first sentence that I can think of."
    sentence_b = "And this is the second."

    a_tokens = tokenizer.tokenize(sentence_a)
    b_tokens = tokenizer.tokenize(sentence_b)

    seq_len = len(a_tokens) + len(b_tokens)
    tokens = [tokenizer.cls_token] + a_tokens + [tokenizer.sep_token] + b_tokens + [tokenizer.sep_token]

    print(tokens)

    tokens, positions, labels = mask_tokens(tokens, tokenizer, vocab, seq_len)
    print(tokens)
    print(positions)
    print(labels)
