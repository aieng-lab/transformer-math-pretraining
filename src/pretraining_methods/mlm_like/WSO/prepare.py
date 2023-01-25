"""
Word Shuffling Objective (StructBERT)
trigram sampling rate: 5% (of unmasked or overall) --> 5% of disjoint unmasked trigrams
alongside with MLM (normal 15%)
"""
from src.pretraining_methods.mlm_like.MLM.prepare import mask_tokens
from src.helpers.Timer import Timer

from transformers import BertTokenizerFast
import random


def shuffle_words(tokens, tokenizer: BertTokenizerFast, vocabulary: list, seq_len):
    """
    :param tokens:
    :param tokenizer:
    :param vocabulary:
    :param seq_len: Number of tokens that are not special tokens
    :return:
    """

    masked_tokens, mask_labels, mask_label_positions = mask_tokens(tokens, tokenizer, vocabulary, seq_len)

    special_tokens = tokenizer.all_special_tokens
    trigrams = []
    trigram_positions = []

    full_words = 0
    current_trigram = []
    current_word = []

    for index, token in enumerate(masked_tokens):
        if not current_word and token.startswith("##"):
            continue
        if (index in trigram_positions or token in special_tokens or index in mask_label_positions):
            current_trigram = []
            current_word = []
            full_words = 0
            continue
        current_word.append(index)
        if index != len(masked_tokens) - 1:
            next_token = masked_tokens[index + 1]
            if not next_token.startswith("##"):
                # completed a word
                current_trigram.append(current_word)
                current_word = []
                full_words += 1
                if full_words == 3:
                    trigrams.append(current_trigram)
                    for word in current_trigram:
                        for index in word:
                            trigram_positions.append(index)
                    current_trigram = []
                    full_words = 0
        if index == len(tokens) - 1:
            if not token.startswith("##"):
                full_words += 1

    if trigrams:
        chosen_trigrams = []
        while len(chosen_trigrams) / len(trigrams) < 0.05:
            trigram = random.choice(trigrams)
            while trigram in chosen_trigrams:
                trigram = random.choice(trigrams)
            chosen_trigrams.append(trigram)
    else:
        chosen_trigrams = []

    shuffled_positions = {}

    for trigram in chosen_trigrams:
        start_position = trigram[0][0]
        new_positions = [0, 1, 2]
        random.shuffle(new_positions)
        for i, position in enumerate(new_positions):
            if i > 0:
                start_position = start_position + len(trigram[new_positions[i-1]])
            word = trigram[position]
            for i, index in enumerate(word):
                shuffled_positions[start_position + i] = index

    shuffled_tokens = []
    shuffle_labels = []
    shuffle_label_positions = []

    for i, token in enumerate(masked_tokens):
        if i in shuffled_positions:
            new_index = shuffled_positions.get(i)
            shuffled_tokens.append(masked_tokens[new_index])
            shuffle_labels.append(token)
            shuffle_label_positions.append(i)
        else:
            shuffled_tokens.append(token)

    labels = []
    label_positions = []

    for i, token in enumerate(masked_tokens):
        if i in mask_label_positions and i in shuffle_label_positions:
            raise AssertionError("Token was chosen for masking and shuffling")
        if i in mask_label_positions:
            label_positions.append(i)
            labels.append(mask_labels.pop(0))
        if i in shuffle_label_positions:
            label_positions.append(i)
            labels.append(shuffle_labels.pop(0))


    return shuffled_tokens, labels, label_positions


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    vocab = list(tokenizer.get_vocab().keys())

    text_a = """Cerro Tuzgle is a volcano near the eastern border of the Argentina Puna.[4] Politically, it is part of the Susques Department of the Jujuy Province.[5] San Antonio de Los Cobres lies 45 km (28 miles) from Cerro Tuzgle and Susques 75 km (47 miles),[6] while the cities of Salta and San Salvador de Jujuy are 280 km (170 miles) and 170 km (110 miles) away, respectively.[5] A locality called "Sey" lies northwest of Cerro Tuzgle.[7] The volcano is visible from Provincial Route 74.[8] Its name, which is also rendered as Tujle, Tugle or Tugler, comes from the Kunza language; it means "knoll" and refers to the shape of the volcano."""

    text_b = """Cerro Tuzgle is a simple volcanic cone[10] and is the largest in the back-arc region of the Andes.[11] It is a well-preserved stratovolcano that rises 1.2 km (0.75 miles)[8] from a surrounding terrain at c. 3.7 km (2.3 miles) elevation[4] to a summit at 5,486 metres (17,999 ft) elevation.[1][2] A 0.5-square-kilometre (0.19 sq mi) platform lies at the summit of the volcano.[12] The mountain is occasionally snow-covered[6] and frost weathering has produced patterned ground[13] and blockfields. In 1926 it was reported that a crater lake lies on the summit."""

    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    #tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token] + tokens_b + [tokenizer.sep_token]

    tokens = ['[CLS]', 'Tim', '##ing', 'how', 'long', 'the', 'signals', '(', 'traveling', 'at', 'the', 'universal', 'limit', 'of', 'the', 'speed', 'of', 'light', ')', 'take', 'to', 'make', 'the', 'round', 'trip', ',', 'na', '##vigator', '##s', 'can', 'calculate', 'how', 'far', 'the', 'probe', 'is', 'and', 'hence', 'where', 'it', 'is', 'in', 'its', 'orbit', '.', '[SEP]', 'Co', '##mb', '##ining', 'these', '(', 'and', 'including', 'other', 'information', 'as', 'well', ')', 'allows', 'them', 'to', 'com', '##pute', 'how', 'strongly', 'Ce', '##res', 'pulls', 'on', 'its', 'orbital', 'companion', '.', 'The', 'strength', 'of', 'its', 'gravitational', 'force', 'reveals', 'its', 'he', '##ft', '.', '[SEP]']

    timer = Timer()
    timer.start()

    shuffled_tokens, labels, label_positions = shuffle_words(tokens, tokenizer, vocab, 128)

    timer.stop()
    timer.print_elapsed()

    labels = [-100 if index not in label_positions else labels.pop(0) for index, token in enumerate(tokens)]

    print("\n\n")

    original_tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token] + tokens_b + [tokenizer.sep_token]

    print(f"{'Original':<25}{'Shuffled':<25}{'Labels':25}")
    for i in range(len(tokens)):
        print(f"{original_tokens[i]:<25}{shuffled_tokens[i]:<25}{labels[i]}")
