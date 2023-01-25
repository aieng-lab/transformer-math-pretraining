"""
Span Masking Objective

- iteratively sampling spans of text (subset of all tokens) until masking budget of 15% has been spent
- first sample span length from geometric distribution (skewed towards smaller spans)
- geometric distribution: l ~ Geo(p), p = 0.2, lmax = 10, mean span length = 3.8
- always mask full words, beginning of sequence must be beginning of a word
- replace 80% with mask, 10% with random, 10% original (decided at span level)

"""

from transformers import BertTokenizerFast
import random
from numpy import random as npr
import re


def mask_spans(tokens, tokenizer: BertTokenizerFast, vocabulary: list, seq_len):
    span_positions = []
    special_tokens = tokenizer.all_special_tokens
    already_masked = 0
    labels = []
    label_positions = []
    span_dict = {}
    special_chars = r"[^\w\s\d]"

    span_finding_iterations = 0
    while span_finding_iterations < 600:
        span_finding_iterations += 1

        current_span = []
        current_already_masked = already_masked

        # sample span length
        length = 11
        geometric_sample_iterations = 0
        while length > 10 and geometric_sample_iterations < 100:
            length = npr.geometric(p=0.2)
            geometric_sample_iterations += 1
        if length > 10:
            length = random.choice([3, 4])  # (paper says mean span length is 3.8)

        # find start of span
        index = random.randint(0, len(tokens) - 1)
        token = tokens[index]

        while (token.startswith(
                "##") or re.match(special_chars,
                                  token)) and index >= 1 and token not in special_tokens and index not in span_positions and token not in special_tokens:
            index = index - 1
            token = tokens[index]
            if re.match(special_chars, token):
                # keep iterating if the token is a special character (no special characters as start of span)`
                token = f"##{token}"

        # start token is found
        if (index in span_positions or token in special_tokens) or (
                index > 0 and (index - 1 in span_positions) or (tokens[index - 1] in special_tokens)):
            # keep at least one token between spans or special tokens
            continue

        if (current_already_masked + 1) / seq_len <= 0.15:
            current_span.append(index)
            current_already_masked += 1
        else:
            break

        full_words = 0
        overlap = False

        while full_words < length and (current_already_masked / seq_len) < 0.15 and index < len(tokens) - 1:
            index += 1
            if index in span_positions or (index < len(tokens) - 1 and index + 1 in span_positions):
                # keep at least one token between spans
                overlap = True
                break
            token = tokens[index]
            if token in special_tokens:
                overlap = True
                break
            if not token.startswith("##"):
                full_words += 1
                if full_words == length:
                    break
            if (current_already_masked + 1) / seq_len <= 0.15:
                current_span.append(index)
                current_already_masked += 1
            else:
                break

        if overlap:
            continue

        last = current_span[-1]
        if last < len(tokens):
            next = tokens[last + 1]
            while next.startswith("##") and last > 0:
                # span does not end on full word
                # crop tokens from end until it does
                current_span.pop()
                current_already_masked -= 1
                if len(current_span) == 0:
                    break
                last = current_span[-1]
                next = tokens[last + 1]

        span_positions.extend(current_span)
        already_masked = current_already_masked

        # decide on replacement strategy
        prob = random.random()
        if prob <= 0.8:
            replace = "mask"
        elif prob <= 0.9:
            replace = "random"
        else:
            replace = "no"

        for id in current_span:
            span_dict[id] = replace

    # indices for replacing are determined

    for index, item in enumerate(tokens):

        if index in span_positions:
            # token is part of span

            if span_dict.get(index) == "mask":
                # replace token with mask, add original token to labels
                labels.append(tokens[index])
                label_positions.append(index)
                tokens[index] = tokenizer.mask_token
            elif span_dict.get(index) == "random":
                # replace token with random token, add original token to labels
                labels.append(tokens[index])
                label_positions.append(index)
                rand_token = vocabulary[random.randrange(len(vocabulary))]
                tokens[index] = rand_token
            elif span_dict.get(index) == "no":
                # keep original token
                labels.append(tokens[index])
                label_positions.append(index)
        else:
            # token was not chosen as mask candidate, do not add token to labels
            pass

    return tokens, labels, label_positions


if __name__ == "__main__":
    text = """The ferry company was founded in 1967 by the Corsican Pascal Lota under the name of Corsica Line with one ferry, the Corsica Express. Since the company’s humble beginnings it has continuously grown and is today the market leader to Corsica and Sardinia. Corsica Sardinia Ferries is the premier ferry operator on the Western Mediterranean Sea transporting more than 2.8 million passengers annually aboard their ferries running to and from France and Italy to Corsica, Sardinia and Elba.[2] Corsica Sardinia Ferries operate two return routes from Italy to Sardinia and 10 return routes from France and Italy to Corsica, with a total of up to 13 crossings daily. The fleet currently consists of 14 vessels. Thanks to its no-frills economical model, the company today holds a market share of 68,7% of the maritime traffic to Corsica (2017).[3] In 2005, Forship Spa, a subsidiary of Corsica Sardinia Ferries, has been condemned to pay 490 000 euros by the Tribunal of Marseille for having "voluntary discharged at sea" offshore of the Cap Corse on May 12, 2004."""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    tokenized = tokenizer.tokenize(text)
    seq_len = len(tokenized)
    # tokens = [tokenizer.cls_token] + tokenized + [tokenizer.sep_token]
    vocab = list(tokenizer.get_vocab().keys())

    tokens = ['[CLS]', '[UNK]', 'ہ', '##م', '##ا', '##ر', '##ا', 'ت', '##ا', '##ج', 'م', '##ح', '##ل', 'ہ', '##ے', 'ا',
              '##و', '##ر', 'ق', '##ط', '##ب', 'م', '##ی', '##ن', '##ا', '##ر', 'ہ', '##ے', '[UNK]', 'ہ', '##م', '##ا',
              '##ر', '##ا', 'ت', '##ا', '##ج', 'م', '##ح', '##ل', 'ہ', '##ے', 'ا', '##و', '##ر', 'ق', '##ط', '##ب', 'م',
              '##ی', '##ن', '##ا', '##ر', 'ہ', '##ے', '[UNK]', 'ہ', '##م', '##ا', '##ر', '##ے', 'م', '##ن', '##د',
              '##ر', 'م', '##س', '##ج', '##د', '[UNK]', 'ک', '##ا', 'گ', '##ر', '##د', '##و', '##ا', '##ر', '##ہ', 'ہ',
              '##ے', '[UNK]', 'ہ', '##م', '##ا', '##ر', '##ے', 'م', '##ن', '##د', '##ر', 'م', '##س', '##ج', '##د',
              '[UNK]', 'ک', '##ا', 'گ', '##ر', '##د', '##و', '##ا', '##ر', '##ہ', 'ہ', '##ے', '[UNK]', 'ہ', '##م',
              '##ا', '##ر', '##ا', 'ت', '##ا', '##ج', 'م', '##ح', '##ل', 'ہ', '##ے', 'ا', '##و', '##ر', 'ق', '##ط',
              '##ب', 'م', '##ی', '##ن', '##ا', '##ر', 'ہ', '##ے', '[UNK]', 'ہ', '##م', '##ا', '##ر', '##ے', 'م', '##ن',
              '##د', '##ر', 'م', '##س', '##ج', '##د', '[UNK]', 'ک', '##ا', 'گ', '##ر', '##د', '##و', '##ا', '##ر',
              '##ہ', 'ہ', '##ے', 'ا', '##س', '[UNK]', 'پ', '##ر', 'ق', '##د', '##م', '[UNK]', '[UNK]', 'ت', '##م',
              '##ہ', '##ا', '##ر', '##ا', 'ہ', '##ے', '!', '[UNK]', 'ت', '##م', '##ہ', '##ا', '##ر', '##ا', 'ہ', '##ے',
              '!', '[UNK]', 'ہ', '##ٹ', '##و', '!', '[UNK]', 'ہ', '##ٹ', '##و', '!', '[UNK]', 'ہ', '##ٹ', '##و', '!',
              'ا', '##ے', 'د', '##ن', '##ی', '##ا', '[UNK]', 'ہ', '##ن', '##د', '##و', '##س', '##ت', '##ان', 'ہ', '##م',
              '##ا', '##ر', '##ا', 'ہ', '##ے', '[UNK]', 'ہ', '##ٹ', '##و', '!', '[SEP]', 'ا', '##ے', 'د', '##ن', '##ی',
              '##ا', '[UNK]', 'ہ', '##ن', '##د', '##و', '##س', '##ت', '##ان', 'ہ', '##م', '##ا', '##ر', '##ا', 'ہ',
              '##ے', 'آ', '##ج', 'ہ', '##م', '##ا', '##ل', '##ی', '##ہ', 'ک', '##ی', 'چ', '##و', '##ٹ', '##ی', 'س',
              '##ے', '[UNK]', 'ہ', '##م', 'ن', '##ے', 'ل', '##ل', '##ک', '##ا', '##ر', '##ا', 'ہ', '##ے', '!', 'آ',
              '##ج', 'ہ', '##م', '##ا', '##ل', '##ی', '##ہ', 'ک', '##ی', 'چ', '##و', '##ٹ', '##ی', 'س', '##ے', '[UNK]',
              'ہ', '##م', 'ن', '##ے', 'ل', '##ل', '##ک', '##ا', '##ر', '##ا', 'ہ', '##ے', '!', '[UNK]', 'ہ', '##ٹ',
              '##و', '!', '[UNK]', 'ہ', '##ٹ', '##و', '!', '[UNK]', 'ہ', '##ٹ', '##و', '!', '[UNK]', 'ہ', '##ٹ', '##و',
              '!', '[SEP]']

    masked_tokens, labels, label_positions = mask_spans(tokens, tokenizer, vocab, seq_len)

    print(label_positions)
    # original_tokens = [tokenizer.cls_token] + tokenizer.tokenize(text) + [tokenizer.sep_token]
    print("\n\n")
    labels = [-100 if index not in label_positions else labels.pop(0) for index, token in enumerate(tokens)]
    print(f"{'Original':<25}{'Corrputed':<25}{'Labels':25}")
    for i in range(len(tokens)):
        print(f"{tokens[i]:<25}{masked_tokens[i]:<25}{labels[i]}")
