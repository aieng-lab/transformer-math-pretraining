"""Masked Language Modeling"""
import warnings
from typing import List

from transformers import BertTokenizerFast, AutoModelForMaskedLM, BertTokenizer
import random

from pretraining_methods.Objectives import Objectives


default_mlm_probability = 0.15

def mask_tokens(tokens, tokenizer: BertTokenizerFast, vocabulary: list, seq_len, mlm_probability=default_mlm_probability):
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
    labels = []
    label_positions = []

    while True:
        index = random.randint(0, len(tokens) - 1)
        if index not in visited and tokens[index] not in special_tokens:
            # position is chosen for masking
            already_masked += 1
            masked_percent = already_masked / seq_len
            if masked_percent > mlm_probability:
                # abort when more than 15% of tokens would be masked
                break

            visited.append(index)
            rand_positions.append(index)

    # indices for replacing are determined

    for index, item in enumerate(tokens):

        if index in rand_positions:
            # token was chosen as mask candidate

            prob = random.random()

            if prob < 0.8:
                # replace token with mask, add original token to labels
                labels.append(tokens[index])
                label_positions.append(index)
                tokens[index] = tokenizer.mask_token

            elif prob < 0.9:
                # replace token with random token, add original token to labels
                labels.append(tokens[index])
                label_positions.append(index)
                rand_token = vocabulary[random.randrange(len(vocabulary))]
                tokens[index] = rand_token

            else:
                # keep original token, add original token to labels
                labels.append(tokens[index])
                label_positions.append(index)

        else:
            # token was not chosen as mask candidate, do not add token to labels
            pass

    return tokens, labels, label_positions

math_words = {
            'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Zero', 'factorial',
            'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth',
            'Sum', 'Difference', 'Product', 'Quotient', 'Equation', 'Equality', 'Inequality', 'Union', 'Intersection',
            'Intersect', 'Complement', 'negation', 'Implication', 'implies', 'Equivalence', 'equivalent', 'Plus',
            'Minus', 'times', 'multiply', 'subtract', 'divide', 'add',
            'equal', 'unequal', 'not', 'less', 'than', 'greater', 'or', 'and',
            'Variable', 'Constant', 'Function', 'Graph', 'Line', 'Curve', 'Slope', 'Intersect', 'Linear',
            'Parallel', 'Perpendicular', 'Angle', 'Triangle', 'Sides', 'Side', 'Square', 'Rectangle', 'Circle',
            'Sphere', 'Cone', 'Cylinder', 'Polygon', 'Vertex', 'Edge', 'Area', 'Perimeter', 'Volume', 'Ratio',
            'Proportion', 'Percentage', 'Decimal', 'Binary', 'Fraction', 'Prime', 'Composite',
            'Factor', 'Multiple', 'Divisor', 'Dividend', 'Exponent', 'Base', 'Logarithm', 'Radical', 'Matrix',
            'Determinant', 'Vector', 'Scalar', 'Transformation', 'Congruent', 'Similar', 'Congruence', 'Similarity',
            'Right', 'domain', 'angle',
            'Pythagorean', 'Theorem', 'Trigonometry', 'Calculus', 'Derivative', 'Integral', 'antiderivative', 'Limit', 'Differentiation',
            'Integration', 'Differential', 'Cartesian', 'Coordinates', 'Polar', 'Asymptote', 'Quadratic', 'Cubic',
            'Polynomial', 'Rational', 'Number', 'Irrational', 'Complex', 'Imaginary', 'Real', 'Natural', 'Whole',
            'Absolute', 'Median', 'Mean', 'Range', 'Interval', 'half', 'open', 'closed',
            'Standard', 'Deviation', 'Probability', 'Permutation', 'Combination', 'Statistical', 'Analysis',
            'Correlation', 'Regression', 'Hypothesis', 'Null', 'Population', 'Data', 'Random', 'Expected', 'Standard',
            'Error', 'Confidence', 'Variance', 'Distribution', 'Normal', 'Binomial', 'Poisson', 'Exponential',
            'Joint', 'Conditional', 'Bayes', 'Law', 'large', 'Combinatorics', 'Game', 'Theory', 'Euclidean', 'Geometry',
            'Discrete', 'Mathematics', 'Algebra', 'Set', 'Logic', 'Boolean', 'Proof', 'Axiom', 'Parity', 'Prime',
            'Fermat', 'Fibonacci', 'Sequence', 'Golden', 'Algorithms', 'Graph', 'Tree', 'Network', 'Homomorphism',
            'Isomorphism', 'Metric', 'Space', 'Continuous', 'Differentiable', 'integrable', 'Lipschitz', 'continuity',
            'Banach', 'Hilbert', 'Fourier', 'Laplace', 'transform', 'Partial', 'arithmetic', 'previous', 'next',
            'True', 'False', 'inverse', 'solutions', 'numerator', 'denominator',
            'solution', 'contradiction', 'rank', 'dimension', 'transcendental', 'algebraic'}

math_words = set(s.lower() for s in math_words)

# from transformers.data.data_collator.DataCollatorForWholeWordMask
def _whole_word_mask(input_tokens: List[str], tokenizer, max_predictions=512, mlm_probability=default_mlm_probability):
    """
    Get 0/1 labels for masked tokens with whole word mask proxy
    """
    if not isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
        warnings.warn(
            "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
            "Please refer to the documentation for more information."
        )

    cand_indexes = []
    for i, token in enumerate(input_tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue

        if len(cand_indexes) >= 1 and (token.startswith("##") or not token.startswith('▁')):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    words = [tokenizer.decode(tokenizer.convert_tokens_to_ids([input_tokens[i] for i in idx])) for idx in cand_indexes]
    math_indices = [i for i, word in enumerate(words) if word.lower() in math_words]
    cand_indexes = [cand_indexes[i] for i in math_indices]

    random.shuffle(cand_indexes)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if random.random() > mlm_probability:
            continue

        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_lms.append(index)

    if len(covered_indexes) != len(masked_lms):
        raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
    mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
    return mask_labels

def mask_math_tokens(input, tokenizer: BertTokenizerFast, vocabulary: list, seq_len, math_mlm_probability=0.2, math_words_probability=0.3, mlm_probability=0.15):
    """
    :param tokens:
    :param tokenizer:
    :param vocabulary:
    :param seq_len: Number of tokens that are not special tokens
    :return:
    """
    import torch


    labels = list(input)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full((len(labels),), 0.0)

    #special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels]
    #special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    dollar = '$' #tokenizer('$', add_special_tokens=False)['input_ids'][0]
    begin_token = r'\begin' # tokenizer('\\begin', add_special_tokens=False)['input_ids'][0]
    end_token = r'\end' #tokenizer('\\end', add_special_tokens=False)['input_ids'][0]
    equation_start_token = r'\[' # tokenizer('\\[', add_special_tokens=False)['input_ids'][0]
    equation_end_token = r']' #tokenizer('\\]', add_special_tokens=False)['input_ids'][0]

    dollar_indices = []
    equation_start_indices = []
    equation_end_indices = []
    begin_indices = []
    end_indices = []
    # Find the index positions of the '$' tokens
    begin_depth = 0
    equation_depth = 0
    for i, token in enumerate(input):
        if token == dollar:
            dollar_indices.append(i)
        elif token == equation_start_token:
            if equation_depth == 0:
                equation_start_indices.append(i)
            equation_depth += 1
        elif token == equation_end_token:
            equation_depth -= 1
            if equation_depth == 0:
                equation_end_indices.append(i)
        elif token == begin_token:
            if begin_depth == 0:
                begin_indices.append(i)
            begin_depth += 1
        elif token == end_token:
            begin_depth -= 1
            if begin_depth == 0:
                end_indices.append(i)

    # Initialize a list to store the tokens between '$' tokens
    math_token_indices = []

    # Iterate over the index positions to extract the tokens
    for i, (start, end) in enumerate(zip(dollar_indices, dollar_indices[1:])):
        # Extract the tokens between each pair of '$' tokens
        if i % 2 == 0:
            # tokens_between = input[start + 1:end]
            # math_token_indices.extend(tokens_between)
            math_token_indices.extend(list(range(start + 1, end)))

    for start, end in [(begin_indices, end_indices), (equation_start_indices, equation_end_indices)]:
        for s, e in zip(start, end):
            math_token_indices.extend(list(range(s + 2, e - 1)))
            # tokens_between = input[s+2:e-1]
            # math_token_indices.extend(tokens_between)

    probability_matrix[math_token_indices] = math_mlm_probability
    input_indices = torch.bernoulli(probability_matrix).bool()


    mask_labels = _whole_word_mask(input, tokenizer, mlm_probability=math_words_probability)
    masked_indices = input_indices | torch.tensor(mask_labels)

    # set probability for all left entries that are not set now and are no special tokens
    mask = input_indices == 0
    total_still_to_be_set = torch.sum(mask)
    total_to_be_set = len(input)
    current_sum = torch.sum(input_indices)
    target_sum = total_to_be_set * mlm_probability
    prob = (target_sum - current_sum) / total_still_to_be_set
    if prob > 0:
        probability_matrix = torch.full((len(labels),), 0.0)
        probability_matrix.masked_fill_(mask, value=prob)
    # torch.sum(probability_matrix) / total_to_be_set = self.mlm_probability

        masked_indices |= torch.bernoulli(probability_matrix).bool()

    labels_ = []
    tokens = list(input)
    label_positions = []
    for index, mask in enumerate(masked_indices):
        if mask:
            prob = random.random()

            if prob < 0.8:
                # replace token with mask, add original token to labels
                labels_.append(tokens[index])
                label_positions.append(index)
                tokens[index] = tokenizer.mask_token

            elif prob < 0.9:
                # replace token with random token, add original token to labels
                labels_.append(tokens[index])
                label_positions.append(index)
                rand_token = vocabulary[random.randrange(len(vocabulary))]
                tokens[index] = rand_token
            else:
                # keep original token, add original token to labels
                labels_.append(tokens[index])
                label_positions.append(index)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels_, label_positions

def only_math_text_mlm(*args, **kwargs):

    return only_mlm(math_text=True, obj=Objectives.MTM, *args, **kwargs)

def only_math_mlm(*args, **kwargs):
    return only_mlm(obj=Objectives.MFM, *args, **kwargs)

def only_mlm(text_corpus, index, tokenizer, vocab, max_len, is_last=False, math_text=False, math_mlm_probability=0.2, math_words_probability=0.3, obj=Objectives.MLM):
    sequence = text_corpus[index]["text"]
    sequence_tokens = tokenizer.tokenize(sequence, truncation=True)

    if len(sequence_tokens) > (max_len - 2):
        sequence_tokens = sequence_tokens[:(max_len - 2)]

    seq_len = len(sequence_tokens)

    sequence_tokens = [tokenizer.cls_token] + sequence_tokens + [tokenizer.sep_token]

    if math_text:
        masked_tokens, labels, label_positions = mask_math_tokens(sequence_tokens, tokenizer, vocab, seq_len, math_mlm_probability=math_mlm_probability, math_words_probability=math_words_probability)
    else:
        masked_tokens, labels, label_positions = mask_tokens(sequence_tokens, tokenizer, vocab, seq_len)


    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    mlm_label_ids = tokenizer.convert_tokens_to_ids(labels)
    mlm_labels = [-100 if index not in label_positions else mlm_label_ids.pop(0) for index, id in enumerate(input_ids)]
    mlm_labels = mlm_labels + ([-100] * (max_len - len(mlm_labels)))

    segment_ids = ([0] * (len(sequence_tokens)))

    return {
        "bert_input": input_ids,
        "bert_label": {obj.name: mlm_labels},
        "segment_label": segment_ids,
    }


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    print("start")
    vocab = list(tokenizer.get_vocab().keys())
    sentence_a = """Schölermann was initially surprised when he was told the writers would be making his straight character gay.[2] He said, "I had to think about it because I wasn't sure if I could play a gay character truthfully or convincingly ... I was really afraid I couldn't do it".[2] Schölermann took some inspiration from the 2005 film Brokeback Mountain, explaining "The movie showed how realistically and beautifully you can portray the love between two men".[9] Acknowledging their chemistry, Weil said "We can be really proud of what we created and how much heart and energy and work we put into this story."[2] Schölermann agreed, saying "It's important that we just not read the lines and do the scene. It's important to talk about what you want to do with the storyline and what you want to create."""
    sentence_b = """Both actors noted that though some touching and kissing between Olli and Christian is scripted, based on the scene they often take it upon themselves to add these intimacies, and "little things that you have when you are in a partnership".[10] Elliot Robinson of So So Gay wrote, "This additional layer of tenderness really gives the couple a warmth and realness that has no doubt been instrumental in placing Christian and Olli in the hearts of many fans. Whereas a gay couple's kiss may in other soaps be written into a script as a deliberate plot device, such as prompting a homophobic reaction in a third party, Christian and Olli's continuous and observable closeness only adds to the believability of the couple's relationship."""

    a_tokens = tokenizer.tokenize(sentence_a)
    b_tokens = tokenizer.tokenize(sentence_b)

    seq_len = len(a_tokens) + len(b_tokens)
    tokens = [tokenizer.cls_token] + a_tokens + [tokenizer.sep_token] + b_tokens + [tokenizer.sep_token]

    print(tokens)

    tokens, labels, label_positions = mask_tokens(tokens, tokenizer, vocab, seq_len)
    print(tokens)
    print(labels)
    print(label_positions)

    bert_input = tokenizer.convert_tokens_to_ids(tokens)
    labels = tokenizer.convert_tokens_to_ids(labels)
    print(labels)
    labels = [-100 if index not in label_positions else labels.pop(0) for index, id in enumerate(bert_input)]
    print(labels)
