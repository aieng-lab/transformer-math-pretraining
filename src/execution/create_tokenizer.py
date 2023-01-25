import json
import os
import re

from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForPreTraining

def remove_suffix(string, suffix):
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string

latex_pattern = r"\\([a-zA-Z]+|\\|[|])"
def get_latex_commands(formula: str):
    tex = ["\\" + x for x in re.findall(latex_pattern, formula) if len(x) > 0 and not (len(x) >= 2 and x.startswith('n') and x[1].isupper() or x == 'n')]
    tex += [x for x in re.findall(r"\\begin\{([^}]+)\}", formula) if len(x) > 0]
    return tex

def analyze_latex_tokens(output='data/tokenizer/'):
    counter_map = Counter()
    data = load_dataset('ddrg/math_text', split='train')
    print(data)
    for e in data['text']:
        latex_commands = get_latex_commands(e)
        for command in latex_commands:
            if command in counter_map:
                counter_map[command] += 1
            else:
                counter_map[command] = 1

    print("Finished iterating")
    os.makedirs(output, exist_ok=True)
    json.dump(counter_map, open(output + 'latex_token.json', 'w+', encoding='utf8'), indent=1)

    print("Most common commands: %s" % counter_map.most_common(100))

    for i in [100, 300, 500, 1000, 2000, 5000, 10000]:
        most_common = counter_map.most_common(i)
        most_common = {i: k for i, k in most_common}
        json.dump(most_common, open(output + 'latex_token_%d_dict.json' % i, 'w+'), indent=2)

        lines = list(sorted([str(k) for k in most_common]))
        with open(output + 'latex_token_%d.txt' % i, 'w+') as f:
            for line in lines:
                f.write(line + '\n')

        lines = list(sorted([str(k) for k in most_common], key=lambda k: -counter_map[k]))
        with open(output + 'latex_token_%d_sorted.txt' % i, 'w+') as f:
            for line in lines:
                f.write(line + '\n')



def create_math_tokenizer(input_model, output_model, max_additional_tokens=300, file='data/tokenizer/latex_token_500_sorted.txt'):
    print("Create Tokenizer from <%s>" % input_model)
    old_tokenizer = AutoTokenizer.from_pretrained(input_model)
    get_latex_tokens = lambda tokenizer: [t for t in tokenizer.get_vocab().keys() if t.startswith('\\')]
    latex_tokens = get_latex_tokens(old_tokenizer)
    print("LaTeX Tokens (%d): %s" % (len(latex_tokens), latex_tokens))
    print(old_tokenizer.tokenize(r'(a - b)^2 = a^2 - 2a\cdot b + b^2 is not \delta'))

    additional_tokens = []

    with open(file, 'r+', encoding='utf8') as f:
        for line in f.readlines():
            line = remove_suffix(line, '\n')

            if len(line) > 0 and line not in old_tokenizer.vocab:
                additional_tokens.append(line)

                if len(additional_tokens) >= max_additional_tokens:
                    break

    output_file = '/'.join(file.split('/')[:-1]) + '/latex_tokens.txt'
    with open(output_file, 'w+', encoding='utf8') as f:
        for token in additional_tokens:
            f.write(token + '\n')


    print(additional_tokens)
    print("Total number of added tokens: %d" % len(additional_tokens))
    old_tokenizer.add_tokens(additional_tokens)
    latex_tokens = get_latex_tokens(old_tokenizer)
    print("LaTeX Tokens (%d) after file new tokens" % len(latex_tokens))

    tokenizer = old_tokenizer

    print(tokenizer.tokenize(r'$(a - b)^2 = a^2 - 2a\cdot b + b^2$ is not $\delta$'))
    latex_tokens = [t for t in tokenizer.get_vocab().keys() if t.startswith('\\')]
    print("LaTeX Tokens (%d): %s" % (len(latex_tokens), latex_tokens))
    latex_tokens = [t for t in tokenizer.get_vocab().keys() if '\\' in t]
    print(latex_tokens)
    tokenizer.save_pretrained(output_model)

    model = AutoModelForPreTraining.from_pretrained(input_model)
    model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(output_model)
    print("Saved pretrained model with LaTeX Tokens")


def get_unique_tokens(model_id_1, model_id_2):
    # Load tokenizers
    tokenizer1 = AutoTokenizer.from_pretrained(model_id_1)
    tokenizer2 = AutoTokenizer.from_pretrained(model_id_2)

    # Get vocabularies as sets of tokens
    vocab1 = set(tokenizer1.get_vocab().keys())
    vocab2 = set(tokenizer2.get_vocab().keys())

    # Tokens only in model 2
    unique_to_model2 = vocab2 - vocab1

    return unique_to_model2



if __name__ == '__main__':
    base_model = 'bert-base-cased'
    output = f'models/tokenized/{base_model}'
    analyze_latex_tokens()
    create_math_tokenizer(base_model, output)


    # optional: compare math tokens to those from ddrg/math_structure_bert
    token_diff = get_unique_tokens('ddrg/math_structure_bert', output)
    if len(token_diff) > 0:
        print(f"Tokens only in {output}:")
        print(token_diff)
    else:
        print(f"The models {output} and ddrg/math_structure_bert have the same tokens.")

