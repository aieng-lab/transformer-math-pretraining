from src.helpers.general_helpers import path_exists, create_path_if_not_exists
from src.helpers.Timer import Timer

import argparse
import gensim
import gensim.downloader as api
import os
import json


class ParamError(Exception):
    pass


def check_null(input):
    if input == "null":
        return None
    elif input in ("True", "true"):
        return True
    elif input in ("False", "false"):
        return False
    else:
        return input

def cast_int(input):
    input = check_null(input)
    if input is not None:
        input = int(input)
    return input

def check_model_size(input):
    if input not in ("small", "large"):
        raise ParamError("model size must be either 'small' or 'large'")
    else:
        return input



def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gensim_path", help="Path where gensim model is stored", type=check_null)
    parser.add_argument("--dict_path", help="Path under which similar word dictionary should be saved", type=check_null)
    parser.add_argument("--sample_size", help="Number of words to precompute", type=cast_int)
    parser.add_argument("--restrict_vocab", help="How many words to consider for knn search", type=cast_int)
    parser.add_argument("--model_size", help="Use google-news-300 or glove-wiki-gigaword-100 as gensim model", type=check_model_size)
    args = parser.parse_args()
    return args


def get_param_value(args, arg_name):
    args = vars(args)
    if args.get(arg_name) is not None:
        return args.get(arg_name)
    else:
        return None



def get_gensim_w2v_model(path, model_size):
    if model_size == "small":
        model_name = "glove-wiki-gigaword-100"
    else:
        model_name = "word2vec-google-news-300"
    file_path = os.path.join(path, f"{model_name}.bin")
    if path_exists(file_path):
        model = gensim.models.KeyedVectors.load(file_path)
    else:
        api.BASE_DIR = path
        model = api.load(model_name)
        model.save(file_path)
    return model


def get_word_list(w2v_model):
    word_list = w2v_model.index_to_key
    print(f"Total number of words: {len(word_list)}\n", flush=True)
    return word_list


def pre_compute(word_list, w2v_model, restrict_vocab=None):
    print(f"Using restrict_vocab of {restrict_vocab}", flush=True)
    sim_words_dict = {}
    for word in word_list:
        try:
            similar_words = w2v_model.most_similar(word, topn=10, restrict_vocab=restrict_vocab)
            candidates = []
            for sim_word in similar_words:
                sim_word = sim_word[0]
                sim_word = " ".join(sim_word.split("_"))
                candidates.append(sim_word)

        except KeyError:
            candidates = []
        sim_words_dict[word] = candidates
    return sim_words_dict



def save_as_json(dict, dict_path, restrict_vocab, sample_size, model_size):
    create_path_if_not_exists(dict_path)
    file_name = f"similar_words"
    if sample_size is not None:
        file_name += f"_{sample_size}_sample"
    if restrict_vocab is not None:
        file_name += f"_restrict_vocab_{restrict_vocab}"
    file_name += f"_{model_size}_model.json"

    path = os.path.join(dict_path, file_name)
    with open(path, "w") as file:
        json.dump(dict, file)




if __name__ == "__main__":
    params = get_params()

    gensim_path = get_param_value(params, "gensim_path")
    sample_size = get_param_value(params, "sample_size")
    restrict_vocab = get_param_value(params, "restrict_vocab")
    dict_path = get_param_value(params, "dict_path")
    model_size = get_param_value(params, "model_size")

    w2v_model = get_gensim_w2v_model(gensim_path, model_size)

    word_list = get_word_list(w2v_model)

    if sample_size is not None:
        word_list = word_list[:sample_size]
        print(f"Using sample of {sample_size} words", flush=True)

    timer = Timer()
    timer.start()

    sim_words_dict = pre_compute(word_list, w2v_model, restrict_vocab=restrict_vocab)

    timer.stop()

    print("\n", flush=True)
    timer.print_elapsed()
    print("\n")

    print(f"Saving json file")
    save_as_json(sim_words_dict, dict_path, restrict_vocab, sample_size, model_size)


