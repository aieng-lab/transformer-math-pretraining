from src.data_sets.text_data.DataStats import DataStats

import argparse
import os


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path where all training data is stored", type=check_null)
    parser.add_argument("--dataset_name", help="The name of the specific dataset (Wikpedia_packed_256.hf etc.)",
                        type=check_null)
    parser.add_argument("--multi_processing", help="Whether to use multiprocessing", type=check_null)
    parser.add_argument("--stats_type", help="PROP or general", type=check_null)
    parser.add_argument("--batch_size", help="Batch size for processing", type=cast_int)
    parser.add_argument("--result_path", help="Where to store dataset stats", type=check_null)
    args = parser.parse_args()
    return args


def get_param_value(args, arg_name):
    args = vars(args)
    if args.get(arg_name) is not None:
        return args.get(arg_name)
    else:
        return None


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


if __name__ == "__main__":

    params = get_params()

    data_path = get_param_value(params, "data")
    dataset_name = get_param_value(params, "dataset_name")
    if "wikipedia" in dataset_name.lower():
        data_path = os.path.join(data_path, "wiki")
    if "openwebtext" in dataset_name.lower():
        data_path = os.path.join(data_path, "openwebtext")
    multi_processing = get_param_value(params, "multi_processing")
    stats_type = get_param_value(params, "stats_type")
    if stats_type.lower() not in ("prop", "general"):
        raise AttributeError("stats type must be PROP or General")
    result_path = get_param_value(params, "result_path")
    batch_size = get_param_value(params, "batch_size")
    data_path = os.path.join(data_path, dataset_name)

    data_stats = DataStats(data_path, dataset_name, result_path, multi_processing=multi_processing,
                           batch_size=batch_size)

    data_stats.load_data()
    data_stats.get_data_len()

    if stats_type.upper() == "PROP":
        data_stats.get_prop_info()

    elif stats_type.lower() == "general":
        data_stats.get_general_stats()
