import random

from src.helpers.general_helpers import flatten

from src.pretraining_methods.Objectives import Objectives

import torch
from torch.utils.data import Dataset
from typing import Callable, List
import os
import json


class PretrainingDataset(Dataset):

    def __init__(self, objectives: List['Objectives'], text_corpus, tokenizer, max_len,
                 processing_func: Callable, batch_size: int, validation_data=False):
        """
        :param text_corpus:
        :param tokenizer:
        :param max_len:
        :param processing_func: Function that returns dict of the form
        {"bert_input": tokens, "bert_label": {"label1: tokens, "label2": tokens, ...}, "segment_label": tokens}
        """
        self.tokenizer = tokenizer
        self.vocab = list(tokenizer.get_vocab().keys())
        self.max_len = max_len
        self.data = text_corpus
        self.proc_func = processing_func
        self.label_pad = 600
        self.epoch_number = 0
        self.objectives = list(objectives)
        self.objective_by_name = {o.name: o for o in self.objectives}

        self.data_len = {}
        for o in self.objectives:
            if isinstance(self.data, dict) and o.name in self.data:
                if isinstance(self.data[o.name], dict):
                    self.data_len[o.name] = len(self.data[o.name][0])
                else:
                    self.data_len[o.name] = len(self.data[o.name])
            elif isinstance(self.data, dict) and len(self.objectives) == 1:
                self.data_len[o.name] = len(list(self.data.values())[0])
            else:
                self.data_len[o.name] = len(self.data)

        self.batch_size = batch_size

        self.indices = {o: self.random_indices(l) for o, l in self.data_len.items()}
        self.validation_data = validation_data

        self.objectives_len = len(self.objectives)
        if self.validation_data:
            self.total_len = sum(self.data_len.values())
        else:
            # hack: use NSP value for length -> if multiple objectives are given, this is in the MLM_MATH, MLM_MATH_TEXT, NSP setting the only Objective with a dict as data, i.e. different data is supplied per epoch, thus we can use the epoch number to select the correct dataset
            if Objectives.NFIR.name in self.data_len:
                self.total_len = self.data_len[Objectives.NFIR.name] * self.objectives_len
            elif Objectives.FFIR in self.data_len:
                self.total_len = self.data_len[Objectives.FFIR.name] * self.objectives_len
            else:
                self.total_len = max(self.data_len.values())

        self.indexer = {}

        self.objective_stats = {o.name: 0 for o in self.objectives}

    def random_indices(self, n):
        integers = list(range(n))
        random.shuffle(integers)
        return integers

    def __len__(self):
        return self.total_len

    def get_current_objective(self, index):
        if self.validation_data:
            ctr = 0
            for objective in self.objectives:
                ctr += self.data_len[objective.name]
                if index < ctr:
                    return objective

            raise ValueError("Could not find a matching objective! %s, %s" % (index, self.data_len))
        elif self.objectives_len == 1:
            return self.objectives[0]
        else:
            objective_index = (index // self.batch_size) % self.objectives_len
            return self.objectives[objective_index]

    def get_current_proc_func(self, objective):
        if isinstance(self.proc_func, dict):
            return self.proc_func[objective.name]
        return self.proc_func

    def get_current_index(self, objective, index):
        if self.validation_data:
            objective_index = self.objectives.index(objective)
            return index - sum(self.data_len[o.name] for o in self.objectives[:max(0, objective_index)])

        if len(self.objectives) == 1:
            return index

        # we have multiple objectives that are trained in a mixed way
        # a batch is trained with the same objective, then the next is used and so on
        # each objective

        if not self.epoch_number in self.indexer:
            self.init_indices(self.epoch_number)

        batch_offset_index = index % self.batch_size  # index within batch
        batch_start_index = index // self.objectives_len  # start index of batch in objective data
        objective_data_index = batch_start_index + batch_offset_index
        return self.indexer[self.epoch_number][objective.name][objective_data_index]

    def init_indices(self, epoch_number):
        self.indexer[epoch_number] = {}
        random.seed(epoch_number)
        for objective, data_len in self.data_len.items():
            starting_index = (epoch_number * data_len) % self.total_len

            indices = list(range(starting_index, min(self.total_len, data_len)))
            random.shuffle(indices)

            for _ in range(len(indices), self.total_len - data_len + 1, data_len):
                sub_indices = list(range(0, data_len))
                random.shuffle(sub_indices)
                indices += sub_indices

            # fill the last indices up
            end_index = self.total_len - len(indices)
            assert 0 <= end_index < data_len
            last_indices = list(range(0, end_index))
            random.shuffle(last_indices)
            indices += last_indices

            assert len(indices) == self.total_len
            self.indexer[epoch_number][objective] = indices


    def __getitem__(self, index):
        if Objectives.PROP in self.objectives:
            return self.get_prop_item(index)

        is_last = False

        objective = self.get_current_objective(index)
        self.objective_stats[objective.name] += 1

        if not self.validation_data and self.objectives_len > 1 and index > 0 and index % 100000 == 0:
            print("Epoch Progress = %f, index = %s, single GPU stats = %s, current objective=%s" % (index / len(self), index, self.objective_stats, objective), flush=True)

        current_data = self.get_current_data(objective)
        proc_func = self.get_current_proc_func(objective)

        data_index = index
        if len(self.objectives) > 1:
            data_index = self.get_current_index(objective, index)

        bert_dict = proc_func(current_data, data_index, self.tokenizer, self.vocab, self.max_len, is_last=is_last)

        bert_input = bert_dict.get("bert_input")[:self.max_len]
        bert_input_pad = bert_input + [self.tokenizer.pad_token_id] * (self.max_len - len(bert_input))

        attention_mask = ([1] * len(bert_input)) + ([0] * (self.max_len - len(bert_input)))

        segment_label = bert_dict.get("segment_label")[:self.max_len]
        segment_label_pad = segment_label + [segment_label[-1]] * (self.max_len - len(segment_label))

        bert_labels = bert_dict.get("bert_label")
        for key, value in bert_labels.items():
            bert_labels[key] = torch.tensor(value)

        return_dict = {
            "bert_input": torch.tensor(bert_input_pad),
            "bert_label": bert_labels,
            "attention_mask": torch.tensor(attention_mask),
            "segment_label": torch.tensor(segment_label_pad),
            "objective": objective.name
        }

        return_dict = flatten(return_dict)

        if return_dict:
            pass
        else:
            print(f"Index {index} returns None!!!!", flush=True)

        return return_dict

    def get_current_data(self, objective):
        if isinstance(self.data, dict):
            if objective.name in self.data:
                data = self.data[objective.name]
            else:
                data = self.data

            if isinstance(data, dict):
                max_ = max(data.keys())
                index = self.epoch_number % (max_ + 1)
                return data[index]

            return data

        return self.data

    def get_prop_item(self, index):
        bert_dict = self.proc_func(self.get_current_data(), index, self.tokenizer, self.vocab, self.max_len, self.epoch_number)

        bert_input = bert_dict.get("bert_input")
        segment_label = bert_dict.get("segment_label")

        bert_inputs = []
        attention_masks = []
        segment_labels = []

        for i in range(2):
            bert_input_i = bert_input[i][:self.max_len]

            bert_input_pad = bert_input_i + [self.tokenizer.pad_token_id] * (self.max_len - len(bert_input_i))
            bert_inputs.append(torch.tensor(bert_input_pad))

            attention_mask = ([1] * len(bert_input_i)) + ([0] * (self.max_len - len(bert_input_i)))
            attention_masks.append(torch.tensor(attention_mask))

            segment_label_i = segment_label[i][:self.max_len]
            segment_label_pad = segment_label_i + [segment_label_i[-1]] * (self.max_len - len(segment_label_i))
            segment_labels.append(torch.tensor(segment_label_pad))

        bert_labels = bert_dict.get("bert_label")
        for key, value in bert_labels.items():
            if isinstance(value, tuple):
                torch_value = []
                for val in value:
                    torch_value.append(torch.tensor(val))
                bert_labels[key] = tuple(torch_value)
            else:
                bert_labels[key] = torch.tensor(value)

        return_dict = {
            "bert_input": tuple(bert_inputs),
            "bert_label": bert_labels,
            "attention_mask": tuple(attention_masks),
            "segment_label": tuple(segment_labels)
        }

        return_dict = flatten(return_dict)

        if return_dict:
            pass
        else:
            print(f"Index {index} returns None!!!!", flush=True)

        return return_dict
