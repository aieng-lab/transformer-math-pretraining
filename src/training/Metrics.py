from typing import List

from src.config.config import Config
from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.mlm_like.MLM.training import get_correct_predictions as get_correct_mlm_predictions, get_correct_predictions_MATH, get_correct_predictions_MATH_TEXT
from src.pretraining_methods.nsp_like.NSP.training import get_correct_predictions as get_correct_nsp_predictions, get_correct_ffir_predictions, get_correct_nfir_predictions
from src.pretraining_methods.nsp_like.IR.training import get_correct_predictions as get_correct_ir_predictions
from src.pretraining_methods.nsp_like.SOP.training import get_correct_predictions as get_correct_sop_predictions
from src.pretraining_methods.nsp_like.SDT.training import get_correct_predictions as get_correct_sdt_predictions
from src.pretraining_methods.nsp_like.SRT.training import get_correct_predictions as get_correct_srt_predictions
from src.pretraining_methods.mlm_like.SMO.training import get_correct_predictions as get_correct_smo_predictions
from src.pretraining_methods.mlm_like.MAC.training import get_correct_predictions as get_correct_mac_predictions
from src.pretraining_methods.mlm_like.WSO.training import get_correct_predictions as get_correct_wso_predictions
from src.pretraining_methods.specialized.PROP.training import get_correct_predictions as get_correct_prop_predictions


class Metrics:

    def __init__(self, config: Config, objectives: List['str']):
        self.config = config
        self.objectives = objectives

    def get_acc_calculator(self, objective_name):
        if objective_name == Objectives.MLM.name:
            return get_correct_mlm_predictions
        elif objective_name == Objectives.MFM.name:
            return get_correct_predictions_MATH
        elif objective_name == Objectives.MTM.name:
            return get_correct_predictions_MATH_TEXT
        elif objective_name == Objectives.NSP.name:
            return get_correct_nsp_predictions
        elif objective_name == Objectives.NFIR.name:
            return get_correct_nfir_predictions
        elif objective_name == Objectives.FFIR.name:
            return get_correct_ffir_predictions
        elif objective_name == Objectives.IR.name:
            return get_correct_ir_predictions
        elif objective_name == Objectives.SOP.name:
            return get_correct_sop_predictions
        elif objective_name == Objectives.SDT.name:
            return get_correct_sdt_predictions
        elif objective_name == Objectives.SRT.name:
            return get_correct_srt_predictions
        elif objective_name == Objectives.SMO.name:
            return get_correct_smo_predictions
        elif objective_name == Objectives.MAC.name:
            return get_correct_mac_predictions
        elif objective_name == Objectives.WSO.name:
            return get_correct_wso_predictions
        elif objective_name == Objectives.PROP.name:
            return get_correct_prop_predictions

    def get_accuracy_values(self, model_outputs, targets):
        nominators = {}
        denominators = {}
        keys = targets.keys()
        for key in keys:
            acc_calculator = self.get_acc_calculator(key)
            acc_nominator, acc_denominator = acc_calculator(model_outputs, targets)
            nominators[key] = acc_nominator
            denominators[key] = acc_denominator
        return nominators, denominators

    def cumulative_accuracy(self, acc_nominators, acc_denominators, current_nominators, current_denominators):
        for key in current_nominators.keys():
            if key in acc_nominators:
                acc_nominators[key] += current_nominators[key]
                acc_denominators[key] += current_denominators[key]
            else:
                acc_nominators[key] = current_nominators[key]
                acc_denominators[key] = current_denominators[key]
        return acc_nominators, acc_denominators

    def calc_accuracies(self, acc_nominators, acc_denominators):
        keys = acc_nominators.keys()
        accuracies = {key: 0 for key in keys}
        for key in keys:
            nominator = acc_nominators.get(key)
            denominator = acc_denominators.get(key)
            if denominator != 0:
                accuracies[key] = nominator / denominator
        return accuracies

    def calc_combined_accuracies(self, accuracies):
        combined_acc = sum([value for key, value in accuracies.items()]) / len(
            list(accuracies.keys()))
        return combined_acc
