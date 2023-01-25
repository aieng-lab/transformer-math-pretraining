from src.pretraining_methods.MLM.mlm_model_layers import MLMLayers
from src.pretraining_methods.NSP.nsp_model_layers import NSPLayers
from src.pretraining_methods.Objectives import Objectives

from torch import nn
from transformers import BertModel, BertConfig
import logging

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class PretrainingModel(nn.Module):

    def __init__(self, objectives: list['Objectives'], vocab_size):
        super().__init__()
        self.bert_config = BertConfig(vocab_size=vocab_size)
        self.bert = BertModel(self.bert_config)
        self.objectives = self.get_pretraining_objectives(objectives)

    def forward(self, input_ids, attention_mask, segment_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=segment_ids)
        bert_final = bert_output.get("last_hidden_state")
        bert_pooled = bert_output.get("pooler_output")

        output_dict = {key: None for key, value in self.objectives.items()}

        for name, head in self.objectives.items():
            if name in Objectives.NSP.name:
                output = head(bert_pooled)
                output_dict[name] = output
            else:
                output = head(bert_final)
                output_dict[name] = output

        return output_dict

    def get_pretraining_objectives(self, objective_names: list['Objectives']):
        objectives = {}
        for name in objective_names:
            if name == Objectives.MLM:
                objectives[Objectives.MLM.name] = MLMLayers(self.bert_config.hidden_size, self.bert_config.vocab_size,
                                                            self.bert_config.hidden_act)
            elif name == Objectives.NSP:
                objectives[Objectives.NSP.name] = NSPLayers(self.bert_config.hidden_size)
            else:
                continue
        return objectives
