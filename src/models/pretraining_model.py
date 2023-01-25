from typing import List

from src.pretraining_methods.mlm_like.MLM.ModelLayers import MLMLayers, DebertaMLMLayers
from src.pretraining_methods.nsp_like.NSP.ModelLayers import NSPLayers
from src.pretraining_methods.nsp_like.SOP.ModelLayers import SOPLayers
from src.pretraining_methods.nsp_like.SDT.ModelLayers import SDTLayers
from src.pretraining_methods.nsp_like.SRT.ModelLayers import SRTLayers
from src.pretraining_methods.mlm_like.SMO.ModelLayers import SMOLayers
from src.pretraining_methods.mlm_like.MAC.ModelLayers import MACLayers
from src.pretraining_methods.mlm_like.WSO.ModelLayers import WSOLayers
from src.pretraining_methods.specialized.PROP.ModelLayers import PROPLayers
from src.pretraining_methods.Objectives import Objectives
from src.pretraining_methods.combinations.MLM_NSP.training import get_losses as get_mlm_nsp_losses
from src.pretraining_methods.combinations.MLM_SOP.training import get_losses as get_mlm_sop_losses
from src.pretraining_methods.combinations.MLM_SDT.training import get_losses as get_mlm_sdt_losses
from src.pretraining_methods.combinations.MLM_SRT.training import get_losses as get_mlm_srt_losses
from src.pretraining_methods.combinations.MAC_NSP.training import get_losses as get_mac_nsp_losses
from src.pretraining_methods.mlm_like.MLM.training import get_losses as get_mlm_losses, get_losses_MLM_MATH, get_losses_MLM_MATH_TEXT
from src.pretraining_methods.combinations.SMO_NSP.training import get_losses as get_smo_nsp_losses
from src.pretraining_methods.combinations.WSO_NSP.training import get_losses as get_wso_nsp_losses
from src.pretraining_methods.combinations.SMO_SDT.training import get_losses as get_smo_sdt_losses
from src.pretraining_methods.combinations.WSO_SDT.training import get_losses as get_wso_sdt_losses
from src.pretraining_methods.combinations.SMO_SRT.training import get_losses as get_smo_srt_losses
from src.pretraining_methods.combinations.WSO_SRT.training import get_losses as get_wso_srt_losses
from src.pretraining_methods.combinations.PROP_MLM.training import get_losses as get_prop_mlm_losses
from src.pretraining_methods.combinations.MAC_SDT.training import get_losses as get_mac_sdt_losses
from src.pretraining_methods.combinations.MAC_SRT.training import get_losses as get_mac_srt_losses
from src.pretraining_methods.nsp_like.NSP.training import get_nsp_losses, get_ffir_losses, get_nfir_losses
from src.pretraining_methods.nsp_like.IR.training import get_ir_losses

import torch
from torch import nn
from transformers import BertModel, BertConfig, AutoConfig, DebertaConfig
import logging

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class PretrainingModel(nn.Module):

    def __init__(self, objectives: List['Objectives'], vocab_size, pretrained_bert=None):
        super().__init__()
        if 'deberta' in str(type(pretrained_bert)).lower():
            self.bert_config = DebertaConfig(vocab_size=vocab_size)
            self.is_deberta = True
        else:
            self.is_deberta = False
            self.bert_config = BertConfig(vocab_size=vocab_size)
        if pretrained_bert is not None:
            self.bert = pretrained_bert
        else:
            self.bert = BertModel(self.bert_config)

        objectives_ = []
        prefixes = []

        for o in objectives:
            if o == Objectives.MFM:
                objectives_.append(Objectives.MLM)
                prefixes.append('mlm_math')
            elif o == Objectives.MTM:
                objectives_.append(Objectives.MLM)
                prefixes.append('mlm_math_text')
            else:
                objectives_.append(o)
                prefixes.append(None)
        
        if self.is_deberta:
            self.objectives = self.get_pretraining_objectives(objectives, objective_prefixes=prefixes, deberta=self.bert)
        else:
            self.objectives = self.get_pretraining_objectives(objectives, objective_prefixes=prefixes)

        for objective in self.objectives.values():
            objective.apply(self.init_weights)
            if objective.__class__ == MLMLayers:
                self.tie_weigths(objective)

    def forward(self, input_ids, attention_mask, segment_ids, labels):

        if Objectives.PROP.name in self.objectives.keys():
            return self.prop_forward(input_ids, attention_mask, segment_ids, labels)

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=segment_ids) # , output_hidden_states=True
        bert_final = bert_output.get("last_hidden_state")
        if 'pooler_output' in bert_output:
            bert_pooled = bert_output.get("pooler_output")
        else:
            # only needed for DeBERTa models wihtout pooler_output (Use DebertaWithPoolingLayer for a deberta model with pooling layer)
            bert_pooled = bert_final[:,0,:]

        objectives = {k: v for k, v in self.objectives.items() if k in labels.keys()}

        output_dict = {key: None for key, value in objectives.items()}
        unused_output_dict = {}

        pooler_used = False
        final_used = False

        for name, head in self.objectives.items():
            if name in (Objectives.NSP.name, Objectives.NFIR.name, Objectives.FFIR.name, Objectives.IR.name, Objectives.SOP.name, Objectives.SDT.name, Objectives.SRT.name):
                output = head(bert_pooled)
                pooler_used = True
            else:
                output = head(bert_final)
                final_used = True

            if name in objectives:
                output_dict[name] = output
            else:
                unused_output_dict[name] = output
        loss = self.get_loss(output_dict, labels)

        # somewhat hacky solution to PyTorch DDP complaining about unused parameters
        # Just add up unused parameters, multiply by zero and add to loss
        if not pooler_used:
            pooler_sum = torch.sum(bert_pooled)
            pooler_zero = pooler_sum * 0.0
            loss = loss + pooler_zero
        if not final_used:
            final_sum = torch.sum(bert_final)
            final_zero = final_sum * 0.0
            loss = loss + final_zero

        for key, output in unused_output_dict.items():
            output_sum = torch.sum(output)
            pooler_zero = output_sum * 0.0
            loss = loss + pooler_zero

        output_dict["loss"] = loss

        return output_dict


    def prop_forward(self, input_ids, attention_mask, segment_ids, labels):
        output_dicts = []

        for i in range(2):
            input_ids_set_i = input_ids[i]
            attention_masks_set_i = attention_mask[i]
            segment_ids_set_i = segment_ids[i]

            bert_output_set_i = self.bert(input_ids=input_ids_set_i, attention_mask=attention_masks_set_i,
                                    token_type_ids=segment_ids_set_i)
            bert_final_set_i = bert_output_set_i.get("last_hidden_state")
            if 'pooler_output' in bert_output_set_i:
                bert_pooled_set_i = bert_output_set_i.get("pooler_output")
            else:
                bert_pooled_set_i = bert_final_set_i[:,0,:]

            output_dict_set_i = {key: None for key, value in self.objectives.items()}

            pooler_used = False
            final_used = False

            for name, head in self.objectives.items():
                if name in (Objectives.NSP.name, Objectives.SOP.name, Objectives.SDT.name, Objectives.SRT.name, Objectives.PROP.name):
                    output_set_i = head(bert_pooled_set_i)
                    pooler_used = True
                else:
                    output_set_i = head(bert_final_set_i)
                    final_used = True
                output_dict_set_i[name] = output_set_i
            output_dicts.append(output_dict_set_i)


        output_dict = {}
        for key in self.objectives.keys():
            values = []
            for item in output_dicts:
                values.append(item.get(key))
            if key != Objectives.PROP.name:
                values = torch.cat((values[0], values[1]), dim=0)
            output_dict[key] = values

        '''for key, value in labels.items():
            if key == Objectives.PROP.name:
                pass
            else:
                new_value = torch.cat((value[0], value[1]), dim=0)
                labels[key] = new_value'''

        loss = self.get_loss(output_dict, labels)

        if not pooler_used:
            pooler_sum = torch.sum(bert_pooled_set_i)
            pooler_zero = pooler_sum * 0.0
            loss = loss + pooler_zero
        if not final_used:
            final_sum = torch.sum(bert_final_set_i)
            final_zero = final_sum * 0.0
            loss = loss + final_zero

        output_dict["loss"] = loss

        return output_dict


    def get_pretraining_objectives(self, objective_names: List['Objectives'], objective_prefixes=None, deberta=None):
        objectives = nn.ModuleDict()
        for i, name in enumerate(objective_names):
            if name in [Objectives.MLM, Objectives.MFM, Objectives.MTM]:
                try:
                    prefix = objective_prefixes[i]
                except Exception:
                    prefix = None

                #if deberta:
                 #   objectives[name.name] = DebertaMLMLayers(self.bert_config.hidden_size, self.bert_config.vocab_size,
                  #                                              self.bert_config.hidden_act, deberta=deberta.deberta)
               # else:
                objectives[name.name] = MLMLayers(self.bert_config.hidden_size, self.bert_config.vocab_size,
                                                                self.bert_config.hidden_act, prefix=prefix)
            elif name == Objectives.NSP:
                objectives[Objectives.NSP.name] = NSPLayers(self.bert_config.hidden_size)
            elif name == Objectives.NFIR:
                objectives[Objectives.NFIR.name] = NSPLayers(self.bert_config.hidden_size)
            elif name == Objectives.FFIR:
                objectives[Objectives.FFIR.name] = NSPLayers(self.bert_config.hidden_size)
            elif name == Objectives.IR:
                objectives[Objectives.IR.name] = NSPLayers(self.bert_config.hidden_size)
            elif name == Objectives.SOP:
                objectives[Objectives.SOP.name] = SOPLayers(self.bert_config.hidden_size)

            elif name == Objectives.SDT:
                objectives[Objectives.SDT.name] = SDTLayers(self.bert_config.hidden_size)

            elif name == Objectives.SRT:
                objectives[Objectives.SRT.name] = SRTLayers(self.bert_config.hidden_size)

            elif name == Objectives.SMO:
                objectives[Objectives.SMO.name] = SMOLayers(self.bert_config.hidden_size, self.bert_config.vocab_size,
                                                            self.bert_config.hidden_act)

            elif name == Objectives.MAC:
                objectives[Objectives.MAC.name] = MACLayers(self.bert_config.hidden_size, self.bert_config.vocab_size,
                                                            self.bert_config.hidden_act)

            elif name == Objectives.WSO:
                objectives[Objectives.WSO.name] = WSOLayers(self.bert_config.hidden_size, self.bert_config.vocab_size,
                                                            self.bert_config.hidden_act)

            elif name == Objectives.PROP:
                objectives[Objectives.PROP.name] = PROPLayers(self.bert_config.hidden_size)

            else:
                continue
        return objectives

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def tie_weigths(self, objective):
        if hasattr(self.bert, 'embeddings'):
            objective.output.weight = self.bert.embeddings.word_embeddings.weight
        else:
            objective.output.weight = self.bert.deberta.embeddings.word_embeddings.weight

    def get_loss(self, output_dict, labels, one_by_one=True):
        keys = list(output_dict.keys())
        if Objectives.MLM.name in keys and Objectives.NSP.name in keys and len(keys) == 2:
            loss = get_mlm_nsp_losses(output_dict, labels)
        elif Objectives.MLM.name in keys and len(keys) == 1:
            loss = get_mlm_losses(output_dict, labels)
        elif Objectives.MFM.name in keys and len(keys) == 1:
            loss = get_mlm_losses(output_dict, labels, obj=Objectives.MFM)
        elif Objectives.MTM.name in keys and len(keys) == 1:
            loss = get_mlm_losses(output_dict, labels, obj=Objectives.MTM)
        elif Objectives.MLM.name in keys and Objectives.SOP.name in keys and len(keys) == 2:
            loss = get_mlm_sop_losses(output_dict, labels)
        elif Objectives.MLM.name in keys and Objectives.SDT.name in keys and len(keys) == 2:
            loss = get_mlm_sdt_losses(output_dict, labels)
        elif Objectives.MLM.name in keys and Objectives.SRT.name in keys and len(keys) == 2:
            loss = get_mlm_srt_losses(output_dict, labels)
        elif Objectives.SMO.name in keys and Objectives.NSP.name in keys and len(keys) == 2:
            loss = get_smo_nsp_losses(output_dict, labels)
        elif Objectives.MAC.name in keys and Objectives.NSP.name in keys and len(keys) == 2:
            loss = get_mac_nsp_losses(output_dict, labels)
        elif Objectives.WSO.name in keys and Objectives.NSP.name in keys and len(keys) == 2:
            loss = get_wso_nsp_losses(output_dict, labels)
        elif Objectives.SMO.name in keys and Objectives.SDT.name in keys and len(keys) == 2:
            loss = get_smo_sdt_losses(output_dict, labels)
        elif Objectives.WSO.name in keys and Objectives.SDT.name in keys and len(keys) == 2:
            loss = get_wso_sdt_losses(output_dict, labels)
        elif Objectives.SMO.name in keys and Objectives.SRT.name in keys and len(keys) == 2:
            loss = get_smo_srt_losses(output_dict, labels)
        elif Objectives.WSO.name in keys and Objectives.SRT.name in keys and len(keys) == 2:
            loss = get_wso_srt_losses(output_dict, labels)
        elif Objectives.PROP.name in keys and Objectives.MLM.name in keys and len(keys) == 2:
            loss = get_prop_mlm_losses(output_dict, labels)
        elif Objectives.MAC.name in keys and Objectives.SDT.name in keys and len(keys) == 2:
            loss = get_mac_sdt_losses(output_dict, labels)
        elif Objectives.MAC.name in keys and Objectives.SRT.name in keys and len(keys) == 2:
            loss = get_mac_srt_losses(output_dict, labels)
        elif Objectives.NSP.name in keys:
            loss = get_nsp_losses(output_dict, labels)
        elif Objectives.NFIR.name in keys:
            loss = get_nfir_losses(output_dict, labels)
        elif Objectives.FFIR.name in keys:
            loss = get_ffir_losses(output_dict, labels)
        elif Objectives.IR.name in keys:
            loss = get_ir_losses(output_dict, labels)
        else:
            loss = get_mixed_loss(output_dict, labels)
        return loss

losses = {
    Objectives.MLM: get_mlm_losses,
    Objectives.MFM: get_losses_MLM_MATH,
    Objectives.MTM: get_losses_MLM_MATH_TEXT,
    Objectives.NSP: get_nsp_losses,
    Objectives.NFIR: get_nfir_losses,
    Objectives.FFIR: get_ffir_losses
}
def get_mixed_loss(output_dict, labels):
    loss = 0
    for key in output_dict:
        output = output_dict[key]
        label = labels[key]
        loss += losses[key](output, label)
    return loss