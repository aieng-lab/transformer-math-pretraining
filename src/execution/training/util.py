import json
from typing import Mapping, Any, Union, Tuple, Optional

import torch
from torch import nn
from transformers import AutoModel, DebertaV2Tokenizer, DebertaV2PreTrainedModel, DebertaV2Model, AutoConfig, \
    AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Embeddings, DebertaV2Encoder
import pathlib


# Copied from transformers.models.deberta.modeling_deberta.DebertaModel with Deberta->DebertaV2
class EMDDebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        abs_positions=None
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict
        )
        encoded_layers = list(encoder_outputs[1])

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]#todo why -2??
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1] + abs_positions
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )
class EMDDebertaWithPoolingLayer(nn.Module):
    def __init__(self, pretrained_model_name, z_steps=2):
        super(EMDDebertaWithPoolingLayer, self).__init__()

        # Load the Deberta model and tokenizer
        self.deberta = EMDDebertaV2Model.from_pretrained(pretrained_model_name)
        self.deberta.z_steps = z_steps
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained_model_name)

        # Add a pooling layer (Linear + tanh activation) for the CLS token
        self.pooling_layer = nn.Sequential(
            nn.Linear(self.deberta.config.hidden_size, self.deberta.config.hidden_size),
            nn.Tanh()
        )

        self.position_embeddings = None
        if z_steps > 0:
            self.position_embeddings = nn.Embedding(self.deberta.config.max_position_embeddings, self.deberta.config.hidden_size)

        try:
            state_dict = torch.load(pretrained_model_name + '/pooling.bin')
            self.pooling_layer[0].load_state_dict(state_dict)
        except FileNotFoundError:
            print("Initialize new DeBERTa Pooling Layer with random values")

        try:
            state_dict = torch.load(pretrained_model_name + '/positions.bin')
            self.position_embeddings.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Initialize new DeBERTa Absolute Position Embeddings with random values")

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # Forward pass through the Deberta model
        if self.position_embeddings is None:
            ids = None
        else:
            ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0).expand(input_ids.shape[0], -1).to(input_ids.device)
        abs_positions = self.position_embeddings(ids)
        outputs = self.deberta(input_ids, attention_mask=attention_mask, abs_positions=abs_positions, *args, **kwargs)

        # Extract the hidden states from the output
        hidden_states = outputs.last_hidden_state

        # Get the CLS token representation (first token)
        cls_token = hidden_states[:, 0, :]

        # Apply the pooling layer to the CLS token representation
        pooled_output = self.pooling_layer(cls_token)
        # Include the pooled_output in the output dictionary as 'pooling_layer'
        outputs["pooler_output"] = pooled_output

        # apply enhanced mask decoding
      #  seq_length = hidden_states.size(1)
       # position_ids = torch.arange(0, seq_length, dtype=torch.long, device=hidden_states.device)
        #position_embeddings = self.position_embeddings(position_ids.long())
        #outputs['position_embeddings'] = position_embeddings
        #outputs['layers'] = outputs.hidden_states

        return outputs

    def save_pretrained(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save the model's state_dict, configuration, and tokenizer
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        self.deberta.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        deberta_state_dict = {k.removeprefix('deberta.'): v for k, v in state_dict.items() if k.startswith('deberta')}
        pooler_state_dict = {k.removeprefix('pooling_layer.0.'): v for k, v in state_dict.items() if k.startswith('pooling')}
        position_state_dict = {k.removeprefix('position_embeddings.'): v for k, v in state_dict.items() if k.startswith('position')}
        self.deberta.load_state_dict(deberta_state_dict, strict=strict)
        self.pooling_layer[0].load_state_dict(pooler_state_dict)
        self.position_embeddings.load_state_dict(position_state_dict)

    @classmethod
    def from_pretrained(cls, name):
        # Initialize the instance
        instance = cls(name)

        try:
            # Load the model's state_dict
            instance.load_state_dict(torch.load(f"{name}/pytorch_model.bin"))

            # Load the configuration and tokenizer
            instance.deberta.config = AutoConfig.from_pretrained(name)
            instance.tokenizer = AutoTokenizer.from_pretrained(name)
        except FileNotFoundError:
            print("Could not find DeBERTa pooling layer. Initialize new values")

        return instance

class DebertaWithPoolingLayer(nn.Module):
    def __init__(self, pretrained_model_name, load_pooling=True):
        super(DebertaWithPoolingLayer, self).__init__()

        # Load the Deberta model and tokenizer
        self.deberta = DebertaV2Model.from_pretrained(pretrained_model_name)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained_model_name)

        # Add a pooling layer (Linear + tanh activation) for the CLS token
        self.pooling_layer = nn.Sequential(
            nn.Linear(self.deberta.config.hidden_size, self.deberta.config.hidden_size),
            nn.Tanh()
        )

        self.position_embeddings = None

        if load_pooling:
            try:
                state_dict = torch.load(pretrained_model_name + '/pooling.bin')
                self.pooling_layer[0].load_state_dict(state_dict)
            except FileNotFoundError:
                print("Initialize new DeBERTa Pooling Layer with random values")

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        # Forward pass through the Deberta model
        outputs = self.deberta(input_ids, attention_mask=attention_mask, *args, **kwargs)

        # Extract the hidden states from the output
        hidden_states = outputs.last_hidden_state

        # Get the CLS token representation (first token)
        cls_token = hidden_states[:, 0, :]

        # Apply the pooling layer to the CLS token representation
        pooled_output = self.pooling_layer(cls_token)
        # Include the pooled_output in the output dictionary as 'pooling_layer'
        outputs["pooler_output"] = pooled_output

        # apply enhanced mask decoding
      #  seq_length = hidden_states.size(1)
       # position_ids = torch.arange(0, seq_length, dtype=torch.long, device=hidden_states.device)
        #position_embeddings = self.position_embeddings(position_ids.long())
        #outputs['position_embeddings'] = position_embeddings
        #outputs['layers'] = outputs.hidden_states

        return outputs

    def save_pretrained(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save the model's state_dict, configuration, and tokenizer
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        self.deberta.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

      #  if include_absolute_positions:
       #     torch.save(self.position_embeddings.state_dict(), path + '/positions.bin')

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        deberta_state_dict = {k.removeprefix('deberta.'): v for k, v in state_dict.items() if k.startswith('deberta')}
        pooler_state_dict = {k.removeprefix('pooling_layer.0.'): v for k, v in state_dict.items() if k.startswith('pooling')}

        self.deberta.load_state_dict(deberta_state_dict, strict=strict)
        self.pooling_layer[0].load_state_dict(pooler_state_dict)
     #   self.position_embeddings.load_state_dict(position_state_dict)

    @classmethod
    def from_pretrained(cls, name):
        # Initialize the instance
        instance = cls(name, load_pooling=False)

        try:
            # Load the model's state_dict
            instance.load_state_dict(torch.load(f"{name}/pytorch_model.bin"))

            # Load the configuration and tokenizer
            instance.deberta.config = AutoConfig.from_pretrained(name)
            instance.tokenizer = AutoTokenizer.from_pretrained(name)
        except FileNotFoundError:
            print("Could not find DeBERTa pooling layer. Initialize new values")

        return instance

def create_model(model_identifier):
    if 'deberta' in model_identifier:
        try:
            return DebertaWithPoolingLayer(model_identifier)
        except Exception:
            pass

    if model_identifier.endswith('.bin'):
        state_dict = torch.load(model_identifier)

        state_dict = {k.removeprefix('bert.'): v for k, v in state_dict.items() if not k.startswith('objectives')}
        model = DebertaWithPoolingLayer('microsoft/deberta-v3-base')
        model.load_state_dict(state_dict)
        return model

    try:
        config = json.load(open(model_identifier + '/config.json', 'r+'))
        architectures = config.get('architectures', [])
        if any('deberta' in s.lower() for s in architectures):
            return DebertaWithPoolingLayer(model_identifier)
    except FileNotFoundError:
        pass

    model = AutoModel.from_pretrained(model_identifier)
    return model

if __name__ == '__main__':
    bert_ir = DebertaWithPoolingLayer.from_pretrained('microsoft/deberta-v3-base')
    new_values = nn.Parameter(torch.Tensor([42.0] * 768))
    bert_ir.pooling_layer[0].bias = new_values

    bert_ir.save_pretrained('test')

    bert_ir2 = DebertaWithPoolingLayer.from_pretrained('test')
    assert all(bert_ir2.pooling_layer[0].bias == new_values)
