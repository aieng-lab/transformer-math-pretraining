"""Masked Language Modeling"""

import torch
from torch import nn
from transformers import activations
from transformers import BertForMaskedLM, BertForPreTraining, PreTrainedModel


class MLMLayers(nn.Module):

    def __init__(self, hidden, vocab_size, activation, prefix=None):
        super().__init__()
        if prefix is None:
            prefix = ""

        self.prefix = prefix


        self.set_attr('dense', nn.Linear(hidden, hidden))
        if isinstance(activation, str):
            self.set_attr('activation', activations.get_activation(activation))
        else:
            self.set_attr('activation', activations.get_activation("gelu"))
        self.set_attr('layer_norm', nn.LayerNorm(hidden))
        output = nn.Linear(hidden, vocab_size, bias=False)
        bias = nn.Parameter(torch.zeros(vocab_size))
        output.bias = bias
        self.set_attr('bias', bias)
        self.output = output

    def set_attr(self, name, value):
        setattr(self, self.prefix + '_' + name, value)

    def get_attr(self, name):
        return getattr(self, self.prefix + '_' + name)

    def forward(self, x):
        hidden = self.get_attr('dense')(x)
        hidden = self.get_attr('activation')(hidden)
        hidden = self.get_attr('layer_norm')(hidden)
        lin_output = self.output(hidden)
        return lin_output

# copied from https://github.com/microsoft/DeBERTa/blob/4d7fe0bd4fb3c7d4f4005a7cafabde9800372098/DeBERTa/apps/models/masked_language_model.py
class EnhancedMaskDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_biased_input = getattr(config, 'position_biased_input', True)

    def forward(self, ctx_layers, target_ids, z_states, attention_mask, encoder,
                relative_pos=None):
        mlm_ctx_layers = self.emd_context_layer(ctx_layers, z_states, attention_mask, encoder, target_ids, relative_pos=relative_pos)

        ctx_layer = mlm_ctx_layers[-1]
        return ctx_layer

    def emd_context_layer(self, encoder_layers, z_states, attention_mask, encoder, target_ids, relative_pos=None):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att_mask = extended_attention_mask.byte()
            attention_mask = att_mask * att_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        hidden_states = encoder_layers[-2]
        if not self.position_biased_input:
            layers = [encoder.layer[-1] for _ in range(2)]
            z_states += hidden_states
            query_states = z_states
            query_mask = attention_mask[:,0]
            outputs = []
            rel_embeddings = encoder.get_rel_embedding()

            for layer in layers:
                # TODO: pass relative pos ids
                output = layer(hidden_states, query_mask, return_attentions=False, query_states=query_states,
                               relative_pos=relative_pos, rel_embeddings=rel_embeddings)
                query_states = output
                outputs.append(query_states)
        else:
            outputs = [encoder_layers[-1]]

        _mask_index = (target_ids > 0).view(-1).nonzero().view(-1)

        def flatten_states(q_states):
            q_states = q_states.view((-1, q_states.size(-1)))
            q_states = q_states.index_select(0, _mask_index)
            return q_states

        return [flatten_states(q) for q in outputs]


class DebertaMLMLayers(nn.Module):

    def __init__(self, hidden, vocab_size, activation, deberta):
        super().__init__()

        config = deberta.config
        self.lm_predictions = EnhancedMaskDecoder(config)
        self.deberta = deberta

        self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
        self.position_buckets = getattr(config, 'position_buckets', -1)
        if self.max_relative_positions < 1:
            self.max_relative_positions = config.max_position_embeddings

        self.dense = nn.Linear(hidden, hidden)
        if isinstance(activation, str):
            self.activation = activations.get_activation(activation)
        else:
            self.activation = activations.get_activation("gelu")
        self.layer_norm = nn.LayerNorm(hidden)
        output = nn.Linear(hidden, vocab_size, bias=False)
        bias = nn.Parameter(torch.zeros(vocab_size))
        output.bias = bias
        self.bias = bias
        self.output = output

    def forward(self, hidden_states, target_ids, attention_mask):
        hidden = self.lm_predictions(hidden_states.hidden_states, target_ids, hidden_states['position_embeddings'], attention_mask, self.deberta.encoder, relative_pos=None)  # enhanced mask decoding
        hidden = self.dense(hidden)
        hidden = self.activation(hidden)
        hidden = self.layer_norm(hidden)
        lin_output = self.output(hidden)
        return lin_output

