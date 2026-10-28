from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from .pytorch_transformer.modeling_bert import (
    BertEmbeddings,
    BertSelfAttention,
    BertAttention,
    BertEncoder,
    BertLayer,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertLayerNorm,
    BertPreTrainedModel,
    BertPredictionHeadTransform,
)

logger = logging.getLogger(__name__)


class VisPosEmbeddings(nn.Module):

    def __init__(self, config):
        super(VisPosEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(24, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_vis_feats, position_ids=None):
        seq_length = input_vis_feats.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_vis_feats.device)
            position_ids = position_ids.unsqueeze(0).repeat(input_vis_feats.size(0), 1)
        vis_embeddings = input_vis_feats
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = vis_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class CaptionBertSelfAttention(BertSelfAttention):

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_scores)
        return outputs


class CaptionBertAttention(BertAttention):

    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.config = config

    def forward(self, input_tensor, attention_mask, head_mask=None, history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class CaptionBertLayer(BertLayer):

    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class CaptionBertEncoder(BertEncoder):

    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [CaptionBertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.config = config

    def forward(self, hidden_states, attention_mask, head_mask=None, encoder_history_states=None):
        for i, layer_module in enumerate(self.layer):
            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], history_state)
            hidden_states = layer_outputs[0]
            if i == self.config.num_hidden_layers - 1:
                slang_attention_score = layer_outputs[1]
        outputs = (hidden_states, slang_attention_score)
        return outputs


class BertImgModel(nn.Module):

    def __init__(self, config):
        super(BertImgModel, self).__init__()
        self.config = config
        self.encoder = CaptionBertEncoder(config)

    def forward(self, input_x, attention_mask=None):
        extended_attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(input_x, extended_attention_mask, head_mask=head_mask)
        outputs = (encoder_outputs[0],) + encoder_outputs[1:]
        return outputs


class WaypointBert(nn.Module):

    def __init__(self, config=None):
        super(WaypointBert, self).__init__()
        self.config = config
        self.bert = BertImgModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_x, attention_mask=None):
        outputs = self.bert(input_x, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        return sequence_output
