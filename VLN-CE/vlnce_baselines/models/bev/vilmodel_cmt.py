import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype
from transformers import BertPreTrainedModel
from vlnce_baselines.common.ops import create_transformer_encoder
from vlnce_baselines.common.ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad

logger = logging.getLogger(__name__)
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class WpXAttention(nn.Module):

    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        ctx_att_mask = extend_neg_masks(ctx_att_mask)
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return (attention_output, attention_scores)


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
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
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, attention_mask, None if head_mask is None else head_mask[i]
            )
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOutAttention(nn.Module):

    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return (context_layer, attention_scores)


class BertXAttention(nn.Module):

    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return (attention_output, attention_scores)


class GraphLXRTXLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask, graph_sprels=None
    ):
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]
        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]
        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)
        return visn_output

    def forward_lang2visn(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(lang_att_output, lang_attention_mask)[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output


class LanguageEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.num_l_layers)])
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds


class CrossmodalEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList([GraphLXRTXLayer(config) for _ in range(self.num_x_layers)])

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds,
                extended_txt_masks,
                img_embeds,
                extended_img_masks,
                graph_sprels=graph_sprels,
            )
        return img_embeds


class ImageEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        if config.use_depth_embedding:
            self.dep_linear = nn.Linear(config.depth_feat_size, config.hidden_size)
            self.dep_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.dep_linear = self.dep_layer_norm = None
        self.nav_type_embedding = nn.Embedding(2, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self,
        traj_view_img_fts,
        traj_obj_img_fts,
        traj_loc_fts,
        traj_nav_types,
        traj_step_lens,
        traj_vp_view_lens,
        traj_vp_obj_lens,
        type_embed_layer,
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None
        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(
                        torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0)
                    )
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens
        traj_embeds = (
            traj_img_embeds
            + self.loc_layer_norm(self.loc_linear(traj_loc_fts))
            + self.nav_type_embedding(traj_nav_types)
            + type_embed_layer(torch.ones(1, 1).long().to(device))
        )
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)
        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )
        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return (split_traj_embeds, split_traj_vp_lens)


class LocalVPEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size * 2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12),
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1] + 1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)
        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)
        return (vp_embeds, vp_masks)

    def forward(self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds


class LocalBEVEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bev_dim = config.bev_dim
        self.bev_fts_embeddings = nn.Sequential(
            nn.Linear(768, config.hidden_size), BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.bev_pos_embeddings = nn.Sequential(
            nn.Linear(3 + 7, config.hidden_size), BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.nav_type_embedding = nn.Embedding(2, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)

    def bev_input_embedding(self, bev_fts, bev_pos_fts, bev_nav_masks):
        bev_embeds = (
            self.bev_fts_embeddings(bev_fts)
            + self.bev_pos_embeddings(bev_pos_fts)
            + self.nav_type_embedding(bev_nav_masks.long())
        )
        return bev_embeds

    def forward(self, txt_embeds, txt_masks, bev_fts, bev_pos_fts, bev_masks, bev_nav_masks):
        bev_embeds = self.bev_input_embedding(bev_fts, bev_pos_fts, bev_nav_masks)
        bev_embeds = self.encoder(txt_embeds, txt_masks, bev_embeds, bev_masks)
        return bev_embeds


class GlobalMapEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gmap_wp_embeddings = nn.Sequential(
            nn.Linear(config.wp_fts, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12),
        )
        self.gmap_traj_embeddings = create_transformer_encoder(
            config, config.num_wp_layers, norm=True
        )
        self.wp_cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.gmap_wp_step_embeddings = nn.Embedding(config.max_wp_steps + 1, config.hidden_size)
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12),
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device
        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = ({}, {})
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = (
                    torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                )
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])
            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts],
            dim=1,
        )
        return batch_gmap_img_fts

    def gmap_input_embedding(
        self,
        split_traj_embeds,
        split_traj_vp_lens,
        traj_vpids,
        traj_cand_vpids,
        gmap_vpids,
        gmap_step_ids,
        gmap_pos_fts,
        gmap_lens,
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = (
            gmap_img_fts
            + self.gmap_step_embeddings(gmap_step_ids)
            + self.gmap_pos_embeddings(gmap_pos_fts)
        )
        gmap_masks = gen_seq_masks(gmap_lens)
        return (gmap_embeds, gmap_masks)

    def forward(
        self,
        txt_embeds,
        txt_masks,
        split_traj_embeds,
        split_traj_vp_lens,
        traj_vpids,
        traj_cand_vpids,
        gmap_vpids,
        gmap_step_ids,
        gmap_pos_fts,
        gmap_lens,
        graph_sprels=None,
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds,
            split_traj_vp_lens,
            traj_vpids,
            traj_cand_vpids,
            gmap_vpids,
            gmap_step_ids,
            gmap_pos_fts,
            gmap_lens,
        )
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None
        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks, graph_sprels=graph_sprels
        )
        return gmap_embeds


class ClsPrediction(nn.Module):

    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


class GlocalTextPathNavCMT(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bev_dim = 11
        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)
        self.img_embeddings = ImageEmbeddings(config)
        self.local_encoder = LocalBEVEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)
        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.sap_fuse_linear = ClsPrediction(
            self.config.hidden_size, input_size=self.config.hidden_size * 2
        )
        self.use_causal = False
        self.local_ctx_attention = WpXAttention(self.config)
        self.init_weights()
        if config.fix_lang_embedding:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False

    @staticmethod
    def _build_causal_mask(L: int, device) -> torch.Tensor:
        m = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        m[0, :] = 0
        m = m.masked_fill(m == 1, float("-inf")).masked_fill(m == 0, 0.0)
        return m

    def forward_txt(self, txt_ids, txt_masks):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds

    def forward_panorama(self, rgb_fts, dep_fts, loc_fts, nav_types, view_lens):
        device = rgb_fts.device
        rgb_embeds = self.img_embeddings.img_layer_norm(self.img_embeddings.img_linear(rgb_fts))
        if self.img_embeddings.dep_linear is not None:
            dep_embeds = self.img_embeddings.dep_layer_norm(self.img_embeddings.dep_linear(dep_fts))
            img_embeds = rgb_embeds + dep_embeds
        else:
            img_embeds = rgb_embeds
        pano_embeds = (
            img_embeds
            + self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts))
            + self.img_embeddings.nav_type_embedding(nav_types)
            + self.embeddings.token_type_embeddings(torch.ones(1, 1).long().to(device))
        )
        pano_embeds = self.img_embeddings.layer_norm(pano_embeds)
        pano_embeds = self.img_embeddings.dropout(pano_embeds)
        pano_lens = view_lens
        pano_masks = gen_seq_masks(pano_lens)
        if self.img_embeddings.pano_encoder is not None:
            pano_embeds = self.img_embeddings.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
        return (pano_embeds, pano_masks)

    def index_cand_embeds(self, nav_masks, embeds, cand_idxs):
        bs, max_len = (cand_idxs.size(0), cand_idxs.size(1))
        batch_idxs = torch.arange(bs)[:, None].repeat(1, max_len).to(cand_idxs.device)
        cand_embeds = embeds[batch_idxs, cand_idxs]
        cand_masks = nav_masks[batch_idxs, cand_idxs]
        return (cand_embeds, cand_masks)

    def traj_cand_embeds(self, embeds, traj_idxs):
        bs, C_max, T_max = (traj_idxs.size(0), traj_idxs.size(1), traj_idxs.size(2))
        traj_idxs = traj_idxs.reshape(bs, C_max * T_max)
        batch_idxs = torch.arange(bs)[:, None].repeat(1, C_max * T_max).to(traj_idxs.device)
        traj_embeds = embeds[batch_idxs, traj_idxs]
        traj_embeds = traj_embeds.reshape(bs, C_max, T_max, -1)
        return traj_embeds

    def forward_navigation(
        self,
        txt_embeds,
        txt_masks,
        gmap_vpids,
        gmap_step_ids,
        gmap_img_fts,
        gmap_pos_fts,
        gmap_masks,
        gmap_visited_masks,
        gmap_pair_dists,
        bev_fts,
        bev_pos_fts,
        bev_masks,
        bev_nav_masks,
        bev_cand_idxs,
        bev_cand_vpids,
        gmap_front_vp_fts,
        gmap_traj_fts,
        gmap_traj_mask,
        bev_cand_trajs_ids,
        bev_traj_mask,
    ):
        batch_size = txt_embeds.size(0)
        B, V_max, W_max, _ = gmap_traj_fts.shape
        gmap_traj_fts = gmap_traj_fts.flatten(start_dim=0, end_dim=1)
        gmap_traj_mask = gmap_traj_mask.flatten(start_dim=0, end_dim=1)
        wp_batch_size = B * V_max
        wp_traj_embeddings = self.global_encoder.gmap_wp_embeddings(gmap_traj_fts)
        wp_cls = self.global_encoder.wp_cls_token.expand(wp_batch_size, 1, -1)
        gmap_wp_fts = torch.cat([wp_cls, wp_traj_embeddings], dim=1)
        pos_ids = (
            torch.cat(
                [
                    torch.zeros(1, device=gmap_wp_fts.device, dtype=torch.long),
                    torch.arange(
                        gmap_wp_fts.shape[1] - 1, device=gmap_wp_fts.device, dtype=torch.long
                    ),
                ]
            )
            .unsqueeze(0)
            .expand(wp_batch_size, -1)
        )
        gmap_wp_fts = gmap_wp_fts + self.global_encoder.gmap_wp_step_embeddings(pos_ids)
        gmap_traj_mask = torch.cat(
            [
                torch.ones(wp_batch_size, 1, dtype=torch.bool, device=gmap_wp_fts.device),
                gmap_traj_mask,
            ],
            dim=1,
        )
        attn_mask = (
            self._build_causal_mask(gmap_wp_fts.shape[1], gmap_wp_fts.device)
            if self.use_causal
            else None
        )
        if gmap_traj_mask.shape == gmap_wp_fts.shape[:2]:
            gmap_traj_fts = self.global_encoder.gmap_traj_embeddings(
                gmap_wp_fts, mask=attn_mask, src_key_padding_mask=gmap_traj_mask.logical_not()
            )[:, 0, :]
            gmap_traj_fts = gmap_traj_fts.reshape(B, V_max, -1)
            cls_pad = torch.zeros(
                B, 1, gmap_traj_fts.shape[2], device=gmap_traj_fts.device, dtype=gmap_traj_fts.dtype
            )
            gmap_traj_fts = torch.cat([cls_pad, gmap_traj_fts], dim=1)
        else:
            gmap_traj_fts = None
        try:
            if gmap_traj_fts is not None:
                gmap_embeds = (
                    gmap_img_fts
                    + self.global_encoder.gmap_step_embeddings(gmap_step_ids)
                    + self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)
                    + gmap_traj_fts
                )
            else:
                gmap_embeds = (
                    gmap_img_fts
                    + self.global_encoder.gmap_step_embeddings(gmap_step_ids)
                    + self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)
                )
        except Exception:
            gmap_embeds = (
                gmap_img_fts
                + self.global_encoder.gmap_step_embeddings(gmap_step_ids)
                + self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)
            )
        if self.global_encoder.sprel_linear is not None:
            graph_sprels = (
                self.global_encoder.sprel_linear(gmap_pair_dists.unsqueeze(3))
                .squeeze(3)
                .unsqueeze(1)
            )
        else:
            graph_sprels = None
        gmap_embeds = self.global_encoder.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks, graph_sprels=graph_sprels
        )
        bev_embeds = self.local_encoder(
            txt_embeds, txt_masks, bev_fts, bev_pos_fts, bev_masks, bev_nav_masks
        )
        CENTER = (self.bev_dim * self.bev_dim - 1) // 2
        fuse_weights = torch.sigmoid(
            self.sap_fuse_linear(torch.cat([gmap_embeds[:, 0], bev_embeds[:, CENTER]], 1))
        )
        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float("inf"))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float("inf"))
        cand_embeds, cand_masks = self.index_cand_embeds(bev_nav_masks, bev_embeds, bev_cand_idxs)
        bs, C_max, T_max = bev_cand_trajs_ids.shape
        bev_traj_mask = bev_traj_mask.flatten(start_dim=0, end_dim=1)
        cand_embeds = cand_embeds.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        cand_traj_embeds = self.traj_cand_embeds(bev_embeds, bev_cand_trajs_ids)
        cand_traj_embeds = cand_traj_embeds.flatten(start_dim=0, end_dim=1)
        cand_embeds, traj_attn_scores = self.local_ctx_attention(
            cand_embeds, cand_traj_embeds, bev_traj_mask
        )
        cand_embeds = cand_embeds.reshape(bs, C_max, -1)
        local_logits = self.local_sap_head(cand_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(cand_masks.logical_not(), -float("inf"))
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]
        for i in range(batch_size):
            visited_nodes = set(
                [vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask]
            )
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(bev_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits
        outs = {
            "gmap_embeds": gmap_embeds,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "fused_logits": fused_logits,
        }
        return outs

    def forward(self, mode, batch, **kwargs):
        if mode == "language":
            txt_embeds = self.forward_text(batch["txt_ids"], batch["txt_masks"])
            return txt_embeds
        elif mode == "panorama":
            pano_embeds, pano_masks = self.forward_panorama_per_step(
                batch["view_img_fts"],
                batch["obj_img_fts"],
                batch["loc_fts"],
                batch["nav_types"],
                batch["view_lens"],
                batch["obj_lens"],
            )
            return (pano_embeds, pano_masks)
        elif mode == "navigation":
            return self.forward_navigation_per_step(
                batch["txt_embeds"],
                batch["txt_masks"],
                batch["gmap_img_embeds"],
                batch["gmap_step_ids"],
                batch["gmap_pos_fts"],
                batch["gmap_masks"],
                batch["gmap_pair_dists"],
                batch["gmap_visited_masks"],
                batch["gmap_vpids"],
                batch["vp_img_embeds"],
                batch["vp_pos_fts"],
                batch["vp_masks"],
                batch["vp_nav_masks"],
                batch["vp_obj_masks"],
                batch["vp_cand_vpids"],
                batch["bev_fts"],
                batch["bev_pos_fts"],
                batch["bev_masks"],
                batch["bev_nav_masks"],
                batch["bev_cand_idxs"],
                batch["bev_cand_vpids"],
            )
