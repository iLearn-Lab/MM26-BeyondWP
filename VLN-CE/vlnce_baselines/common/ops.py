import torch
from .transformer import TransformerEncoder, TransformerEncoderLayer

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    BertLayerNorm = torch.nn.LayerNorm


def create_transformer_encoder(config, num_layers, norm=False):
    enc_layer = TransformerEncoderLayer(
        config.hidden_size,
        config.num_attention_heads,
        dim_feedforward=config.intermediate_size,
        dropout=config.hidden_dropout_prob,
        activation=config.hidden_act,
        normalize_before=True,
    )
    if norm:
        norm_layer = BertLayerNorm(config.hidden_size, eps=1e-12)
    else:
        norm_layer = None
    return TransformerEncoder(enc_layer, num_layers, norm=norm_layer, batch_first=True)


def extend_neg_masks(masks, dtype=None):
    if dtype is None:
        dtype = torch.float
    extended_masks = masks.unsqueeze(1).unsqueeze(2)
    extended_masks = extended_masks.to(dtype=dtype)
    extended_masks = (1.0 - extended_masks) * -10000.0
    return extended_masks


def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)
    batch_size = len(seq_lens)
    device = seq_lens.device
    masks = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    masks = masks < seq_lens.unsqueeze(1)
    return masks


def gen_wp_pad_mask(list_seq_lens):
    B = len(list_seq_lens)
    V_max = max((len(vps) for vps in list_seq_lens))
    lengths_rows = [int(wps.item()) for g in list_seq_lens for wps in g]
    W_max = max(lengths_rows)
    if W_max == 0:
        W_max = 1
    vp_lens = torch.zeros(B, V_max, dtype=torch.long)
    for b in range(B):
        for v, t in enumerate(list_seq_lens[b]):
            vp_lens[b, v] = t
    vp_lens = vp_lens.view(-1)
    masks = gen_seq_masks(vp_lens, W_max)
    masks = masks.reshape(B, V_max, W_max)
    return masks


def pad_wp_tensors(list_list_seqs):
    B = len(list_list_seqs)
    W_max = max((wps.size(0) for g in list_list_seqs for wps in g))
    V_max = max((len(vps) for vps in list_list_seqs))
    W_D = list(list_list_seqs[0][0].size()[1:])
    vp_list = []
    for vps in list_list_seqs:
        vps = pad_tensors_wgrad(vps, max_len=W_max).reshape(len(vps), -1)
        vp_list.append(vps)
    waypoint_trajs = pad_tensors_wgrad(vp_list)
    waypoint_trajs = waypoint_trajs.reshape(B, V_max, W_max, *W_D)
    return waypoint_trajs


def pad_tensors_wgrad(tensors, lens=None, max_len=None):
    if lens is None:
        lens = [t.size(0) for t in tensors]
    if max_len is None:
        max_len = max(lens)
    batch_size = len(tensors)
    hid = list(tensors[0].size()[1:])
    device = tensors[0].device
    dtype = tensors[0].dtype
    output = []
    for i in range(batch_size):
        if lens[i] < max_len:
            tmp = torch.cat(
                [tensors[i], torch.zeros([max_len - lens[i]] + hid, dtype=dtype).to(device)], dim=0
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output
