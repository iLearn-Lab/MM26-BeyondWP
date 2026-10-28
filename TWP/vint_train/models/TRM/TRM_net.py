import torch
import torch.nn as nn
import numpy as np
from vint_train.models.TRM.utils import get_attention_mask
from vint_train.models.TRM.transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def TRM_predict(mode, predictor, rgb_feats, depth_feats):
    vis_logits = predictor(rgb_feats, depth_feats)
    vis_probs = torch.sigmoid(vis_logits)
    if mode == "train":
        return vis_logits
    elif mode == "eval":
        return (vis_probs, vis_logits)


class BinaryDistPredictor_TRM(nn.Module):

    def __init__(self, device):
        super(BinaryDistPredictor_TRM, self).__init__()
        self.device = device
        BertLayerNorm = torch.nn.LayerNorm
        self.num_angles = 120
        self.num_imgs = 12
        self.n_classes = 12
        self.TRM_LAYER = 2
        self.TRM_NEIGHBOR = 1
        self.HEATMAP_OFFSET = 5
        hidden_dim = 768
        self.rgb_feat_dim = 384
        self.visual_fc_rgb = nn.Sequential(nn.Linear(self.rgb_feat_dim, hidden_dim), nn.ReLU(True))
        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(), nn.Linear(np.prod([128, 4, 4]), hidden_dim), nn.ReLU(True)
        )
        self.rgb_depth_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=12, batch_first=True
        )
        config = BertConfig()
        config.model_type = "visual"
        config.finetuning_task = "waypoint_predictor"
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = self.TRM_LAYER
        self.waypoint_TRM = WaypointBert(config=config)
        layer_norm_eps = config.layer_norm_eps
        self.mergefeats_LayerNorm = BertLayerNorm(hidden_dim, eps=layer_norm_eps)
        self.mask = get_attention_mask(num_imgs=self.num_imgs, neighbor=self.TRM_NEIGHBOR).to(
            device
        )
        rgbd_mask = torch.ones(self.num_imgs, self.num_imgs)
        rgbd_mask = rgbd_mask.fill_diagonal_(0.0)
        rgbd_mask = rgbd_mask.masked_fill(rgbd_mask > 0, -10000.0)
        self.register_buffer("rgbd_attn_mask", rgbd_mask)
        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(self.n_classes * (self.num_angles / self.num_imgs))),
        )

    def forward(self, batch):
        rgb_feats = batch["img_fts"]
        depth_feats = batch["dep_fts"]
        bsi = rgb_feats.size(0) // self.num_imgs
        rgb_x = self.visual_fc_rgb(rgb_feats).reshape(bsi, self.num_imgs, -1)
        depth_x = self.visual_fc_depth(depth_feats).reshape(bsi, self.num_imgs, -1)
        attn_out, _ = self.rgb_depth_attn(
            query=rgb_x, key=depth_x, value=depth_x, attn_mask=self.rgbd_attn_mask
        )
        vis_x = self.mergefeats_LayerNorm(attn_out)
        attention_mask = self.mask.repeat(bsi, 1, 1, 1)
        vis_rel_x = self.waypoint_TRM(vis_x, attention_mask=attention_mask)
        vis_logits = self.vis_classifier(vis_rel_x)
        vis_logits = vis_logits.reshape(bsi, self.num_angles, self.n_classes)
        vis_logits = torch.cat(
            (vis_logits[:, self.HEATMAP_OFFSET :, :], vis_logits[:, : self.HEATMAP_OFFSET, :]),
            dim=1,
        )
        return (vis_rel_x, vis_logits)
