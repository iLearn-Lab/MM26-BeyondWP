import math
import torch
import torch.nn as nn


class NoMaD(nn.Module):

    def __init__(self, vision_encoder, noise_pred_net, direction_classifier, fusion_layer):
        super(NoMaD, self).__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.direction_classifier = direction_classifier
        self.fusion_layer = fusion_layer

    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder":
            output = self.vision_encoder(kwargs["batch"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=kwargs["global_cond"],
            )
        elif func_name == "direction_classifier":
            output = self.direction_classifier(kwargs["global_cond"])
        elif func_name == "fusion_layer":
            output = self.fusion_layer(
                kwargs["global_cond"],
                kwargs["direction_label"],
                kwargs["angle_label"],
                kwargs["distance_label"],
            )
        else:
            raise NotImplementedError
        return output


class DenseNetwork(nn.Module):

    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 16, 1),
        )

    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output


class Vis_classifier(nn.Module):

    def __init__(self, config):
        super(Vis_classifier, self).__init__()
        self.hid_dim = config.hidden_size
        self.cls_nums = config.num_imgs
        self.dis_nums = config.num_distance
        self.ang_nums = config.num_angle
        self.HEATMAP_OFFSET = config.HEATMAP_OFFSET
        self.vis_classifier = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, int(self.dis_nums * (self.ang_nums / self.cls_nums))),
        )

    def forward(self, x):
        bsi = x.shape[0]
        vis_logits = self.vis_classifier(x)
        vis_logits = vis_logits.reshape(bsi, self.ang_nums, self.dis_nums)
        vis_logits = torch.cat(
            (vis_logits[:, self.HEATMAP_OFFSET :, :], vis_logits[:, : self.HEATMAP_OFFSET, :]),
            dim=1,
        )
        return vis_logits


class Fusion(nn.Module):

    def __init__(self, config):
        super(Fusion, self).__init__()
        self.use_sincos = config.use_sincos
        self.embed_dim = config.embed_dim
        self.dis_embed_dim = config.dis_embed_dim
        self.output_dim = config.hidden_size
        if not self.use_sincos:
            self.angle_embed = nn.Embedding(config.num_angle, self.embed_dim)
            self.distance_embed = nn.Embedding(config.num_distance, self.dis_embed_dim)
        else:
            raise ValueError
        mlp_mult = getattr(config, "fusion_mlp_mult", 4)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(
                self.embed_dim + self.dis_embed_dim + self.output_dim, self.output_dim * mlp_mult
            ),
            nn.ReLU(),
            nn.Linear(self.output_dim * mlp_mult, self.output_dim * mlp_mult),
            nn.ReLU(),
            nn.Linear(self.output_dim * mlp_mult, self.output_dim),
        )

    def forward(self, condition_feat, direction, angle, distance):
        B, _, D = condition_feat.shape
        B2, _ = direction.shape
        assert B == B2
        direction_indices = direction.clamp(min=0)
        angle_indices = angle.clamp(min=0)
        distance_indices = distance.clamp(min=0)
        gathered_feat = torch.gather(
            condition_feat, dim=1, index=direction_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        if not self.use_sincos:
            angle_vec = self.angle_embed(angle_indices)
            distance_vec = self.distance_embed(distance_indices)
        else:
            raise ValueError
        fused_input = torch.cat([gathered_feat, angle_vec, distance_vec], dim=-1)
        x = self.fusion_mlp(fused_input)
        return x
