from trajectory import (
    ACTION_STATS,
    get_action,
    get_action_torch,
    get_delta,
    normalize_data,
    unnormalize_data,
)
from copy import deepcopy
from typing import Dict, Optional
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo.policy import Net
from vlnce_baselines.models.bev.vlnbert_init import get_vlnbert_models
from vlnce_baselines.models.dif_wp.dp_init import get_dif_wp_models
from vint_train.visualizing.visualize_utils import from_numpy, to_numpy
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import InstructionEncoder
from vlnce_baselines.models.encoders.resnet_encoders import (
    VlnResnetDepthEncoder,
    CLIPEncoderB16,
    RGBEncoder,
    RGBResEncoder,
)
from vlnce_baselines.models.policy import ILPolicy
from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele,
    dir_angle_feature_with_ele,
    angle_feature_torch,
    length2mask,
)
import math


def _actions_to_global_traj_torch(actions: torch.Tensor, angle_index: torch.Tensor) -> torch.Tensor:
    if actions.numel() == 0:
        return actions
    dir_step = 10
    ang_step = math.pi / 6
    ldir_max = 5
    angle_index = angle_index.to(device=actions.device)
    angle_index_long = angle_index.to(dtype=torch.long)
    n_d = torch.zeros_like(angle_index_long)
    for _ in range(12):
        cond = angle_index_long - n_d * dir_step > ldir_max
        if not torch.any(cond):
            break
        n_d = n_d + cond.to(n_d.dtype)
    n_d = torch.remainder(n_d, 12).to(dtype=actions.dtype)
    actions_xy = actions[..., :2]
    x = actions_xy[..., 0].clone().contiguous()
    z = actions_xy[..., 1].clone().contiguous()
    r = torch.hypot(x, z)
    local_dir = torch.atan2(z, x)
    global_dir = local_dir + n_d.unsqueeze(1) * ang_step
    dx = -r * torch.sin(global_dir)
    dz = r * torch.cos(global_dir)
    return torch.stack((dx, dz), dim=-1)


def _global_deltas_to_actions_torch(
    global_deltas: torch.Tensor, angle_index: torch.Tensor
) -> torch.Tensor:
    if global_deltas.numel() == 0:
        return global_deltas
    dir_step = 10
    ang_step = math.pi / 6
    ldir_max = 5
    angle_index = angle_index.to(device=global_deltas.device)
    angle_index_long = angle_index.to(dtype=torch.long)
    n_d = torch.zeros_like(angle_index_long)
    for _ in range(12):
        cond = angle_index_long - n_d * dir_step > ldir_max
        if not torch.any(cond):
            break
        n_d = n_d + cond.to(n_d.dtype)
    n_d = torch.remainder(n_d, 12).to(dtype=global_deltas.dtype)
    dx = global_deltas[..., 0].clone().contiguous()
    dz = global_deltas[..., 1].clone().contiguous()
    r = torch.sqrt(dx * dx + dz * dz)
    global_dir = torch.atan2(-dx, dz)
    local_dir = global_dir - n_d.unsqueeze(1) * ang_step
    x = r * torch.cos(local_dir)
    z = r * torch.sin(local_dir)
    return torch.stack((x, z), dim=-1)


def _actions_to_ndeltas_torch(actions: torch.Tensor, action_stats=ACTION_STATS) -> torch.Tensor:
    deltas = get_delta(actions.clone().contiguous())
    ndeltas = normalize_data(deltas, action_stats)
    return ndeltas


def _tsdf_fallback_enabled() -> bool:
    flag = os.environ.get("TSDF_FALLBACK", "1")
    return str(flag).lower() in ("1", "true", "yes", "y")


def apply_tsdf_guidance(
    diffusion_output: torch.Tensor,
    angle_labels: torch.Tensor,
    tsdf_cost_maps: Optional[torch.Tensor],
    tsdf_meta: Optional[Dict[str, torch.Tensor]],
    pad_mask: Optional[torch.Tensor],
    env_valid_mask: Optional[torch.Tensor],
    guide_cfg: Dict[str, object],
) -> Optional[torch.Tensor]:
    if tsdf_cost_maps is None or tsdf_meta is None:
        return diffusion_output
    device = diffusion_output.device
    bs, dirs = angle_labels.shape
    bd, _, _ = diffusion_output.shape
    if bd != bs * dirs:
        return diffusion_output
    if pad_mask is None:
        pad_mask = torch.zeros((bs, dirs), dtype=torch.bool, device=device)
    elif pad_mask.dtype is not torch.bool:
        pad_mask = pad_mask.to(device=device) < 0.5
    valid_mask = pad_mask.reshape(-1).to(device=device, dtype=diffusion_output.dtype)
    if env_valid_mask is not None:
        env_mask = env_valid_mask.to(device=device)
        if env_mask.dim() == 1:
            env_mask = env_mask[:, None].expand(bs, dirs)
        valid_mask = valid_mask * env_mask.reshape(-1).to(dtype=diffusion_output.dtype)
    if torch.sum(valid_mask) <= 0:
        return diffusion_output
    actions = get_action_torch(diffusion_output, ACTION_STATS)
    if not torch.isfinite(actions).all():
        return diffusion_output
    pos_scale = float(guide_cfg.get("pos_scale", 0.25))
    if pos_scale != 1.0:
        actions = actions * pos_scale
    angle_flat = angle_labels.reshape(-1).to(device=device)
    rotated = _actions_to_global_traj_torch(actions, angle_flat)
    rotated = rotated.detach().requires_grad_(True)
    env_ids = torch.arange(bs, device=device).repeat_interleave(dirs)
    start_x = (
        tsdf_meta["start_x"].to(device=device, dtype=diffusion_output.dtype)[env_ids].unsqueeze(1)
    )
    start_z = (
        tsdf_meta["start_z"].to(device=device, dtype=diffusion_output.dtype)[env_ids].unsqueeze(1)
    )
    res = (
        tsdf_meta["resolution"]
        .to(device=device, dtype=diffusion_output.dtype)[env_ids]
        .unsqueeze(1)
    )
    size_x = (
        tsdf_meta["size_x"].to(device=device, dtype=diffusion_output.dtype)[env_ids].unsqueeze(1)
    )
    size_z = (
        tsdf_meta["size_z"].to(device=device, dtype=diffusion_output.dtype)[env_ids].unsqueeze(1)
    )
    ix = (rotated[..., 0] - start_x) / res
    iz = (rotated[..., 1] - start_z) / res
    nx = ix / (size_x - 1.0) * 2.0 - 1.0
    nz = iz / (size_z - 1.0) * 2.0 - 1.0
    grid = torch.stack([nz, nx], dim=-1).unsqueeze(2)
    cost_maps = tsdf_cost_maps.to(device=device, dtype=diffusion_output.dtype)
    cost_maps = cost_maps[env_ids].unsqueeze(1)
    sampled = F.grid_sample(
        cost_maps, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    costs = sampled.squeeze(1).squeeze(-1)
    costs = costs * valid_mask.unsqueeze(1)
    cost_scale = float(guide_cfg.get("cost_scale", 0.003))
    loss = cost_scale * costs.sum()
    grad = torch.autograd.grad(
        loss, rotated, retain_graph=False, create_graph=False, allow_unused=True
    )[0]
    if grad is None:
        return diffusion_output
    scale = float(guide_cfg.get("scale", 1.0))
    rotated = rotated - scale * grad
    rotated = rotated.detach()
    if pos_scale != 1.0:
        rotated = rotated / pos_scale
    actions = _global_deltas_to_actions_torch(rotated, angle_flat)
    ndeltas = _actions_to_ndeltas_torch(actions, ACTION_STATS)
    if ndeltas is None or not torch.is_tensor(ndeltas):
        if _tsdf_fallback_enabled():
            return diffusion_output
        return None
    return ndeltas


@baseline_registry.register_policy
class PolicyViewSelectionBEV(ILPolicy):

    def __init__(self, observation_space: Space, action_space: Space, model_config: Config):
        super().__init__(
            BEV(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: Space, action_space: Space):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class Critic(nn.Module):

    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Dropout(drop_ratio), nn.Linear(512, 1)
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class BEV(Net):

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device
        self.vln_bert = get_vlnbert_models(config=model_config)
        self.nomad, self.depth_prior, self.noise_scheduler, dp_params = get_dif_wp_models(
            config=model_config
        )
        self.action_dim = dp_params["action_dim"]
        self.pred_horizon = dp_params["len_traj_pred"]
        self.drop_env = nn.Dropout(p=0.4)
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=2))
        self.grid_pool_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d((14, 14)), nn.Flatten(start_dim=1)
        )
        self.rgb_encoder = CLIPEncoderB16(self.device)
        self.twp_rgb_encoder = RGBEncoder(self.device)
        self.wp_rgb_encoder = RGBResEncoder(self.device)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=2))
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)
        pano_angle_rad_c = (1 - self.pano_img_idxes / 12) * 2 * math.pi
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

    @property
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(
        self,
        mode=None,
        txt_ids=None,
        txt_masks=None,
        txt_embeds=None,
        waypoint_predictor=None,
        observations=None,
        in_train=True,
        rgb_fts=None,
        dep_fts=None,
        loc_fts=None,
        nav_types=None,
        view_lens=None,
        gmap_vp_ids=None,
        gmap_step_ids=None,
        gmap_img_fts=None,
        gmap_pos_fts=None,
        gmap_masks=None,
        gmap_visited_masks=None,
        gmap_pair_dists=None,
        bev_fts=None,
        bev_pos_fts=None,
        bev_masks=None,
        bev_nav_masks=None,
        bev_cand_idxs=None,
        bev_cand_vpids=None,
        batch=None,
        observation_conditions=None,
        direction_labels=None,
        angle_labels=None,
        distance_labels=None,
        gmap_front_vp_fts=None,
        gmap_traj_fts=None,
        gmap_traj_mask=None,
        bev_cand_trajs_ids=None,
        bev_traj_mask=None,
        tsdf_guidance_enabled=None,
        tsdf_cost_maps=None,
        tsdf_meta=None,
        tsdf_angle_labels=None,
        tsdf_pad_mask=None,
        tsdf_valid_mask=None,
        tsdf_guide_cfg=None,
    ):
        if mode == "language":
            encoded_sentence = self.vln_bert.forward_txt(txt_ids, txt_masks)
            return encoded_sentence
        elif mode == "waypoint":
            batch_size = observations["rgb"].shape[0]
            NUM_ANGLES = 120
            NUM_IMGS = 12
            NUM_CLASSES = 12
            depth_batch = torch.zeros_like(observations["depth"]).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations["rgb"]).repeat(NUM_IMGS, 1, 1, 1)
            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if "depth" in k:
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                        depth_batch[ra_count + bi * NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi * NUM_IMGS] = observations[
                            k.replace("depth", "rgb")
                        ][bi]
                    a_count += 1
            obs_view12 = {}
            obs_view12["depth"] = depth_batch
            obs_view12["rgb"] = rgb_batch
            depth_embeds = self.depth_encoder(obs_view12)
            depth_grid_embeds = self.grid_pool_depth(depth_batch.permute(0, 3, 1, 2))
            rgb_embeds, rgb_grid_embeds = self.rgb_encoder(obs_view12)
            wp_rgb_embedding = self.twp_rgb_encoder(obs_view12)
            wp_res_rgb_embedding = self.wp_rgb_encoder(obs_view12)
            batch = {"img_fts": wp_rgb_embedding, "dep_fts": depth_embeds}
            observation_condition, waypoint_heatmap_logits = self.nomad(
                "vision_encoder", batch=batch
            )
            waypoint_heatmap_logits = waypoint_predictor(wp_res_rgb_embedding, depth_embeds)
            rgb_embeds_reshape = rgb_embeds.reshape(batch_size, NUM_IMGS, 768, 1, 1)
            rgb_feats = torch.cat(
                [rgb_embeds_reshape[:, 0:1, :], torch.flip(rgb_embeds_reshape[:, 1:, :], [1])],
                dim=1,
            )
            depth_embeds_reshape = depth_embeds.reshape(batch_size, NUM_IMGS, 128, 4, 4)
            depth_feats = torch.cat(
                [depth_embeds_reshape[:, 0:1, :], torch.flip(depth_embeds_reshape[:, 1:, :], [1])],
                dim=1,
            )
            rgb_grid_feats = rgb_grid_embeds.reshape(batch_size, NUM_IMGS, 196, 768)
            depth_grid_feats = depth_grid_embeds.reshape(batch_size, NUM_IMGS, 196, 1)
            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(batch_size, NUM_ANGLES * NUM_CLASSES), dim=1
            )
            batch_x_norm = batch_x_norm.reshape(batch_size, NUM_ANGLES, NUM_CLASSES)
            batch_x_norm_wrap = torch.cat(
                (batch_x_norm[:, -1:, :], batch_x_norm, batch_x_norm[:, :1, :]), dim=1
            )
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=5, sigma=(7.0, 5.0)
            )
            batch_output_map = batch_output_map.squeeze(1)[:, 1:-1, :]
            if in_train:
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (
                        waypoint_heatmap_logits[:, -HEATMAP_OFFSET:, :],
                        waypoint_heatmap_logits[:, :-HEATMAP_OFFSET, :],
                    ),
                    dim=1,
                )
                batch_way_heats_regional = batch_way_heats_regional.reshape(batch_size, 12, 10, 12)
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                for j in range(batch_size):
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    img_idxes = (angle_idxes.cpu().numpy() + 5) // 10
                    img_idxes[img_idxes == 12] = 0
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(
                        img_idxes.size, -1
                    )
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(way_act // 12 + angle_pointer)
                        sample_distance_idxes.append(way_act % 12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)
            rgb_feats = self.space_pool_rgb(rgb_feats)
            depth_feats = self.space_pool_depth(depth_feats)
            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_angle_idxes = []
            cand_distances = []
            cand_distance_idxes = []
            dp_img_idx = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                angle_rad_c = angle_idxes.cpu().float() / 120 * 2 * math.pi
                angle_rad_cc = 2 * math.pi - angle_idxes.float() / 120 * 2 * math.pi
                cand_angle_fts.append(angle_feature_torch(angle_rad_c))
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append(((distance_idxes + 1) * 0.25).tolist())
                img_idxes = 12 - (angle_idxes.cpu().numpy() + 5) // 10
                img_idxes[img_idxes == 12] = 0
                cand_img_idxes.append(img_idxes)
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])
                c_img_idx = (angle_idxes.cpu().numpy() + 5) // 10
                c_img_idx[c_img_idx == 12] = 0
                dp_img_idx.append(torch.as_tensor(c_img_idx, device=distance_idxes.device))
                distance_idxes = torch.where(
                    distance_idxes > 11,
                    torch.tensor(11, device=distance_idxes.device, dtype=distance_idxes.dtype),
                    distance_idxes,
                )
                cand_angle_idxes.append(angle_idxes)
                cand_distance_idxes.append(distance_idxes)
            pano_rgb = rgb_feats
            pano_depth = depth_feats
            pano_angle_fts = deepcopy(self.pano_angle_fts)
            pano_img_idxes = deepcopy(self.pano_img_idxes)
            outputs = {
                "cand_rgb": cand_rgb,
                "cand_depth": cand_depth,
                "cand_angle_fts": cand_angle_fts,
                "cand_img_idxes": cand_img_idxes,
                "cand_angles": cand_angles,
                "cand_distances": cand_distances,
                "pano_rgb": pano_rgb,
                "pano_depth": pano_depth,
                "pano_rgb_grid": rgb_grid_feats,
                "pano_depth_grid": depth_grid_feats,
                "pano_angle_fts": pano_angle_fts,
                "pano_img_idxes": pano_img_idxes,
                "angle_idxes": cand_angle_idxes,
                "distance_idxes": cand_distance_idxes,
                "img_idxes": dp_img_idx,
                "rgb_embedding": wp_rgb_embedding,
                "depth_embedding": depth_embeds,
                "observation_condition": observation_condition,
            }
            return outputs
        elif mode == "panorama":
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(rgb_fts, dep_fts, loc_fts, nav_types, view_lens)
            return outs
        elif mode == "diffusion_observation":
            outs = self.nomad(
                "fusion_layer",
                global_cond=observation_conditions,
                direction_label=direction_labels,
                angle_label=angle_labels,
                distance_label=distance_labels,
            )
            return outs
        elif mode == "dp_trajectory":
            bs, dirs, hidden_dim = observation_conditions.shape
            observation_conditions = observation_conditions.reshape(-1, hidden_dim)
            noisy_diffusion_output = torch.randn(
                (len(observation_conditions), self.pred_horizon, self.action_dim),
                device=self.device,
            )
            diffusion_output = noisy_diffusion_output
            tsdf_guidance = (
                bool(tsdf_guidance_enabled)
                and tsdf_cost_maps is not None
                and (tsdf_meta is not None)
                and (tsdf_angle_labels is not None)
            )
            tsdf_cfg = tsdf_guide_cfg or {}
            for t_idx, k in enumerate(self.noise_scheduler.timesteps[:]):
                noise_pred = self.nomad(
                    "noise_pred_net",
                    sample=diffusion_output,
                    timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device),
                    global_cond=observation_conditions,
                )
                diffusion_output = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=diffusion_output
                ).prev_sample
                if tsdf_guidance:
                    max_step = int(tsdf_cfg.get("max_step", 0))
                    period = int(tsdf_cfg.get("period", 1))
                    step_ok = max_step < 0 or t_idx < max_step
                    if step_ok and t_idx % max(1, period) == 0:
                        diffusion_output_prev = diffusion_output
                        with torch.enable_grad():
                            diffusion_output = diffusion_output.detach().requires_grad_(True)
                            diffusion_output = apply_tsdf_guidance(
                                diffusion_output,
                                tsdf_angle_labels,
                                tsdf_cost_maps,
                                tsdf_meta,
                                tsdf_pad_mask,
                                tsdf_valid_mask,
                                tsdf_cfg,
                            )
                        if diffusion_output is None:
                            if _tsdf_fallback_enabled():
                                diffusion_output = diffusion_output_prev
                            else:
                                raise RuntimeError(
                                    "TSDF guidance returned None and fallback is disabled."
                                )
                        diffusion_output = diffusion_output.detach()
            _, pred_len, act_dim = diffusion_output.shape
            outs = get_action(diffusion_output, ACTION_STATS)
            outs = outs.reshape(bs, dirs, pred_len, act_dim)
            return outs
        elif mode == "navigation":
            bev_fts = self.drop_env(bev_fts)
            outs = self.vln_bert.forward_navigation(
                txt_embeds,
                txt_masks,
                gmap_vp_ids,
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
            )
            return outs


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
