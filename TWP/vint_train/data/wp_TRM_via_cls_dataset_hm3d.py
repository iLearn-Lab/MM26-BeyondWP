from acm_paths import feature_root
import numpy as np
import h5py
import jsonlines
import random
from PIL import Image
import torchvision
from torchvision.transforms import v2
import json
import pickle as pkl
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.ndimage import gaussian_filter
from .common import pad_tensors, gen_seq_masks, random_mapping
from .data_utils import (
    to_local_coords,
    to_wp_polar_coords,
    get_angle_fts,
    get_view_rel_angles,
    heading_from_quaternion,
)


class Hm3d_WpDataset(Dataset):

    def __init__(self, anno_files, nomad_config, data_config, in_memory=True, angle_feat_size=4):
        self.data = []
        self.data_id = []
        for anno_file in anno_files:
            with open(anno_file) as f:
                wp_dict = json.load(f)
                for scan, nodes in wp_dict.items():
                    for node_a, info in nodes.items():
                        info["source_node"] = node_a
                        info["scan"] = scan.split("-")[1]
                        self.data.append(info)
        self.RGB_INPUT_DIM = 224
        self.DEPTH_INPUT_DIM = 256
        self.NUM_IMGS = 12
        self.rgb_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((self.RGB_INPUT_DIM, self.RGB_INPUT_DIM), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.len_traj_pred = nomad_config["len_traj_pred"]
        self.angle_nums = nomad_config["angle_nums"]
        self.distance_nums = nomad_config["distance_nums"]
        self.metric_waypoint_spacing = data_config["metric_waypoint_spacing"]
        self.in_memory = in_memory
        self.normalize = nomad_config["normalize"]
        if self.in_memory:
            self._feature_store = {}
            self._feature_store_depth = {}
        self.angle_feat_size = angle_feat_size
        self.all_point_rel_angles = get_view_rel_angles(baseViewId=0)
        self.num_imgs = nomad_config["num_imgs"]

    def __len__(self):
        return len(self.data)

    def get_traj_pano_fts(self, scan, node_id):
        view_fts, dep_fts = self.get_scanvp_feature(scan, node_id)
        view_img_fts, view_dep_fts, view_angles = ([], [], [])
        view_img_fts = view_fts
        view_dep_fts = dep_fts
        view_angles = self.all_point_rel_angles
        view_angles = np.stack(view_angles, 0)
        view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
        last_vp_angles = view_angles
        return (view_img_fts, view_dep_fts, view_ang_fts, last_vp_angles)

    def get_scanvp_feature(self, scan, node_id):
        key = "%s_%s" % (scan, node_id)
        if self.in_memory and key in self._feature_store:
            rgb_feats = self._feature_store[key]
            depth_feats = self._feature_store_depth[key]
        else:
            feat_dir = f"{feature_root()}/hm3d/rgbd_fov90_d_1_25_fts/hm3d/{scan}/{scan}_{node_id}_hfov90_hm3d_imgs.pkl"
            rgb_depth_fts = pkl.load(open(feat_dir, "rb"))
            rgb_feats = torch.from_numpy(rgb_depth_fts["rgb"])
            depth_feats = torch.from_numpy(rgb_depth_fts["depth"])
            if self.in_memory:
                self._feature_store[key] = rgb_feats
                self._feature_store_depth[key] = depth_feats
        return (rgb_feats, depth_feats)

    def _compute_actions(self, wp_index, trajectories):
        positions = [d[: self.len_traj_pred + 1] for d in trajectories]
        start_positions = [np.array(position[0]) for position in positions]
        positions = [np.array(position[1:]) for position in positions]
        angle_index = [ad[0] for ad in wp_index]
        for i in range(len(start_positions)):
            start_pos = start_positions[i]
            if start_pos.shape:
                break
            if i == len(start_positions) - 1:
                raise ValueError
        for position in positions:
            assert position.shape == (
                self.len_traj_pred,
                2,
            ), f"{position.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
        try:
            actions, direction_labels = to_wp_polar_coords(positions, start_pos, angle_index)
        except Exception:
            raise ValueError
        for action in actions:
            assert action.shape == (
                self.len_traj_pred,
                2,
            ), f"{action.shape} and {(self.len_traj_pred, 2)} should be equal"
        actions = np.stack(actions, axis=0)
        if self.normalize:
            actions[:, :, :2] /= self.metric_waypoint_spacing
        return (actions, direction_labels)

    def __getitem__(self, idx):
        input = self.data[idx]
        scan = input["scan"]
        source_node = input["source_node"]
        direction_angle_labels = input["target"]
        wp_node = input["target_node"]
        wp_index = input["target_index"]
        trajectories = input["full_locations"]
        action_mask = input["locations_mask"]
        traj_locations = []
        traj_steps = []
        for node in wp_node:
            traj_locations.append(trajectories[node])
        action_masks = []
        for node in wp_node:
            a_m = action_mask[node]
            action_length = np.zeros(self.len_traj_pred)
            action_length[:a_m] = 1.0
            action_masks.append(action_length)
        view_img_fts, view_dep_fts, view_ang_fts, _ = self.get_traj_pano_fts(scan, source_node)
        action_labels, direction_labels = self._compute_actions(wp_index, traj_locations)
        direction_angle_labels = np.array(direction_angle_labels)
        dir_label = np.zeros(self.num_imgs)
        for cand_d in direction_labels:
            dir_label[cand_d] = 1
        output = {}
        output["img_fts"] = view_img_fts
        output["dep_fts"] = view_dep_fts
        output["view_ang_fts"] = torch.from_numpy(view_ang_fts)
        output["action_labels"] = torch.from_numpy(action_labels)
        output["direction_onehot"] = torch.from_numpy(dir_label)
        output["direction_label"] = torch.tensor(direction_labels)
        output["num_directions"] = len(direction_labels)
        action_masks = np.array(action_masks)
        output["action_masks"] = torch.from_numpy(action_masks)
        output["angle_dirction"] = torch.from_numpy(direction_angle_labels)
        angle_index = torch.tensor([ad[0] for ad in wp_index])
        distance_index = torch.tensor([ad[1] for ad in wp_index])
        output["angle_label"] = angle_index
        output["distance_label"] = distance_index
        return output


def wp_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}
    batch["img_fts"] = torch.stack(batch["img_fts"], dim=0)
    batch["dep_fts"] = torch.stack(batch["dep_fts"], dim=0)
    batch["view_ang_fts"] = torch.stack(batch["view_ang_fts"], dim=0)
    batch["action_labels"] = pad_tensors(batch["action_labels"])
    batch["direction_onehot"] = torch.stack(batch["direction_onehot"], dim=0)
    batch["direction_label"] = pad_tensors(batch["direction_label"])
    batch["mask"] = gen_seq_masks(batch["num_directions"])
    batch["mask"] = torch.from_numpy(batch["mask"])
    batch["angle_onehot"] = torch.stack(batch["angle_dirction"], dim=0)
    batch["angle_label"] = pad_tensors(batch["angle_label"])
    batch["distance_label"] = pad_tensors(batch["distance_label"])
    batch["action_masks"] = pad_tensors(batch["action_masks"])
    return batch
