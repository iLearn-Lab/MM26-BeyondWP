import os
import math
import json
import numpy as np
import networkx as nx
import random
import torch


def pad_tensors(tensors, lens=None, pad=0):
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid
    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def random_mapping(small_list, large_list):
    if len(small_list) > len(large_list):
        raise ValueError("small_list cannot be larger than large_list")
    sampled_large = random.sample(large_list, k=len(small_list))
    mapping = dict(zip(small_list, sampled_large))
    return mapping


def gen_seq_masks(seq_lens, max_len=None):
    seq_lens = np.array(seq_lens)
    if max_len is None:
        max_len = max(seq_lens)
    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=np.bool)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


def get_view_rel_angles(baseViewId=0):
    rel_angles = np.zeros((12, 2), dtype=np.float32)
    base_heading = 0
    base_elevation = 0
    elevation = 0
    for ix in range(12):
        if ix == 0:
            heading = 0
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation - base_elevation
    return rel_angles


def load_nav_graphs(connectivity_file):

    def distance(pose1, pose2):
        return (
            (pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2 + (pose1[2] - pose2[2]) ** 2
        ) ** 0.5

    scans = [x.strip() for x in open(connectivity_file).readlines()]
    graphs = {}
    for scan in scans:
        G = nx.Graph()
        scan = json.loads(scan)
        episode_id = scan["episode_id"]
        positions = {}
        for i in range(len(scan["vp_ids"])):
            cur_vp_id = scan["vp_ids"][i]
            cur_vp_pos = scan["vp_pos"][i]
            positions[cur_vp_id] = np.array(cur_vp_pos)
            if i == len(scan["vp_ids"]) - 1:
                pass
            else:
                next_vp_id = scan["vp_ids"][i + 1]
                next_vp_pos = scan["vp_pos"][i + 1]
                G.add_edge(cur_vp_id, next_vp_id, weight=distance(cur_vp_pos, next_vp_pos))
            nx.set_node_attributes(G, values=positions, name="position")
        graphs[episode_id] = G
    shortest_distances = {}
    shortest_paths = {}
    for episode_id, G in graphs.items():
        shortest_distances[episode_id] = dict(nx.all_pairs_dijkstra_path_length(G))
        shortest_paths[episode_id] = dict(nx.all_pairs_dijkstra_path(G))
    return (graphs, shortest_distances, shortest_paths)


def softmax(logits, dim=1):
    tmp = np.exp(logits)
    return tmp / np.sum(tmp, axis=dim, keepdims=True)


def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0, to_clock=False):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-08)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-08)
    heading = np.arcsin(-dx / xz_dist)
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    if to_clock:
        heading = 2 * np.pi - heading
    elevation = np.arcsin(dz / xyz_dist)
    elevation -= base_elevation
    return (heading, elevation, xyz_dist)


def normalize_angle(x):
    pi2 = 2 * math.pi
    x = x % pi2
    x = np.where(x > math.pi, x - pi2, x)
    return x
