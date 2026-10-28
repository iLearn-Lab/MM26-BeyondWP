import math
from turtle import heading
import torch
import numpy as np


def angle_feature(headings, device=None):
    heading_enc = torch.zeros(len(headings), 64, dtype=torch.float32)
    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor([math.sin(head), math.cos(head)] * (64 // 2))
    return heading_enc.to(device)


def dir_angle_feature(angle_list, device=None):
    feature_dim = 64
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1
    heading_enc = torch.zeros(batch_size, max_leng, feature_dim, dtype=torch.float32)
    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
                [math.sin(angle_rad), math.cos(angle_rad)] * (feature_dim // 2)
            )
    return heading_enc


def angle_feature_with_ele(headings, device=None):
    heading_enc = torch.zeros(len(headings), 128, dtype=torch.float32)
    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor(
            [math.sin(head), math.cos(head), math.sin(0.0), math.cos(0.0)] * (128 // 4)
        )
    return heading_enc.to(device)


def angle_feature_torch(headings: torch.Tensor):
    return (
        torch.stack(
            [
                torch.sin(headings),
                torch.cos(headings),
                torch.sin(torch.zeros_like(headings)),
                torch.cos(torch.zeros_like(headings)),
            ]
        )
        .float()
        .T
    )


def dir_angle_feature_with_ele(angle_list, device=None):
    feature_dim = 128
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1
    heading_enc = torch.zeros(batch_size, max_leng, feature_dim, dtype=torch.float32)
    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
                [math.sin(angle_rad), math.cos(angle_rad), math.sin(0.0), math.cos(0.0)]
                * (128 // 4)
            )
    return heading_enc


def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (
        torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
        > (torch.LongTensor(length) - 1).unsqueeze(1)
    ).cuda()
    return mask


def generate_bev_path(node_a, node_b, step=0.25, include_end=False):
    x_a, y_a, z_a = (node_a[0], node_a[1], node_a[2])
    x_b, y_b, z_b = (node_b[0], node_b[1], node_b[2])
    a = np.array([x_a, y_a, z_a], dtype=np.float32)
    b = np.array([x_b, y_b, z_b], dtype=np.float32)
    vec = b - a
    dist = np.linalg.norm(vec)
    if dist == 0:
        return np.array([a])
    direction = vec / dist
    n_steps = int(np.floor(dist / step))
    if n_steps == 0:
        path = np.array([a])
    else:
        path = np.array([a + i * step * direction for i in range(n_steps)])
    if include_end and (not np.allclose(path[-1], b)):
        path = np.vstack([path, b])
    return path
