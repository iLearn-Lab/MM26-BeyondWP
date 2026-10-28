import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple
import quaternion
from typing import List
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
import math
from typing import Union
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3


def heading_from_quaternion(quat: list) -> float:
    quat = quaternion.quaternion(quat[0], quat[1], quat[2], quat[3])
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)


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


def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError
    return (positions - curr_pos).dot(rotmat)


def to_wp_polar_coords(
    positions: List[np.ndarray], curr_pos: np.ndarray, angle_index: List[int], offset_deg=0
) -> List[np.ndarray]:
    for position in positions:
        if not position.shape[-1] == 2:
            raise ValueError
    results = []
    direction_labels = []
    dir_step = 10
    ang_step = np.pi / 6
    ldir_max = 5
    for i, pos in enumerate(positions):
        delta_pos = pos - curr_pos[None, :]
        angle_id = angle_index[i]
        r = np.linalg.norm(delta_pos, axis=1)
        global_directions = -np.arctan2(1.0, 0.0) + np.arctan2(delta_pos[:, 1], delta_pos[:, 0])
        global_directions = (global_directions + math.radians(offset_deg)) % (2 * np.pi)
        n_d = 0
        while angle_id - n_d * dir_step > ldir_max:
            n_d += 1
        if n_d == 12:
            n_d = 0
        direction_labels.append(n_d)
        local_directions = global_directions - n_d * ang_step
        local_directions = (local_directions + 2 * math.pi) % (2 * math.pi)
        x = r * np.cos(local_directions)
        z = r * np.sin(local_directions)
        result = np.stack([x, z], axis=1)
        results.append(result)
    return (results, direction_labels)


def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {"image": ".jpg"}
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image,
    transform: transforms,
    image_resize_size: Tuple[int, int],
    aspect_ratio: float = IMAGE_ASPECT_RATIO,
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return (viz_img, transf_img)


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(
    path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]
) -> torch.Tensor:
    return resize_and_aspect_crop(Image.open(path), image_resize_size)
