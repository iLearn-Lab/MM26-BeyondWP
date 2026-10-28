import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np

try:
    import quaternion
except Exception:
    quaternion = None
try:
    from scipy.ndimage import binary_dilation, gaussian_filter, distance_transform_edt
except Exception:
    binary_dilation = None
    gaussian_filter = None
    distance_transform_edt = None


@dataclass
class TsdfCostMapConfig:
    resolution: float = 0.1
    map_size_m: float = 10.0
    min_depth: float = 0.1
    max_depth: float = 5.0
    floor_height: float = -1.37
    ceiling_height: float = 0.43
    floor_slice: float = 0.2
    use_camera_height: bool = True
    sigma_expand: float = 2.0
    obstacle_threshold: float = 0.01
    free_space_threshold: float = 0.8
    sigma_smooth: float = 1.5
    inflate_obstacles: bool = True
    fill_floor_square: bool = True
    fill_floor_height: float = 1.29
    fill_floor_fov_deg: float = 90.0
    save_ply: bool = False
    ply_dir: str = "scripts/hm3d_training_data/tsdf_ply"


class TsdfCostMap:

    def __init__(
        self,
        cfg: Optional[TsdfCostMapConfig] = None,
        intrinsics: Optional[np.ndarray] = None,
        robot_width: Optional[float] = None,
    ):
        if binary_dilation is None or gaussian_filter is None or distance_transform_edt is None:
            raise ImportError("scipy.ndimage is required for TSDF guidance")
        self.cfg = cfg or TsdfCostMapConfig()
        self.intrinsics = intrinsics
        self.robot_width = float(robot_width) if robot_width is not None else 0.2
        self.last_cost_map = None
        self.last_meta = None
        self.initial_position = None

    def build_from_pano_depths(
        self,
        depths,
        cam_pos,
        base_rotation,
        angle_step_deg: float = 90.0,
        angle_count: int = 4,
        angle_ccw: bool = True,
    ):
        self._set_initial_position(cam_pos)
        points_all = []
        if depths and isinstance(depths[0], (tuple, list)) and (len(depths[0]) >= 3):
            for depth, pos, rot in depths:
                depth = np.array(depth, dtype=np.float32)
                if np.nanmax(depth) <= 1.0 + 0.001:
                    depth = depth * float(self.cfg.max_depth)
                depth = self._preprocess_depth(depth, self.cfg.min_depth, self.cfg.max_depth)
                if np.sum(depth) <= 0:
                    continue
                cam_points = self._get_pointcloud_from_depth(depth, self.intrinsics)
                if cam_points.shape[0] == 0:
                    continue
                rel_pos = self._get_relative_position(np.array(pos, dtype=np.float32))
                rot_m = self._habitat_rotation(rot)
                points = self._translate_to_world(cam_points, rel_pos, rot_m)
                if points.shape[0] > 0:
                    points_all.append(points)
        else:
            for i in range(angle_count):
                yaw_sign = 1.0 if angle_ccw else -1.0
                yaw = math.radians(angle_step_deg * i * yaw_sign)
                rot = self._compose_yaw(base_rotation, yaw)
                points = self._depth_to_world_points(depths[i], cam_pos, rot)
                if points.shape[0] > 0:
                    points_all.append(points)
        if len(points_all) == 0:
            obs_points = np.zeros((0, 3), dtype=np.float32)
            free_points = np.zeros((0, 3), dtype=np.float32)
        else:
            points_world = np.concatenate(points_all, axis=0)
            obs_points, free_points = self._split_obstacle_free(points_world, cam_pos)
        center_x, center_z = (0.0, 0.0)
        cost_map, meta = self._create_tsdf_map(obs_points, free_points, center_x, center_z)
        self.last_cost_map = cost_map
        self.last_meta = meta
        return (cost_map, meta)

    def _depth_to_world_points(self, depth: np.ndarray, cam_pos, cam_rot) -> np.ndarray:
        if self.intrinsics is None:
            raise ValueError("Camera intrinsics are missing.")
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth[:, :, 0]
            elif depth.shape[0] == 1:
                depth = depth[0]
        if np.nanmax(depth) <= 1.0 + 0.001:
            depth = depth * float(self.cfg.max_depth)
        depth = self._preprocess_depth(depth, self.cfg.min_depth, self.cfg.max_depth)
        if np.sum(depth) <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        cam_points = self._get_pointcloud_from_depth(depth, self.intrinsics)
        if cam_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        current_pos = self._get_relative_position(np.array(cam_pos, dtype=np.float32))
        current_rot = self._habitat_rotation(cam_rot)
        points_world = (current_rot @ cam_points.T).T + current_pos
        return points_world

    def _rotation_matrix(self, rot) -> np.ndarray:
        if isinstance(rot, np.ndarray) and rot.shape == (3, 3):
            return rot.astype(np.float32)
        if hasattr(rot, "rotation_matrix"):
            return np.array(rot.rotation_matrix, dtype=np.float32)
        if hasattr(rot, "w") and hasattr(rot, "x") and hasattr(rot, "y") and hasattr(rot, "z"):
            w, x, y, z = (float(rot.w), float(rot.x), float(rot.y), float(rot.z))
            return np.array(
                [
                    [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
                ],
                dtype=np.float32,
            )
        raise ValueError("Unsupported rotation type for camera pose.")

    def _compose_yaw(self, base_rot, yaw: float):
        if (
            hasattr(base_rot, "w")
            and hasattr(base_rot, "x")
            and hasattr(base_rot, "y")
            and hasattr(base_rot, "z")
        ):
            bw, bx, by, bz = (
                float(base_rot.w),
                float(base_rot.x),
                float(base_rot.y),
                float(base_rot.z),
            )
            base_q = np.array([bw, bx, by, bz], dtype=np.float32)
        else:
            base_q = None
        if base_q is None:
            rot_m = self._rotation_matrix(base_rot)
            yaw_m = np.array(
                [
                    [math.cos(yaw), 0.0, math.sin(yaw)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(yaw), 0.0, math.cos(yaw)],
                ],
                dtype=np.float32,
            )
            return rot_m @ yaw_m
        yw = math.cos(yaw / 2.0)
        yy = math.sin(yaw / 2.0)
        yaw_q = np.array([yw, 0.0, yy, 0.0], dtype=np.float32)
        w1, x1, y1, z1 = base_q
        w2, x2, y2, z2 = yaw_q
        out = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

        class _Q:

            def __init__(self, q):
                self.w, self.x, self.y, self.z = q

        return _Q(out)

    def _split_obstacle_free(
        self, points_world: np.ndarray, cam_pos: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if points_world.shape[0] == 0:
            return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
        h = points_world[:, 2]
        floor_h = self.cfg.floor_height
        ceiling_h = self.cfg.ceiling_height
        if self.cfg.use_camera_height and cam_pos is not None:
            cam_h = self._get_relative_position(np.array(cam_pos, dtype=np.float32))[2]
            floor_h = cam_h + self.cfg.floor_height
            ceiling_h = cam_h + self.cfg.ceiling_height
        free_mask = (h < floor_h) & (h > floor_h - self.cfg.floor_slice)
        obs_mask = (h > floor_h) & (h < ceiling_h)
        return (points_world[obs_mask], points_world[free_mask])

    def _create_tsdf_map(
        self, obs_points: np.ndarray, free_points: np.ndarray, center_x: float, center_z: float
    ):
        half = self.cfg.map_size_m / 2.0
        start_x = center_x - half
        start_z = center_z - half
        num_x = int(round(self.cfg.map_size_m / self.cfg.resolution))
        num_z = int(round(self.cfg.map_size_m / self.cfg.resolution))
        obs_map = np.ones((num_x, num_z), dtype=np.float32)
        free_map = np.zeros((num_x, num_z), dtype=np.float32)
        obs_idx = self._points_to_indices(obs_points, start_x, start_z)
        free_idx = self._points_to_indices(free_points, start_x, start_z)
        for i, j in obs_idx:
            if 0 <= i < num_x and 0 <= j < num_z:
                obs_map[i, j] = 0.0
        for i, j in free_idx:
            if 0 <= i < num_x and 0 <= j < num_z:
                free_map[i, j] = 1.0
        if self.cfg.fill_floor_square:
            half_size_m = float(self.cfg.fill_floor_height) * math.tan(
                math.radians(self.cfg.fill_floor_fov_deg / 2.0)
            )
            half_cells = max(1, int(round(half_size_m / self.cfg.resolution)))
            cx = int(round((0.0 - start_x) / self.cfg.resolution))
            cz = int(round((0.0 - start_z) / self.cfg.resolution))
            x0 = max(0, cx - half_cells)
            x1 = min(num_x, cx + half_cells + 1)
            z0 = max(0, cz - half_cells)
            z1 = min(num_z, cz + half_cells + 1)
            free_map[x0:x1, z0:z1] = 1.0
        if self.cfg.inflate_obstacles and obs_idx.size > 0:
            radius = max(1, int(round(self.robot_width / 2.0 / self.cfg.resolution)))
            obs_map = binary_dilation(obs_map < 0.5, iterations=radius).astype(np.float32)
            obs_map = 1.0 - obs_map
        free_map[free_map > self.cfg.free_space_threshold] = 1.0
        free_map[obs_map < 1.0 - self.cfg.obstacle_threshold] = 0.0
        tsdf = distance_transform_edt(free_map)
        tsdf[tsdf > 0.0] = np.log(tsdf[tsdf > 0.0] + math.e)
        tsdf = gaussian_filter(tsdf, sigma=self.cfg.sigma_smooth)
        meta = {
            "start_x": float(start_x),
            "start_z": float(start_z),
            "resolution": float(self.cfg.resolution),
            "size": (int(num_x), int(num_z)),
            "center": (float(center_x), float(center_z)),
        }
        return (tsdf, meta)

    def _points_to_indices(self, points: np.ndarray, start_x: float, start_z: float) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int32)
        xs = points[:, 0]
        zs = points[:, 1]
        ix = np.floor((xs - start_x) / self.cfg.resolution).astype(np.int32)
        iz = np.floor((zs - start_z) / self.cfg.resolution).astype(np.int32)
        return np.stack((ix, iz), axis=1)

    def _preprocess_depth(self, depth: np.ndarray, lower: float, upper: float) -> np.ndarray:
        depth[(depth < lower) | (depth > upper)] = 0
        return depth

    def _get_pointcloud_from_depth(self, depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        filter_z, filter_x = np.where(depth > 0)
        if filter_z.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        depth_values = depth[filter_z, filter_x]
        pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
        pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
        pixel_y = depth_values
        return np.stack([pixel_x, pixel_z, -pixel_y], axis=-1).astype(np.float32)

    def _habitat_translation(self, position: np.ndarray) -> np.ndarray:
        return np.array([position[0], position[2], position[1]], dtype=np.float32)

    def _habitat_rotation(self, rotation) -> np.ndarray:
        if isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            rot_m = rotation.astype(np.float32)
        elif (
            hasattr(rotation, "w")
            and hasattr(rotation, "x")
            and hasattr(rotation, "y")
            and hasattr(rotation, "z")
        ):
            if quaternion is None:
                w, x, y, z = (
                    float(rotation.w),
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                )
                rot_m = np.array(
                    [
                        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
                    ],
                    dtype=np.float32,
                )
            else:
                rotation = quaternion.quaternion(
                    float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)
                )
                rot_m = quaternion.as_rotation_matrix(rotation).astype(np.float32)
        else:
            raise ValueError("Unsupported rotation type for camera pose.")
        transform = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
        return transform @ rot_m

    def _get_relative_position(self, position: np.ndarray) -> np.ndarray:
        trans = self._habitat_translation(position)
        if self.initial_position is None:
            self.initial_position = trans
        return trans - self.initial_position

    def _set_initial_position(self, position: np.ndarray) -> None:
        self.initial_position = self._habitat_translation(np.array(position, dtype=np.float32))

    def _translate_to_world(
        self, pointcloud: np.ndarray, position: np.ndarray, rotation: np.ndarray
    ) -> np.ndarray:
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[0:3, 0:3] = rotation
        extrinsic[0:3, 3] = position
        points_h = np.concatenate(
            (pointcloud, np.ones((pointcloud.shape[0], 1), dtype=np.float32)), axis=-1
        )
        world_points = (extrinsic @ points_h.T).T
        return world_points[:, 0:3]

    def _get_traj_relative_position(self, position: np.ndarray) -> np.ndarray:
        trans = self._habitat_translation(position)
        return trans - self.initial_position

    def world_to_map_index(self, x: float, y: float, z: float, meta: Dict[str, float]):
        rel = self._get_relative_position(np.array([x, y, z], dtype=np.float32))
        ix = int(math.floor((rel[0] - meta["start_x"]) / meta["resolution"]))
        iz = int(math.floor((rel[1] - meta["start_z"]) / meta["resolution"]))
        return (ix, iz)

    def traj_world_to_map_index(self, x: float, y: float, z: float, meta: Dict[str, float]):
        rel = self._get_traj_relative_position(np.array([x, y, z], dtype=np.float32))
        ix = int(math.floor((rel[0] - meta["start_x"]) / meta["resolution"]))
        iz = int(math.floor((rel[1] - meta["start_z"]) / meta["resolution"]))
        return (ix, iz)

    def save_tsdf_png(
        self,
        cost_map: np.ndarray,
        meta: Dict[str, float],
        current_pos: Tuple[float, float, float],
        out_path: str,
        enabled: bool = False,
        traj_before: Optional[np.ndarray] = None,
        traj_after: Optional[np.ndarray] = None,
        traj_relative: bool = False,
    ):
        if not enabled:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise ImportError("matplotlib is required for TSDF visualization") from exc

        def _ensure_trajs(traj):
            if traj is None:
                return []
            if isinstance(traj, np.ndarray):
                if traj.ndim == 2:
                    return [traj]
                if traj.ndim == 3:
                    return [t for t in traj]
            if isinstance(traj, (list, tuple)) and len(traj) == 0:
                return []
            if isinstance(traj, (list, tuple)) and len(traj) > 0:
                first = traj[0]
                if isinstance(first, (list, tuple, np.ndarray)) and len(first) > 0:
                    if isinstance(first[0], (int, float, np.floating, np.integer)):
                        return [np.array(traj, dtype=np.float32)]
            return [np.array(t, dtype=np.float32) for t in traj]

        if traj_relative:
            ix = int(math.floor((current_pos[0] - meta["start_x"]) / meta["resolution"]))
            iz = int(math.floor((current_pos[1] - meta["start_z"]) / meta["resolution"]))
        else:
            ix, iz = self.world_to_map_index(current_pos[0], current_pos[1], current_pos[2], meta)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cost_map.T, origin="upper", cmap="viridis")
        ax.scatter([ix], [iz], c="red", s=25, marker="o", label="current")
        drew_before = False
        for traj in _ensure_trajs(traj_before):
            if traj is None or len(traj) <= 1:
                continue
            pts = []
            for p in traj:
                if traj_relative:
                    px = int(math.floor((p[0] - meta["start_x"]) / meta["resolution"]))
                    pz = int(math.floor((p[1] - meta["start_z"]) / meta["resolution"]))
                else:
                    px, pz = self.traj_world_to_map_index(p[0], p[1], p[2], meta)
                pts.append((px, pz))
            xs = [p[0] for p in pts]
            zs = [p[1] for p in pts]
            ax.plot(
                xs,
                zs,
                color="cyan",
                linewidth=1.5,
                label="traj_before" if not drew_before else None,
            )
            drew_before = True
        drew_after = False
        for traj in _ensure_trajs(traj_after):
            if traj is None or len(traj) <= 1:
                continue
            pts = []
            for p in traj:
                if traj_relative:
                    px = int(math.floor((p[0] - meta["start_x"]) / meta["resolution"]))
                    pz = int(math.floor((p[1] - meta["start_z"]) / meta["resolution"]))
                else:
                    px, pz = self.traj_world_to_map_index(p[0], p[1], p[2], meta)
                pts.append((px, pz))
            xs = [p[0] for p in pts]
            zs = [p[1] for p in pts]
            ax.plot(
                xs,
                zs,
                color="orange",
                linewidth=1.5,
                label="traj_after" if not drew_after else None,
            )
            drew_after = True
        ax.set_title("TSDF Cost Map")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path


@dataclass
class TsdfGuideConfig:
    pos_scale: float = 0.25
    cost_scale: float = 0.003
    scale: float = 1.0
    max_step: int = 3
    period: int = 1
    depth_root: Optional[str] = None
    hfov_deg: float = 90.0
    angle_step_deg: float = 90.0
    angle_count: int = 4
    angle_ccw: bool = True
    pano_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    @classmethod
    def from_env(cls) -> "TsdfGuideConfig":
        return cls(
            pos_scale=float(os.environ.get("TSDF_POS_SCALE", "0.25")),
            cost_scale=float(os.environ.get("TSDF_COST_SCALE", "0.003")),
            scale=float(os.environ.get("TSDF_GUIDE_SCALE", "1.0")),
            max_step=int(os.environ.get("TSDF_MAX_STEP", "3")),
            period=int(os.environ.get("TSDF_PERIOD", "1")),
            hfov_deg=float(os.environ.get("TSDF_HFOV_DEG", "90.0")),
            angle_step_deg=float(os.environ.get("TSDF_ANGLE_STEP_DEG", "90.0")),
            angle_count=int(os.environ.get("TSDF_ANGLE_COUNT", "4")),
            angle_ccw=str(os.environ.get("TSDF_ANGLE_CCW", "1")).lower()
            in ("1", "true", "yes", "y"),
        )

    def update_from(self, data: Optional[Dict[str, object]]) -> "TsdfGuideConfig":
        if not data:
            return self
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def _infer_intrinsics(depth: np.ndarray, hfov_deg: float) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        height, width = (depth.shape[0], depth.shape[1])
    elif depth.ndim == 2:
        height, width = depth.shape
    else:
        height, width = (depth.shape[-2], depth.shape[-1])
    fx = width / 2.0 / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


class TsdfGuide:

    def __init__(self, cfg: Optional[TsdfGuideConfig] = None):
        self.cfg = cfg or TsdfGuideConfig()
        self.tsdf: Optional[TsdfCostMap] = None

    def build_cost_map(
        self,
        depths,
        cam_pos,
        base_rotation,
        guide_cfg: Optional[Dict[str, object]] = None,
        tsdf_cost_cfg: Optional[TsdfCostMapConfig] = None,
    ):
        cfg = TsdfGuideConfig().update_from(self.cfg.as_dict()).update_from(guide_cfg)
        if len(depths) == 0:
            return (None, None)
        use_pose_tuples = isinstance(depths[0], (tuple, list)) and len(depths[0]) >= 3
        if use_pose_tuples:
            selected = list(depths)
            if len(selected) < cfg.angle_count:
                return (None, None)
            if len(selected) > cfg.angle_count:
                idx = cfg.pano_indices
                if len(idx) != cfg.angle_count:
                    idx = list(range(cfg.angle_count))
                selected = [selected[i] for i in idx]
            intrinsics_depth = np.array(selected[0][0], dtype=np.float32)
        else:
            depths_list = [np.array(d, dtype=np.float32) for d in list(depths)]
            if len(depths_list) < cfg.angle_count:
                return (None, None)
            if len(depths_list) > cfg.angle_count:
                idx = cfg.pano_indices
                if len(idx) != cfg.angle_count:
                    idx = list(range(cfg.angle_count))
                depths_list = [depths_list[i] for i in idx]
            selected = depths_list
            intrinsics_depth = depths_list[0]
        tsdf_cfg = tsdf_cost_cfg or TsdfCostMapConfig()
        intrinsics = _infer_intrinsics(intrinsics_depth, cfg.hfov_deg)
        tsdf = TsdfCostMap(
            cfg=tsdf_cfg, intrinsics=intrinsics, robot_width=getattr(tsdf_cfg, "robot_width", None)
        )
        self.tsdf = tsdf
        cost_map, meta = tsdf.build_from_pano_depths(
            selected,
            cam_pos=np.array(cam_pos, dtype=np.float32),
            base_rotation=base_rotation,
            angle_step_deg=cfg.angle_step_deg,
            angle_count=cfg.angle_count,
            angle_ccw=bool(cfg.angle_ccw),
        )
        return (cost_map, meta)

    def save_tsdf_png(
        self,
        cost_map: np.ndarray,
        meta: Dict[str, float],
        current_pos: Tuple[float, float, float],
        out_path: str,
        enabled: bool = False,
        traj_before: Optional[np.ndarray] = None,
        traj_after: Optional[np.ndarray] = None,
        traj_relative: bool = False,
    ):
        if self.tsdf is None:
            raise RuntimeError(
                "TsdfGuide.save_tsdf_png called before build_cost_map initialized TsdfCostMap"
            )
        return self.tsdf.save_tsdf_png(
            cost_map=cost_map,
            meta=meta,
            current_pos=current_pos,
            out_path=out_path,
            enabled=enabled,
            traj_before=traj_before,
            traj_after=traj_after,
            traj_relative=traj_relative,
        )
