import json
import jsonlines
import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy
import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import get_checkpoint_id, poll_checkpoint_folder
from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *
from habitat_extensions.measures import NDTW
from fastdtw import fastdtw
from ..utils import get_camera_orientations12
from ..utils import length2mask, dir_angle_feature, dir_angle_feature_with_ele

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    pass


class BaseVLNCETrainer(BaseILTrainer):
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def collect_infer_traj(self):
        from habitat_extensions.task import ALL_ROLES_MASK, RxRVLNCEDatasetV1

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        if "rxr" in self.config.BASE_TASK_CONFIG_PATH:
            if "{role}" in self.config.IL.RECOLLECT_TRAINER.gt_file:
                ep_data = {}
                for role in RxRVLNCEDatasetV1.annotation_roles:
                    if (
                        ALL_ROLES_MASK not in self.config.TASK_CONFIG.DATASET.ROLES
                        and role not in self.config.TASK_CONFIG.DATASET.ROLES
                    ):
                        continue
                    with gzip.open(
                        self.config.TASK_CONFIG.DATASET.DATA_PATH.format(split=split, role=role),
                        "rt",
                    ) as f:
                        ep_data.update(json.load(f))
            else:
                with gzip.open(self.config.TASK_CONFIG.DATASET.DATA_PATH.format(split=split)) as f:
                    ep_data = json.load(f)
        else:
            with gzip.open(self.config.TASK_CONFIG.DATASET.DATA_PATH.format(split=split)) as f:
                ep_data = json.load(f)
        ep_ids = [x["episode_id"] for x in ep_data["episodes"]]
        ep_ids = ep_ids[self.config.local_rank :: self.config.GPU_NUMBERS]
        return ep_ids

    def collect_val_traj(self):
        from habitat_extensions.task import ALL_ROLES_MASK, RxRVLNCEDatasetV1

        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        if "rxr" in self.config.BASE_TASK_CONFIG_PATH:
            if "{role}" in self.config.IL.RECOLLECT_TRAINER.gt_file:
                gt_data = {}
                for role in RxRVLNCEDatasetV1.annotation_roles:
                    if (
                        ALL_ROLES_MASK not in self.config.TASK_CONFIG.DATASET.ROLES
                        and role not in self.config.TASK_CONFIG.DATASET.ROLES
                    ):
                        continue
                    with gzip.open(
                        self.config.IL.RECOLLECT_TRAINER.gt_file.format(split=split, role=role),
                        "rt",
                    ) as f:
                        gt_data.update(json.load(f))
            else:
                with gzip.open(self.config.IL.RECOLLECT_TRAINER.gt_path.format(split=split)) as f:
                    gt_data = json.load(f)
        else:
            with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)) as f:
                gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank :: self.config.GPU_NUMBERS]
        return trajectories

    def eval(self):
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"
        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank
        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ["POSITION", "STEPS_TAKEN", "COLLISIONS"]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        if "HIGHTOLOW" in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index("HIGHTOLOW")
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = "HIGHTOLOWEVAL"
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend="nccl", init_method="env://")
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_val_traj()
        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            if os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                self._eval_checkpoint(
                    self.config.EVAL.CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=self.get_ckpt_id(self.config.EVAL.CKPT_PATH_DIR),
                )
            else:
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL.CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)
                    if self.local_rank < 1:
                        logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=self.get_ckpt_id(current_ckpt),
                    )

    def get_ckpt_id(self, path):
        import re

        match = re.search("iter(\\d+)", str(path))
        return int(match.group(1)) // self.config.IL.log_every - 1 if match else 0
