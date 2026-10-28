import os
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader, Subset
from vint_train.data.wp_TRM_via_cls_dataset import WpDataset, wp_collate
from vint_train.data.wp_TRM_via_cls_dataset_hm3d import Hm3d_WpDataset


def loader(root, annotations, batch_size=32, train=False, limit=None):
    os.environ["TWP_DATA_ROOT"] = str(Path(root).resolve())
    cfg = {
        "num_imgs": 12,
        "len_traj_pred": 12,
        "normalize": True,
        "learn_angle": False,
        "angle_nums": 120,
        "distance_nums": 12,
    }
    sets = []
    for filename in annotations:
        cls = Hm3d_WpDataset if "hm3d" in str(filename).lower() else WpDataset
        sets.append(cls([str(filename)], cfg, {"metric_waypoint_spacing": 0.25}, in_memory=False))
    dataset = ConcatDataset(sets)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=0, collate_fn=wp_collate
    )


def on_device(batch, device):
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    batch["img_fts"] = batch["img_fts"].reshape(-1, batch["img_fts"].shape[-1])
    batch["dep_fts"] = batch["dep_fts"].reshape(-1, *batch["dep_fts"].shape[-3:])
    return batch
