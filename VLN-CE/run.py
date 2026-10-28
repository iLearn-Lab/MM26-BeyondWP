import argparse
import copy
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import yaml
from configuration import TASK, EXPERIMENT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-type", choices=["train", "eval", "inference"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--weights-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--episodes", type=int, default=-1)
    p.add_argument("--iterations", type=int, default=16000)
    p.add_argument("--max-decisions", type=int, default=15)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--split")
    p.add_argument("--resume")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
    )
    a = p.parse_args()
    a.split = a.split or ("test" if a.run_type == "inference" else "val_unseen")
    data = Path(a.data_root).resolve()
    weights = Path(a.weights_root).resolve()
    output = Path(a.output_dir).resolve()
    if a.resume:
        a.resume = str(Path(a.resume).resolve())
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ACM_WEIGHTS_ROOT"] = str(weights)
    os.chdir(output)
    try:
        Path("data").symlink_to(data, target_is_directory=True)
    except FileExistsError:
        if Path("data").resolve() != data:
            raise RuntimeError("Output directory uses a different data root")
    import habitat_extensions
    import vlnce_baselines
    from habitat import logger
    from habitat_baselines.common.baseline_registry import baseline_registry
    from vlnce_baselines.config.default import get_config

    task = copy.deepcopy(TASK)
    exp = copy.deepcopy(EXPERIMENT)
    task["DATASET"]["SCENES_DIR"] = str(data / "scene_datasets")
    task["DATASET"]["DATA_PATH"] = str(
        data / "datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz"
    )
    task["ENVIRONMENT"]["MAX_EPISODE_STEPS"] = a.max_steps
    world = int(os.environ.get("WORLD_SIZE", 1))
    exp.update(
        GPU_NUMBERS=world,
        NUM_ENVIRONMENTS=a.num_envs,
        SIMULATOR_GPU_IDS=list(range(world)),
        TORCH_GPU_IDS=list(range(world)),
        local_rank=a.local_rank,
    )
    exp["MODEL"]["DEPTH_ENCODER"]["ddppo_checkpoint"] = str(weights / "depth.pth")
    exp["MODEL"]["pretrained_path"] = (
        str(weights / "pretrained.pt") if a.run_type == "train" else None
    )
    exp["IL"].update(
        iters=a.iterations,
        lr=1.5e-5,
        log_every=min(200, a.iterations),
        decay_interval=4000,
        load_from_ckpt=bool(a.resume),
        is_requeue=False,
        back_algo="teleport" if a.run_type == "train" else "control",
        waypoint_aug=a.run_type == "train",
        max_traj_len=a.max_decisions,
    )
    if a.resume:
        exp["IL"]["ckpt_to_load"] = a.resume
    if a.run_type == "train":
        task["DATASET"]["DATA_PATH"] = str(
            data
            / "datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_dagger_bertidx.json.gz"
        )
    exp["EVAL"].update(
        CKPT_PATH_DIR=str(weights / "navigator.pth"), SPLIT=a.split, EPISODE_COUNT=a.episodes
    )
    exp["INFERENCE"].update(
        CKPT_PATH=str(weights / "navigator.pth"),
        SPLIT=a.split,
        EPISODE_COUNT=a.episodes,
        PREDICTIONS_FILE=str(output / "predictions.json"),
    )
    for key, name in [
        ("TENSORBOARD_DIR", "tensorboard"),
        ("CHECKPOINT_FOLDER", "checkpoints"),
        ("RESULTS_DIR", "results"),
        ("VIDEO_DIR", "videos"),
    ]:
        exp[key] = str(output / name)
    with tempfile.TemporaryDirectory(dir=output) as temp:
        taskpath = Path(temp) / "task.yaml"
        exppath = Path(temp) / "experiment.yaml"
        taskpath.write_text(yaml.safe_dump(task))
        exp["BASE_TASK_CONFIG_PATH"] = str(taskpath)
        exppath.write_text(yaml.safe_dump(exp))
        cfg = get_config(str(exppath))
    logger.add_filehandler(str(output / "run.log"))
    random.seed(cfg.TASK_CONFIG.SEED)
    np.random.seed(cfg.TASK_CONFIG.SEED)
    torch.manual_seed(cfg.TASK_CONFIG.SEED)
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    trainer = baseline_registry.get_trainer(cfg.TRAINER_NAME)(cfg)
    if a.run_type == "train":
        trainer.train()
    elif a.run_type == "eval":
        trainer.eval()
    else:
        trainer.inference()


if __name__ == "__main__":
    main()
