# Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation

<div align="center">

**ACM Multimedia (ACM MM), 2026**

Haoxiang Shi, Xiang Deng, Haoyu Zhang, Qiaohui Chu, Yaowei Wang, Liqiang Nie

[![arXiv](https://img.shields.io/badge/arXiv-2606.07244-b31b1b.svg)](https://arxiv.org/abs/2606.07244)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2606.07244)

</div>

## Overview

Vision-Language Navigation in Continuous Environments (VLN-CE) requires an agent to follow natural-language instructions while moving safely through realistic 3D environments. Most existing systems use isolated waypoints: a waypoint predictor proposes a target, a navigator selects one, and a low-level controller decides how to reach it. This separation can produce physically unreachable targets and inconsistencies between planning and execution.

We introduce **Trajectory Waypoint**, a trajectory-centric navigation paradigm in which every candidate waypoint is grounded in a continuous, executable trajectory. The framework contains two core components:

- **Trajectory Waypoint Predictor (TWP):** generates diverse trajectory candidates with an environment-guided diffusion policy. Inference-time TSDF guidance pushes trajectories away from obstacles, while adaptive truncation supports variable action horizons.
- **Trajectory-Enhanced Navigator (TEN):** grounds each candidate trajectory in a topo-metric hybrid map and selects the trajectory that best aligns with the navigation instruction. The selected path is executed directly, coupling high-level semantic planning with low-level control.

## Highlights

- Reframes the atomic navigation unit from an isolated waypoint to an executable trajectory candidate.
- Uses TSDF-guided diffusion to improve geometric feasibility and collision avoidance.
- Encodes path geometry and visual semantics in a trajectory-enhanced topo-metric map.
- Achieves strong results on the VLN-CE/R2R-CE benchmark, including **60.3 SR** and **51.4 SPL** on the Val-Unseen split.

## Abstract

Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to follow natural-language instructions while navigating in real-world-like environments. Most VLN-CE approaches adopt a three-stage framework: a waypoint predictor proposes navigable waypoints, a navigator selects the best waypoint, and a low-level controller executes the movement. However, this decoupled paradigm often leads to unreachable waypoints or inconsistencies between planning and control. Instead of predicting isolated waypoints, we introduce a novel paradigm called Trajectory Waypoint, which grounds each candidate waypoint in an executable trajectory. We design a Trajectory Waypoint Predictor formulated as a TSDF-guided diffusion policy, which steers trajectory generation away from obstacles and improves the reachability of predicted waypoints. We further propose a Trajectory-Enhanced Navigator that injects the associated trajectory into planning, enabling consistency between high-level semantic decisions and low-level execution. Extensive experiments on the VLN-CE benchmark demonstrate the effectiveness of the proposed trajectory-centric paradigm.

## Main Results

### Trajectory Waypoint Prediction on VLN-CE Val-Unseen

| Model | &#124;Δ&#124; | %Open ↑ | $d_c$ ↓ | $d_h$ ↓ |
|---|---:|---:|---:|---:|
| Baseline | 1.37 | 80.18 | 1.08 | 2.16 |
| U-Net | 1.21 | 52.54 | 1.01 | 2.00 |
| RecBERT | 1.40 | 79.86 | 1.07 | 2.00 |
| ETPNav | 1.39 | 84.05 | 1.04 | 2.01 |
| SmartWay | 1.41 | 87.26 | 1.03 | 1.96 |
| **TWP (Ours)** | 1.47 | **95.84** | **0.54** | **1.95** |

### R2R-CE Navigation

| Split | NE ↓ | OSR ↑ | SR ↑ | SPL ↑ |
|---|---:|---:|---:|---:|
| Val-Seen | **3.75** | **74.6** | **68.8** | **60.2** |
| Val-Unseen | **4.57** | **68.1** | **60.3** | **51.4** |

The values above are the results reported in the paper.

## Release Scope

This release contains a minimal implementation of TWP and VLN-CE navigation:

- `TWP/`: prior training, diffusion training, and trajectory prediction on precomputed features.
- `VLN-CE/`: navigator training, Habitat evaluation, and benchmark inference with TSDF guidance.
- `environment.yaml` and root launch scripts.

Datasets, scenes, model weights, experiment logs, and generated outputs are stored externally and are not included. The navigation checkpoint selected for this release is iteration 14800 from the TSDF experiment (historical Val-Unseen SR 60.30%, SPL 52.46%). This experiment record is distinct from the paper results above.

## Environment Installation

Use Linux with an NVIDIA GPU and a CUDA-compatible driver:

```bash
conda env create -f environment.yaml
conda activate twp-vlnce
```

The code was smoke-tested in the existing Python 3.8.19 / PyTorch 2.1.2 + CUDA 11.8 / Habitat 0.1.7 environment. A fresh installation from the environment file has not yet been validated.

## Data Preparation

Keep all inputs outside this repository. VLN-CE expects the following paths under `DATA_ROOT`:

```text
scene_datasets/
datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz
datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/train/train_dagger_bertidx.json.gz
datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz
```

TWP consumes prepared annotation files and RGB/depth feature pickles. The loaders specify the annotation fields and feature layout in `TWP/vint_train/data/wp_TRM_via_cls_dataset.py` and `wp_TRM_via_cls_dataset_hm3d.py`. Feature extraction and dataset download scripts are not included.

## Training, Evaluation, and Inference

Set the variables below to absolute paths for your prepared inputs, checkpoints, and an external output directory. Run these commands from the repository root.

TWP prior training, diffusion training, and prediction:

```bash
bash train_twp.sh --stage prior --data-root "$TWP_DATA_ROOT" \
  --annotations "$ANNOTATIONS" --output-dir "$RUN_ROOT/prior"
bash train_twp.sh --stage diffusion --data-root "$TWP_DATA_ROOT" \
  --annotations "$ANNOTATIONS" --init-weights "$RUN_ROOT/prior/latest.pth" \
  --output-dir "$RUN_ROOT/diffusion"
bash predict_twp.sh --data-root "$TWP_DATA_ROOT" --annotations "$ANNOTATIONS" \
  --weights "$WEIGHTS_ROOT/twp.pth" --output "$RUN_ROOT/twp_predictions.pt"
```

VLN-CE navigator training, evaluation, and benchmark inference:

```bash
bash train_vlnce.sh --data-root "$DATA_ROOT" --weights-root "$WEIGHTS_ROOT" \
  --output-dir "$RUN_ROOT/train"
bash eval_vlnce.sh --data-root "$DATA_ROOT" --weights-root "$WEIGHTS_ROOT" \
  --output-dir "$RUN_ROOT/eval" --split val_unseen
bash infer_vlnce.sh --data-root "$DATA_ROOT" --weights-root "$WEIGHTS_ROOT" \
  --output-dir "$RUN_ROOT/inference" --split test
```

Use `--episodes 1` for a short evaluation/inference check and `--help` for other options. Set `NPROC_PER_NODE` for distributed navigator training. Benchmark inference writes `predictions.json`; this release does not provide a physical robot deployment interface.

Validation covered TWP prediction, both TWP training stages, a navigator training iteration, and one-episode navigation evaluation and inference. The complete 1839-episode benchmark has not been rerun on the cleaned code.

## Model Weights

Checkpoint downloads are not yet published. Running the commands above requires the externally prepared assets:

- `twp.pth`: the effective TWP state extracted from the selected navigator checkpoint.
- `navigator.pth`: the selected iteration-14800 navigator state.
- `depth.pth`, `dinov3.pth`, `resnet50.pth`, and `waypoint.pth`: frozen visual encoders and waypoint prior.
- `siglip/`, `bert-base-uncased/`, and `xlm-roberta-base/`: local model/processor/tokenizer assets expected by the loaders.
- `pretrained.pt`: navigator initialization used for training.

Keep the selected navigator and TWP checkpoints paired. Data and pretrained models remain subject to their respective upstream access and license terms.

## Citation

If you find this work useful, please cite:

```bibtex
@article{shi2026beyond,
  title   = {Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm for Vision-Language Navigation},
  author  = {Shi, Haoxiang and Deng, Xiang and Zhang, Haoyu and Chu, Qiaohui and Wang, Yaowei and Nie, Liqiang},
  journal = {arXiv preprint arXiv:2606.07244},
  year    = {2026}
}
```

The BibTeX entry will be updated with the official ACM MM proceedings metadata and DOI when they become available.

## License

Bundled third-party components retain their upstream terms. Their license texts are preserved in the corresponding package's `__license__` value, including the MIT licenses for ViNT, diffusion_policy, and Habitat Baselines, and the DINOv3 License for `TWP/dinov3`. A license for project-authored code has not yet been specified. Please refer to the [arXiv record](https://arxiv.org/abs/2606.07244) for the paper's distribution information.

## Contact

For questions about the paper, please open an issue in this repository or contact Haoxiang Shi at [Shihaoxiang1999@gmail.com](mailto:Shihaoxiang1999@gmail.com).

