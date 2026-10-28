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

## Code Structure

- `TWP/`: prior training, diffusion training, and trajectory prediction on precomputed features.
- `VLN-CE/`: navigator training, Habitat evaluation, and benchmark inference with TSDF guidance.
- `environment.yaml` and root launch scripts.

## Installation

Use Linux with an NVIDIA GPU. The environment uses Python 3.8, PyTorch 2.1.2 with CUDA 11.8, and [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) / [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) 0.1.7. See also the [ETPNav setup guide](https://github.com/MarSaKi/ETPNav#setup) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).

1. Clone the repository:

   ```bash
   git clone https://github.com/iLearn-Lab/MM26-BeyondWP.git
   cd MM26-BeyondWP
   ```

2. Create and activate the environment:

   ```bash
   conda env create -f environment.yaml
   conda activate twp-vlnce
   ```

   `environment.yaml` installs the headless Habitat-Sim 0.1.7 package and pins Habitat-Lab to the [0.1.7 source revision](https://github.com/facebookresearch/habitat-lab/tree/d6ed1c0a0e786f16f261de2beafe347f4186d0d8). The required Habitat Baselines modules are included in `VLN-CE/habitat_baselines/`.

3. Check the installation:

   ```bash
   python -c "import torch, habitat, habitat_sim; print(torch.__version__, habitat.__version__, torch.cuda.is_available())"
   ```

## Data Preparation

Follow the [BEVBert data setup instructions](https://github.com/MarSaKi/VLN-BEVBert#setup) to prepare the VLN-CE annotations and Matterport3D scenes. Organize data and weights as follows:

```text
data/
├── datasets/
│   ├── R2R_VLNCE_v1-2_preprocessed_BERTidx/
│   │   ├── train/
│   │   │   ├── train_bertidx.json.gz
│   │   │   └── train_dagger_bertidx.json.gz
│   │   ├── val_seen/val_seen_bertidx.json.gz
│   │   ├── val_unseen/val_unseen_bertidx.json.gz
│   │   └── test/test_bertidx.json.gz
│   └── R2R_VLNCE_v1-2_preprocessed/
│       └── {split}/{split}_gt.json.gz
├── scene_datasets/
│   └── mp3d/{scene}/{scene}.glb
├── weights/
│   ├── twp.pth
│   ├── navigator.pth
│   ├── pretrained.pt
│   ├── depth.pth
│   ├── dinov3.pth
│   ├── resnet50.pth
│   ├── waypoint.pth
│   ├── siglip/
│   ├── bert-base-uncased/
│   └── xlm-roberta-base/
└── twp/
    ├── annotations/
    │   ├── train.json
    │   ├── val_unseen.json
    │   └── hm3d_train.json
    ├── mp3d/rgbd_fov90_d_1_25_fts/
    │   └── {split}/{scan}/{scan}_{node_id}_mp3d_imgs.pkl
    └── hm3d/rgbd_fov90_d_1_25_fts/hm3d/
        └── {scan}/{scan}_{node_id}_hfov90_hm3d_imgs.pkl
```

## Model Weights

| File or directory | Model | Source repository |
|---|---|---|
| `twp.pth` | Trajectory Waypoint Predictor (TWP) | This work |
| `navigator.pth` | Navigator | This work |
| `pretrained.pt` | Navigator pretraining | [BEVBert](https://github.com/MarSaKi/VLN-BEVBert) |
| `depth.pth` | DD-PPO depth encoder | [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo) |
| `dinov3.pth` | DINOv3 | [DINOv3](https://github.com/facebookresearch/dinov3) |
| `resnet50.pth` | ResNet-50 | [Torchvision](https://github.com/pytorch/vision) |
| `waypoint.pth` | Candidate Waypoint Predictor (CWP) | [Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN) |
| `siglip/` | SigLIP 2 | [big_vision](https://github.com/google-research/big_vision) |
| `bert-base-uncased/` | BERT | [BERT](https://github.com/google-research/bert) |
| `xlm-roberta-base/` | XLM-RoBERTa | [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/xlmr) |

## Training, Evaluation, and Inference

Set the paths below to match your data directory and run commands from the repository root:

```bash
export DATA_ROOT=/absolute/path/to/data
export WEIGHTS_ROOT="$DATA_ROOT/weights"
export TWP_DATA_ROOT="$DATA_ROOT/twp"
export ANNOTATIONS="$TWP_DATA_ROOT/annotations/train.json"
export RUN_ROOT=/absolute/path/to/runs
```

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

Use `--episodes 1` to run one episode and `--help` for other options. Set `NPROC_PER_NODE` for distributed navigator training. Inference writes `predictions.json`.

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

