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

## Method

### Trajectory Waypoint Predictor

TWP first predicts coarse navigation intentions and uses them as location priors. A conditional diffusion policy then generates continuous trajectory candidates. During denoising, a differentiable safety cost derived from a local Truncated Signed Distance Field (TSDF) map guides candidate paths away from obstacles. Finally, adaptive trajectory truncation converts fixed-horizon predictions into variable-length executable trajectories.

### Trajectory-Enhanced Navigator

TEN projects panoramic visual features into a local metric map and samples path-aligned visual features along every candidate trajectory. These sequential representations enrich both the topological graph and the metric map. An instruction-conditioned cross-modal navigator then scores the complete trajectory candidates and directly executes the selected path.

## Main Results

### Trajectory Waypoint Prediction on VLN-CE Val-Unseen

| Model | $|\Delta|$ | %Open ↑ | $d_c$ ↓ | $d_h$ ↓ |
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

This initial repository is a **paper-only project page**. It does not currently include source code or other executable artifacts.

| Artifact | Current status |
|---|---|
| Paper and project description | Available |
| Source code and configuration files | Not included in this release |
| Environment specification | To be provided with a future code release |
| Data-processing scripts | Not included in this release |
| Model checkpoints | Not included in this release |
| Training, evaluation, and inference commands | Not included in this release |

## Environment Installation

No software installation is required for this paper-only release. A reproducible environment specification will be added if and when the implementation is released.

## Data Preparation

The experiments are conducted in the Habitat simulator on VLN-CE/R2R-CE. TWP training uses Matterport3D (MP3D) together with Habitat-Matterport 3D (HM3D); the paper reports more than 120,000 waypoint-trajectory training pairs. TEN is first pre-trained on R2R and then fine-tuned online in Habitat with DAgger.

No dataset or data-processing pipeline is distributed in this paper-only release. Users should obtain the underlying datasets from their official providers and comply with the corresponding licenses and terms of use.

## Training, Evaluation, and Inference

Runnable commands are not provided because source code is outside the scope of this initial release. This section will be updated together with the implementation to avoid publishing incomplete or unverifiable instructions.

## Model Weights

No model weights are distributed in this paper-only release. Download instructions and model documentation will be added if checkpoints are released in the future.

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

This repository currently contains project documentation only. It does not grant a software, model, or dataset license. Please refer to the [arXiv record](https://arxiv.org/abs/2606.07244) for the paper's distribution information. Licensing terms will be updated if additional artifacts are released.

## Contact

For questions about the paper, please open an issue in this repository or contact Haoxiang Shi at [Shihaoxiang1999@gmail.com](mailto:Shihaoxiang1999@gmail.com).

