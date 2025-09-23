# vit-vo-slam-study
The application of ViT to Visual Odometry 

# Research Log 

## 2025/07/14

- **SmallViT, TinyViT를 TSFormer-VO에 어떻게 적용할 수 있는지 학습**
  - Tiny, Small, Base ViT의 차이와 모델 파라미터 구성 이해
  - Patch size, embed dimension (edim), heads, depth가 VO 성능 및 속도에 미치는 영향 정리

- **PocketViT 연구**
  - 경량화 Vision Transformer 구성 방식 학습
  - 토큰 감소 (token reduction), 경량 attention, distillation 적용 방식 정리
  - [PocketViT GitHub](https://github.com/prabhxyz/pocket-vit)
  - [PocketViT Blog](https://medium.com/@prabhs./compressing-a-large-vit-into-a-5m-parameter-tiny-model-that-still-reaches-strong-accuracy-on-2f01ec93fd9d)

- **ViT-VO 논문/자료 연구**
  - ViT 기반 Visual Odometry 아키텍처, attention 방식, temporal modeling 방식 정리

- **Tesla K80 GPU 학습**
  - K80 환경에서 PyTorch, TensorFlow 실행 시 주의점 정리
  - Mixed precision, batch size 조정으로 최적화 가능성 확인


## 2025/07/15 - 2026/07/16

- **SLAM & Visual Odometry (VO) 최신 자료 정리**
  - [Awesome Transformer-based SLAM](https://kwanwaipang.github.io/Awesome-Transformer-based-SLAM/)
  - [EINet GitHub (Edge-aware Inertial Network)](https://github.com/BIT-XJY/EINet)
  - [VO KITTI GitHub (Visual Odometry on KITTI)](https://github.com/241abhishek/visual-odometry-KITTI)
  - [Google Vision Transformer](https://github.com/google-research/vision_transformer)
  - [LiDAR-SLAM Python Kitware](https://github.com/Kitware/pyLiDAR-SLAM/blob/master/slam/dataset/kitti_dataset.py)
  - [SLAM + ViT GitHub 예제](https://github.com/MisterEkole/slam_with_vit)
  - [SLAM/VO 강의 YouTube](https://www.youtube.com/playlist?list=PLrHDCRerOaI9HfgZDbiEncG5dx7S3Nz6X)

- **TSFormer-VO 모델 구성 파악**
  - 모델 구성 표:

    | 모델 타입     | dim | heads | patch_size | depth |
    | --------- | --- | ----- | ----------- | ----- |
    | Tiny ViT  | 192 | 3     | 16          | 12    |
    | Small ViT | 384 | 6     | 16          | 12    |
    | Base ViT  | 768 | 12    | 16          | 12    |

- **Edge Device에서의 MobileViT 방향성 연구 필요**
  - MobileViT: local (CNN) + global (ViT) 융합 구조
  - On-device real-time VO/SLAM 응용 가능성 분석 예정

---

## 2025/07/17
### Prepared Presentation Draft

- Completed slides for:
  - Introduction to SLAM & Visual Odometry (VO)
  - Problem statement & motivation
- Prepared diagrams:
  - SLAM pipeline inputs (camera, LiDAR, IMU)
  - Attention map examples (temporal & spatial)

---
### Summary of Code Changes

We updated `train_trio.py` with the following major improvements:

1️. **Automatic Mixed Precision (AMP)**
- Integrated `torch.cuda.amp` to enable mixed-precision training.
- Benefits:
  - Reduces GPU memory usage (~30-50% savings).
  - Speeds up training on compatible GPUs.
- Uses `GradScaler()` for stable backward pass.

2️. **Sequential Training for All Models**
- The script now automatically trains:
  - `small-ViT` → `tiny-ViT` → `base-ViT`
- Runs in a single execution.
- Saves checkpoints separately under:

`checkpoints/Exp4/small_vit/
checkpoints/Exp4/tiny_vit/
checkpoints/Exp4/base_vit/`



3️. **Training & Validation Loss Curve Plotting**
- After each model finishes training:
- Saves a `loss_curve.png` showing training & validation loss over epochs.
- Location:

`checkpoints/Exp4/{model_size}_vit/loss_curve.png`
## Next Steps

- Add slides: architecture details, challenges, and research directions
- Integrate lightweight transformers (MobileViT, SmallViT)
- Explore token pruning, sparse attention, weight sharing
- Evaluate accuracy vs. speed on KITTI dataset
- Plan benchmark tests on edge devices
##  2025/07/30 – Daily Update

## Inference & Visualization Pipeline Summary

---

###  `predict_poses_trio.py` – Inference Script

 Purpose:
- Run each trained ViT variant (`small`, `tiny`, `base`) on KITTI sequences.
- Save raw 6-DoF relative pose predictions to `.npy` format.

 Inputs:
- KITTI images: `datasets/sequences_jpg/<sequence>/`
- Model checkpoints:
```
checkpoints/Exp4/{small_vit,tiny_vit,base_vit}/
├── args.pkl
└── checkpoint_best.pth
```

 Outputs:
- For each model and sequence:
```
checkpoints/Exp4/{model_size}vit/checkpoint_best/
└── pred_poses<sequence>.npy
```
(Shape: `[N_clips, window_size–1, 6]`)

 Key Algorithm Steps:
1. Loop over models: `["small", "tiny", "base"]`
2. Load config from `args.pkl` and model weights
3. For each sequence:
 - Build `KITTI` dataset loader
 - Run inference (batch size = 1)
 - Save raw relative poses as `.npy` file

---

### `plot_results_trio.py` – Post-processing & Visualization

Purpose:
- Convert raw `.npy` pose predictions into full trajectory
- Save absolute camera pose files & trajectory plots vs. KITTI ground truth

Inputs:
- `.npy` outputs from `predict_poses_trio.py`
- KITTI ground-truth poses:
```
datasets/poses/<sequence>.txt
```
- Model hyperparameters (`args.pkl`)

 Outputs:
- Absolute camera poses:
```
checkpoints/Exp4/{model_size}_vit/checkpoint_best/pred_poses/<sequence>.txt
```
(Each line = 12 float values of a 3×4 pose matrix)

- X–Z trajectory plots:
```
checkpoints/Exp4/{model_size}_vit/checkpoint_best/plots/<sequence>.png
```

Key Algorithm Steps:
1. Loop over models: `["small", "tiny", "base"]`
2. For each sequence:
 - Load `.npy` predictions and post-process into 6-DoF
 - Convert relative poses → 4×4 matrices → chained trajectory
 - Compare against GT using X-Z projection
 - Save both `.txt` files of poses and `.png` files of trajectory.
Mobilevit: [Unofficial implementation of Mobilevit](https://github.com/jaiwei98/mobile-vit-pytorch)
---

## 2025/07/31 

### Summary

This report documents training and inference results for the TSFormer-VO model family (Tiny, Small, Base ViT variants) on the KITTI dataset.

All three models were trained using a unified pipeline (`train_trio.py`) with AMP support and consistent hyperparameters. Their performance was then evaluated using `predict_poses_trio.py` and `plot_results_trio.py`.



###  Observations

- All three models **showed similar training dynamics** with decreasing loss trends.
- However, **trajectory quality diverged sharply**:
  - **Small ViT** exhibited major drift from the KITTI ground-truth path.
  - **Tiny and Base ViT** achieved similar and more accurate reconstructions.


###  Insights

- **Loss curves alone may be misleading**: Despite good training loss, the Small ViT model failed to produce reliable global trajectories.
- This suggests **generalization or architectural limitations** in that configuration.

###  Future Direction

We propose the following enhancements:

- **MobileViT Integration**:
  - Brings CNN inductive bias + global attention.
  - Suitable for edge-device deployment (e.g., drones, mobile SLAM).
- **Model Compression**:
  - Token pruning
  - Sparse attention
  - Weight sharing

These directions aim to improve trajectory robustness while keeping the model lightweight and fast for real-time applications.



##  2025/08/04 MobileViT + TSFormer for Visual Odometry on KITTI

This project implements a lightweight Transformer-based Visual Odometry (VO) pipeline by combining **MobileViT** as a spatial encoder and **TSFormer** as a temporal encoder. The system is evaluated on the **KITTI Odometry Dataset**.

---

###  Overview

- **Input**: Sequence of 3 RGB frames from KITTI
- **Backbone**: [MobileViT](https://github.com/apple/ml-cvnets) (lightweight ViT hybrid)
- **Temporal Modeling**: TSFormer-style transformer encoder
- **Output**: 6-DoF relative pose (translation + rotation)
- **Loss**: SE(3) regression loss (translation + rotation vector)

---

### Project Structure

```bash
.
├── data/
│   └── kitti/
│       ├── sequences/         # KITTI image sequences
│       └── poses/             # Ground truth poses
├── models/
│   ├── mobilevit.py           # MobileViT encoder
│   └── tsformer_vo.py         # TSFormer + regression head
├── datasets/
│   └── kitti_vo_dataset.py    # KITTI Dataset loader
├── train_vo.py                # Training script
├── eval_vo.py                 # Evaluation script
├── utils/
│   ├── losses.py              # SE(3) pose loss
│   ├── metrics.py             # ATE, RPE computation
│   └── vis.py                 # Trajectory visualization
├── checkpoints/
│   └── model_best.pth         # Saved model weights
└── README.md
````

---

### Setup

```bash
# Create virtual environment
conda create -n vo-transformer python=3.10
conda activate vo-transformer

# Install dependencies
pip install torch torchvision timm einops kornia opencv-python matplotlib pandas pyquaternion
```

---

### Dataset

Download KITTI VO dataset:

* [KITTI Odometry (Official)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

Expected structure:

```bash
data/kitti/
├── sequences/
│   ├── 00/
│   │   └── image_2/000000.png ...
└── poses/
    └── 00.txt
```

---

### Model Summary

**Architecture:**

```text
Input RGB Frames → MobileViT Encoder (per-frame)
                 → Stack features across time
                 → TSFormer Temporal Attention
                 → MLP Regression Head
                 → Output: [tx, ty, tz, rx, ry, rz]
```

---

### Training

```bash
python train_vo.py --data_root ./data/kitti \
                   --seqs 00 01 02 04 06 \
                   --model mobilevit_tsformer \
                   --epochs 100 \
                   --lr 1e-4
```

---

### Evaluation

```bash
python eval_vo.py --checkpoint ./checkpoints/model_best.pth \
                  --data_root ./data/kitti \
                  --seq 09
```

Metrics:

* ATE (Absolute Trajectory Error)
* RPE (Relative Pose Error)

---

### Visualization

```bash
# Predicted vs Ground Truth
python utils/vis.py --pred traj_pred.csv --gt traj_gt.csv
```

---

### Loss Function

```python
L = || t̂ - t ||² + β * || r̂ - r ||²
```

Where:

* `t` = translation vector (3D)
* `r` = rotation vector (axis-angle)
* `β` = scaling factor for balancing loss


### References

* [MobileViT](https://arxiv.org/abs/2110.02178)
* [TSFormer-VO](https://arxiv.org/abs/2305.06121)
* [KITTI Dataset](http://www.cvlibs.net/datasets/kitti)

## Update: 2025-08-06  MobileViT for Visual Odometry

- Added a lightweight `MobileViT_VO` backbone for feature extraction in visual odometry tasks.
- Replaced ImageNet classifier head with `[B, C, H, W]` feature map output suitable for TSFormer-VO integration.
- Included `visualize_feature_map()` utility to inspect spatial feature activations.
- Default input: 256×256 RGB images.
- Tested with dummy input and ready for real image loading.

### Example Usage
```python
from mobilevit_vo import mobilevit_xxs_vo, visualize_feature_map
import torch

# Create model
model = mobilevit_xxs_vo()

# Test with random image
x = torch.randn(1, 3, 256, 256)
features = model(x)
print(features.shape)  # → [1, C, H, W]

# Visualize first 8 channels
visualize_feature_map(features, title=\"Feature Map (XXS)\")
```
## Daily Update: 2025/08/13

### Summary

Today we completed a full rewrite of the hybrid **MobileViT + Transformer-based** visual odometry model. Although the file was originally named `mobilevit_timesformer.py`, it **does not use the TimeSformer** implementation from the ViT repo. Instead, it uses **PyTorch’s built-in `nn.TransformerEncoder`** for temporal modeling across video frames.

---

### Files Updated

#### 1. `mobilevit_timesformer.py`

**Rewritten to:**
- Use a **MobileViT** variant (`XXS`, `XS`, or `S`) as the *spatial encoder*.
- Extract spatial features frame-by-frame and aggregate them into a temporal sequence.
- Project per-frame features (`[B, T, backbone_dim]`) to a transformer embedding dimension via a `Linear` layer.
- Feed the sequence into a `TransformerEncoder` for **temporal attention modeling**.
- Use a final MLP head (`output_head`) for pose regression (`[B, dim] → [B, 6 × (T−1)]`).

>  Note: TimeSformer was **not** used in this implementation. The name remains for compatibility.

---

#### 2. `build_model.py`


**Key changes:**
- Separated configuration dictionaries into `spatial_model_params` (MobileViT) and `transformer_params` (TransformerEncoder).
- Replaced `timesformer_params` with `transformer_params` to match the revised module.
- Updated model instantiation to use the new `MobileViT_TimeSformer` class.

---

#### 3. `train.py`

**Fixes applied:**
- Updated the model loading code to only expect a single `model` return from `build_model(...)`.
- Fixed argument passing to match new architecture.
- Confirmed compatibility with KITTI dataloader and training loop.

---

###  Model Architecture Overview

```text
Input: [B, C, T, H, W]
   │
   ▼
MobileViT Backbone (per frame): [B, C, H, W] → [B, D]
   │
   ▼
Temporal stacking: [B, T, D]
   │
   ▼
Linear projection: [B, T, D] → [B, T, dim]
   │
   ▼
TransformerEncoder:
    - Input: [T, B, dim]
    - Captures temporal dependencies across frames
   │
   ▼
Mean pooling: [T, B, dim] → [B, dim]
   │
   ▼
Output head (pose regression): [B, dim] → [B, num_classes]
```
## 2025-08-20 — Daily Update: Milestone A (Foundations)

**Objective:** Set up repo skeleton, core geometry utilities, and KITTI I/O with unit tests.

### What we did
- **Repo scaffold**
  - Created: `configs/`, `models/`, `utils/`, `tests/`, plus stubs `train.py`, `evaluate.py`
  - Added `__init__.py` to `utils/` and `models/` (fix import paths)
- **Geometry (`utils/geometry.py`)**
  - Implemented: `skew`, `vee`, `so3_exp`, `so3_log`, **batch-safe** `se3_exp` (stable left-Jacobian w/ small-angle series), `se3_inv`, `se3_mul`, `project_points`, `bilinear_sampler`
  - Fixed broadcasting bug in `se3_exp` (correct scalar expansion + identity expansion)
- **KITTI I/O (`utils/kitti_io.py`)**
  - Parsed `calib.txt` (P2/P3) → `K_left`, `K_right`, **baseline** `B = -P3[0,3]/fx`
  - Helpers to list `image_2/` & `image_3/`, plus CLI sanity print
  - Linked dataset at `./data` (symlink to KITTI odometry root)
- **Config**
  - Added `configs/kitti_monocular.yaml` (placeholders for `data_root`, clip length, etc.)
- **Tests**
  - `tests/test_geometry.py`: SO(3)/SE(3) round-trip, compose·inverse, grid-sampler identity
  - `tests/test_kitti_io.py`: synthetic checks for `K` and baseline extraction
  - Resolved `ModuleNotFoundError: utils` via package inits and repo-root pytest

#### How we verified

- run unit tests (disable unrelated plugins, e.g., ROS)
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
- real-data sanity (expect K matrix, B ≈ 0.54 m, and image counts)
```
python -m utils.kitti_io --seq_root ./data/sequences/09
```
# Daily Update 2025-08-25 

## TL;DR

* Added a **simple MobileViT + Temporal Transformer** VO model (`MVitTSVO`) that takes a 2-frame clip and predicts the 6-DoF relative pose.
* Wrote a **minimal training script** (`train_core.py`) that plugs into existing KITTI pair loader and uses our established **geodesic SO(3) + Smooth-L1** losses.
* Kept things **plain & stable**: no AMP/TF32/compile tricks—just clean PyTorch.

---

## What changed

###  New

* `models/mvit_tsvo.py`

  * `MobileViTSpatial` (per-frame features via `timm` MobileViT, global pooled).
  * `TemporalTransformer` (tiny `nn.TransformerEncoder` over time).
  * `PairPoseHead` (predicts 6-DoF per adjacent pair).
  * `MVitTSVO` wrapper returning `[B, (T-1)*6]` (for T=2 → `[B,6]`).

###  Updated

* `train_core.py`

  * Uses `KittiMonocularPairs` to get `(img_i, img_j, T_ij)`.
  * Packs each pair into a **2-frame clip** and feeds `MVitTSVO`.
  * Computes losses:

    * **Rotation**: geodesic SO(3) (radians).
    * **Translation**: Smooth-L1.
  * Evaluates Δ=1 **RPE** (m, deg).
  * Checkpointing:

    * Best: `ckpts/tsvo_best.pt`
    * Periodic every 20 epochs: `ckpts/tsvo_epXXX.pt`
    * Last: `ckpts/tsvo_last.pt`

---

## Config updates (example)

Add this (or keep your existing YAML and append `model:`):

```yaml
data_root: ./data
sequences_train: [0,1,2,3,4,5,6,7,8]
sequences_val: [9]
img_short_side: 224
frame_delta: 1
batch_size: 4
num_workers: 2

epochs: 20
lr: 1e-4
weight_decay: 0.05
grad_clip: 1.0
loss: { rot_w: 1.0, trans_w: 1.0 }

model:
  mvit_variant: mobilevit_xs     # mobilevit_xxs | mobilevit_xs | mobilevit_s
  mvit_pretrained: false         # set true if you want ImageNet init
  t_heads: 4
  t_layers: 2
  t_dropout: 0.1
  head_hidden: 512
```

---

### How to run

```bash
# install deps (if needed)
pip install timm tqdm tensorboard einops

# train
python train_core.py --config configs/kitti_monocular.yaml
```

**Outputs**

* Checkpoints → `ckpts/tsvo_best.pt`, `ckpts/tsvo_last.pt`, `ckpts/tsvo_ep020.pt` …
* Console logs print per-epoch train/val loss + RPE.

---

### Notes & decisions

* We deliberately **skipped speed features** (AMP/TF32/compile) to keep the baseline simple and debuggable.
* Losses and evaluation match Foundation-B conventions, so later comparisons are apples-to-apples.
* The model supports longer clips (`T>2`) if/when we switch the dataset to provide windows.

---

### Next steps

*  (Optional) turn on `mvit_pretrained: true` to test if ImageNet init stabilizes convergence.
*  Add TensorBoard & CSV logging (simple to wire in, if desired).
*  Quick qualitative: run on a short sequence and visualize integrated trajectory for sanity.
*  Explore longer temporal windows (`window_size>2`) once training is stable on pairs.


# Efficient Vision Transformer Architecture for Visual Odometry in SLAM Applications on Edge Devices

## Overview

This project proposes the development, implementation, and evaluation of lightweight Vision Transformer (ViT) architectures specifically designed for **monocular Visual Odometry (VO)** in **SLAM pipelines**, with a focus on real-time operation on **embedded and edge devices**.


## Motivation and Problem Statement

- Classical VO/SLAM systems:
  - Depend on feature-based or CNN-based methods.
  - Struggle in low-texture, dynamic environments.
  - Are not optimized for edge deployment.

- **Transformers**:
  - Provide global spatial + temporal attention.
  - Achieve superior context modeling but at high computational cost.

- **Research Goal**:
  - Bring transformer-based VO models to edge devices via lightweight, efficient architectures.


## Background

- **What is SLAM?**
  - Simultaneous Localization and Mapping: estimate robot’s position + build a map.

- **What is Visual Odometry (VO)?**
  - Frame-to-frame pose estimation from visual input.

- **Why Vision Transformers?**
  - Use self-attention to capture global features.
  - Outperform CNNs in various vision tasks.

---

## TSFormer-VO Overview

- TimeSformer backbone.
- Divided temporal + spatial attention.
- Applied to KITTI dataset for monocular VO.
- Current bottleneck:
  - Computational load.
  - Edge-device unsuitability.
  - Generalization to non-KITTI datasets.

---

## Challenges

- High compute cost (latency, memory).
- Accuracy gaps under dynamic or challenging conditions.
- Poor generalization across diverse datasets.
- Need for explainability in decision-making.

---

## Proposed Research Directions

- Apply **MobileViT, SmallViT, TinyViT** to replace base ViT backbone.
- Explore:
  - Token pruning.
  - Sparse attention.
  - Weight sharing.
  - Hybrid CNN-Transformer architectures.
- Add explainable attention visualization tools.
- Optimize for:
  - Quantization.
  - Distillation.
  - Edge-aware deployment.

---

## Methodology

- **Model Design**:
  - Modify transformer architecture for efficiency.
- **Training Strategy**:
  - Multi-task learning.
  - Regularization.
  - Pretraining on large-scale datasets.
- **Evaluation Metrics**:
  - Odometry accuracy (ATE, RPE).
  - Latency, power, memory footprint.
- **Benchmarking**:
  - Edge devices: Jetson, Snapdragon, Apple Neural Engine.

---

## Expected Contributions

- Lightweight, efficient transformer-based VO models.
- Benchmarking suite for VO-SLAM on edge hardware.
- Visual analytics tools (temporal/spatial attention).
- Design guidelines for real-time deployment.

---

## Preliminary Plans

- Baseline: TSFormer-VO on KITTI.
- Next steps:
  - Integrate MobileViT, SmallViT.
  - Extend experiments to cross-domain datasets.
  - Conduct edge-device deployment tests.

---

# Efficient Vision Transformer Architecture for Visual Odometry in SLAM Applications on Edge Devices

## 📌 Introduction

Simultaneous Localization and Mapping (SLAM) is a foundational capability for autonomous navigation in robotics and intelligent vehicles.  
Visual Odometry (VO) estimates frame-to-frame motion using camera input and is a critical component within SLAM pipelines.

Recent advances in Vision Transformers (ViTs) show great potential for modeling global context and long-range dependencies, but their computational cost poses challenges for deployment on resource-constrained edge devices.

This project aims to design, implement, and evaluate **lightweight Vision Transformer architectures** tailored for real-time monocular VO on embedded and edge platforms.

---

## 📚 Related Work

- Classical VO-SLAM approaches (feature-based, CNN-based) and their limitations
- Vision Transformer architectures:
  - ViT, MobileViT, SmallViT, TinyViT, Swin Transformer
- Transformer applications in SLAM:
  - Temporal & spatial attention for motion estimation
  - SLAM with transformer backbones (e.g., TimeSformer-VO)

---

## 🔬 Background

- **Vision Transformer (ViT)**: uses self-attention over tokenized image patches
- **Self-attention types**:
  - Spatial attention → within-frame relationships
  - Temporal attention → across-frame dynamics
- **Advantages over CNNs**:
  - Global context modeling
  - Long-range dependency capture
- **TimeSformer-VO overview**:
  - Divided space-time attention backbone
  - Applied to KITTI dataset for monocular VO

---

## 🚩 Problem Definition

- High computational cost for real-time edge deployment
- Accuracy gap vs. ground truth, especially in dynamic scenes
- Dataset generalization challenges (e.g., KITTI → real-world, indoor, or urban environments)
- Lack of explainability in learned attention mechanisms

---

## 💡 Proposed Research Directions

- Apply **lightweight backbones**: MobileViT, SmallViT, TinyViT
- Explore model optimizations:
  - Token pruning
  - Sparse attention
  - Weight sharing
- Investigate **hybrid CNN-Transformer** architectures
- Optimize for edge deployment:
  - Quantization
  - Knowledge distillation
- Develop attention visualization tools

---

## ⚙️ Methodology

- **Model design**: architecture modifications & compression
- **Training strategy**:
  - Multi-task learning
  - Pretraining on large datasets
  - Regularization methods
- **Evaluation metrics**:
  - Accuracy (pose estimation error)
  - Latency and FPS (real-time performance)
  - Power consumption on edge devices (e.g., Jetson, Raspberry Pi, Snapdragon)
- **Benchmarking**:
  - KITTI dataset
  - Cross-domain datasets for generalization

---

## 🌟 Expected Contributions

✅ A novel lightweight VO-Transformer architecture  
✅ Open-source benchmarking suite & reproducible experiments  
✅ Visualization tools for interpreting temporal/spatial attention  
✅ Insights into real-time deployment of transformers for robotics

---

## 🧪 Preliminary Plans

- Baseline experiments with TimeSformer-VO on KITTI  
- Integration of MobileViT, SmallViT backbones  
- Planned cross-dataset validation and edge device profiling

---

## 📁 Resources

- **SLAM + Transformer Survey**: https://kwanwaipang.github.io/Awesome-Transformer-based-SLAM/  
- **Small-ViT-VO Article**: https://medium.com/@prabhs./compressing-a-large-vit-into-a-5m-parameter-tiny-model-that-still-reaches-strong-accuracy-on-2f01ec93fd9d  
- **Pocket-ViT GitHub**: https://github.com/prabhxyz/pocket-vit  
- **Visual Odometry KITTI GitHub**: https://github.com/241abhishek/visual-odometry-KITTI  
- **SLAM Starting Code**: https://github.com/MisterEkole/slam_with_vit  

---

## 🔭 Future Work

- Extend to multi-modal SLAM (e.g., visual-inertial, LiDAR-camera fusion)  
- Apply to embedded systems (e.g., Jetson Nano, Raspberry Pi, Apple Neural Engine)  
- Collaborate with robotics teams for real-world deployment

---

# 2025/09/05 Introduction Section - Peer Review Feedback Summary (Daily Update)

## I. Summary of Peer Feedback

- 기존 VO 문제의 흐름 설명이 부족함
- CNN 기반 VO의 구조 및 한계에 대한 설명 미비
- 기존 → CNN → ViT → Video Transformer 로 구성 흐름 필요
- context vector, global alignment 용어는 Introduction에서 Background로 이동 권고
- Grayscale vs RGB / Image vs Patch 구분 필요
- FLOP 수치 제시의 목적 명확화 필요
- Edge 디바이스 스펙 비교 출처 불명, 미시적 조건 타당성 재확인 필요
- MobileViT-VO 관련 설명 중 6-DoF fine-tuning metric의 한계 명확화 필요
- Evaluation 슬라이드는 Introduction에서 제거 권고
- 전체적으로 영어 표기 짤림/문법 오류 수정 필요

---

## II. Slide Restructuring Plan

### ✔️ 기술 흐름 시각화 구조
```
[Traditional VO]
   ↓
[CNN-based VO]
   ↓
[ViT-based VO]
   ↓
[Video Transformer for VO]
```

- 기존 방식의 한계 → CNN의 지역성 한계 → ViT의 무거움 → Temporal reasoning 필요
- Evaluation 파트는 Introduction이 아닌 실험 결과 파트로 분리
- Problem Statement 중복 내용 정리 및 축약 필요

---

## III. 용어 정리 및 시각자료 보완

### ✅ 용어 재배치
- `context vector`, `global alignment` → Background 섹션으로 이동
- MobileViT 및 6-DoF 출력 구조에 대한 간결한 시각화 필요

### ✅ Grayscale vs RGB 정리
| 항목 | Grayscale | RGB |
|------|-----------|-----|
| 채널 수 | 1 | 3 |
| 정보량 | 낮음 | 높음 |
| FLOP 영향 | 낮음 | 높음 |

### ✅ Image vs Patch
- ViT는 이미지를 **patch 단위**로 나누어 처리
- FLOP 계산은 patch 수와 embedding dimension에 비례

---

## IV. 수정 및 향후 작업 리스트

| 구분 | 수정/작업 항목 | 우선순위 |
|------|----------------|-----------|
| 슬라이드 | Problem Statement 슬라이드 간소화 | 🔴 높음 |
| 슬라이드 | 기술 흐름 정리된 다이어그램 추가 | 🟠 중간 |
| 설명 | FLOP 수치 제시 목적 및 비교 기준 명확화 | 🔴 높음 |
| 설명 | MobileViT-VO에서의 6-DoF metric 설명 추가 | 🟠 중간 |
| 시각자료 | Image vs Patch 비교 및 FLOP 관계 추가 | 🟢 낮음 |
| 용어 | 영어 표기 오류 수정 및 문장 다듬기 | 🔴 높음 |
| 출처 | Edge 디바이스 비교 조건의 출처 명시 | 🟠 중간 |

# Daily Update — 2025/09/22

## Highlights

* Finished **stereo migration** of TSFormer-VO (KITTI `image_0`/`image_1`).
* Added **stereo training pipeline** with speed-ups, CSV+plot logging, and optional episodic restarts.
* Implemented **inference + visualization**: predicts poses with the stereo model, saves CSVs, and plots **GT vs Pred** trajectories.

## Code Added / Changed

* **Model**

  * `build_model.py`: added `StereoTimeSformer` (Conv3D **2→3** adapter) + `args.stereo`/`args.input_channels`.
  * Kept `PatchEmbed` unchanged (still sees 3ch via adapter).
* **Dataset**

  * `datasets/kitti_stereo.py`: emits windows as **\[C=2, T, H, W]** (C0=Left, C1=Right), reuses GT delta logic, grayscale normalize `[0.5]`.
* **Training**

  * `train_stereo.py` (or `models/stereo/train_stereo_vo.py`):

    * AMP (`--amp`), `cudnn.benchmark`, tuned DataLoader, optional `torch.compile`.
    * CosineAnnealingWarmRestarts.
    * **CSV** (`training_log.csv`) + **plots** (`loss_curve.png`, `lr_curve.png`) auto-saved to `--checkpoint_path`.
    * **Optional** episodic **restart-from-best** (`--restart_from_best`, default OFF) with `--episode_len` (default 20).
* **Inference / Viz**

  * `predict_stereo_poses.py`: loads stereo model via `build_model`, runs per-sequence inference, **denormalizes** deltas, integrates to global poses, outputs:

    * `pred_vs_gt_<seq>.csv` (deltas + global XYZ for Pred/GT),
    * `pred_deltas_<seq>.npy`,
    * `traj_<seq>.png` (GT vs Pred in X-Z).

## How to Train (defaults: 100 epochs, **no restarts**)

```bash
# run from repo root
export PYTHONPATH=.
python train_stereo.py \
  --data_path data/sequences_jpg --gt_path data/poses \
  --sequences 00 02 08 09 \
  --epochs 100 --bsize 4 --num_workers 4 \
  --amp --pretrained_ViT
```

Enable episodic restart-from-best:

```bash
python train_stereo.py \
  --data_path data/sequences_jpg --gt_path data/poses \
  --sequences 00 02 08 09 \
  --epochs 100 --episode_len 20 --restart_from_best \
  --bsize 4 --num_workers 4
```

## How to Predict + Plot

```bash
export PYTHONPATH=.
python predict_stereo_poses.py \
  --checkpoint_path checkpoints/ExpStereo \
  --checkpoint_name checkpoint_best.pth \
  --sequences 01 03 04 05 06 07 10
```

Outputs per sequence in `<checkpoint_path>/<checkpoint_name_stem>/`:

* `pred_vs_gt_<seq>.csv`, `pred_deltas_<seq>.npy`, `traj_<seq>.png`

## Notes / Fixes

* If you see `ModuleNotFoundError: No module named 'models'`, either:

  * run script directly (`python train_stereo.py`), **or**
  * create package layout `models/stereo/__init__.py` and run `python -m models.stereo.train_stereo_vo` with `PYTHONPATH=.`.
* Default options when no flags are passed: **epochs=100**, **no restarts**, Adam `1e-4`, window size **2**, stereo enabled via adapter, CSV/plots always saved.

## Rationale

* Stereo provides **metric scale** and more stable VO.
* The **2→3 adapter** preserves ImageNet pretrained ViT weights with minimal code churn.
* Episodic **restart-from-best** is opt-in: helps escape bad drift; baseline remains unchanged by default.

## Next Steps

* Optional stereo **cost-volume**/correlation for stronger geometry.
* Optionally add left↔right **photometric loss** using KITTI `calib.txt` intrinsics and baseline.
* Compute dataset-specific grayscale stats for `image_0`/`image_1` to replace `[0.5]`.


## Evaluation & Profiling 2025/09/23

This repo includes two tools:

1. **KITTI Odometry Metrics** — translational error `t_rel` (%) and rotational error `r_rel` (deg/100m) computed over standard segment lengths.
2. **Cost Evaluation** — measures latency/FPS, CUDA memory usage, and (optionally) GPU power.

---

### 1) KITTI Odometry Metrics

**Script:** `tools/kitti_evaluate.py`  
**Input:** One or more CSVs from `predict_stereo_poses.py` (files named like `pred_vs_gt_<seq>.csv`).  
Each CSV must contain columns:

```

idx,pred\_rz,pred\_ry,pred\_rx,pred\_tx,pred\_ty,pred\_tz,pred\_px,pred\_py,pred\_pz,gt\_px,gt\_py,gt\_pz

````

**Usage (strict, no alignment):**
```
bash
python tools/kitti_evaluate.py \
  --csv checkpoints/ExpStereo/checkpoint_best/pred_vs_gt_03.csv \
       checkpoints/ExpStereo/checkpoint_best/pred_vs_gt_04.csv \
  --align none
````

**Options:**

* `--csv`: one or more CSV files to evaluate (can mix sequences).
* `--align`: `none` (default, **recommended for stereo**), `scale` (scale-only), or `se3` (full Procrustes).
* `--out`: optional path for the summary CSV (default: `kitti_eval_summary.csv` next to first input CSV).

**Outputs:**

* Pretty-printed table of segment-wise averages (100–800 m).
* `kitti_eval_summary.csv` with per-length means/std and overall means/std.

**Notes:**

* Use `--align none` to demonstrate true **metric scale** (stereo).
* `--align scale` or `--align se3` are useful for shape-only diagnostics or monocular baselines.

---

### 2) Cost Evaluation (Latency / FPS / Memory / Power)

**Script:** `tools/cost_eval.py`
**Input:** A trained checkpoint + a KITTI Stereo loader.
**What it does:** Runs several warmup and measured iterations on your model, reports:

* **Latency per batch** (ms)
* **Throughput** in **frames/s** (`frames = batch_size × window_size`)
* **Max CUDA memory** allocated/reserved
* **GPU power** (optional; requires NVIDIA + `pynvml`)

**Usage (AMP on):**

```
bash
python tools/cost_eval.py \
  --checkpoint_path checkpoints/ExpStereo \
  --checkpoint_name checkpoint_best.pth \
  --data_path data/sequences_jpg \
  --gt_path data/poses \
  --sequences 00 \
  --bsize 4 --num_workers 4 \
  --iters_measure 50 \
  --amp
```

**Optional power sampling (NVIDIA only):**

```bash
pip install pynvml
python tools/cost_eval.py ... --power
```

**Key Options:**

* `--checkpoint_path`: folder containing `args.pkl` and the checkpoint file.
* `--checkpoint_name`: checkpoint filename (e.g., `checkpoint_best.pth`).
* `--data_path`, `--gt_path`: override paths; otherwise read from saved `args.pkl`.
* `--sequences`: one or more sequences to iterate over for input.
* `--bsize`: batch size for profiling (**affects FPS/latency**).
* `--iters_warmup`: warmup iterations (default 10).
* `--iters_measure`: measured iterations (default 50).
* `--amp`: enable mixed precision (recommended on modern GPUs).
* `--power`: sample GPU power (requires `pynvml` and an NVIDIA GPU).
* `--out_csv`: output CSV path (default: `<checkpoint_path>/cost_eval.csv`).

**Outputs:**

* Console summary (latency/FPS/memory/power).
* CSV (default `cost_eval.csv`) with columns:

  ```
  bsize,window_size,amp,iters_measure,
  latency_ms_mean,latency_ms_std,
  fps_mean,fps_std,
  cuda_mem_alloc_mib,cuda_mem_reserved_mib,
  gpu_power_w_mean,gpu_power_w_std
  ```

**Notes:**

* FPS is computed as **frames/s**, where frames = `batch_size × window_size`.
  If you prefer “windows/s”, divide FPS by `window_size`.
* Enable `--amp` to profile realistic training/inference settings.
* For memory, both **allocated** and **reserved** are reported (MiB).


### Quick Tips

* Run `predict_stereo_poses.py` first to generate the per-sequence CSVs used by the KITTI evaluator.
* For **fair KITTI reporting** on stereo VO, use `--align none`.
* To compare against monocular baselines, try `--align scale` (common in papers) and report both.
* When changing `--bsize` or `--window_size`, re-run **cost\_eval**; these directly impact latency/FPS.


