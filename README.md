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


