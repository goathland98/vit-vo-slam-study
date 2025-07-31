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


