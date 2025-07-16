# vit-vo-slam-study
The application of ViT to Visual Odometry 

2025/07/14 

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
  - [SLAM Survey Paper](https://onlinelibrary.wiley.com/journal/14779730)
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

## Next Steps

- [ ] MobileViT, TinyViT, SmallViT 모델을 TSFormer-VO에 적용해 성능 비교 실험
- [ ] K80에서 batch size, mixed precision 최적화 실험
- [ ] Attention map, temporal heatmap 시각화 추가
- [ ] Edge device (Jetson, Raspberry Pi) 대상 실험 설계
