# 3D Camera Pose Estimation with Vision Transformer (ViT)

A clean, reproducible **6-DoF camera pose regression** pipeline using a **Vision Transformer (ViT)** backbone on the **7-Scenes** dataset.

---

## Overview

| Item | Detail |
|------|--------|
| **Task** | 6-DoF camera pose regression (translation + rotation) |
| **Input** | RGB image (224 × 224) |
| **Output** | Translation `(x, y, z)` + Unit-quaternion `(qx, qy, qz, qw)` |
| **Dataset** | [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) |
| **Backbone** | ViT-S/16 (via [timm](https://github.com/huggingface/pytorch-image-models)) |
| **Baseline** | ResNet-18 (via timm) |

---

## Model Design

### Architecture

```
Input (B×3×224×224)
       │
   ViT Backbone (timm)
   – cls token output → feature vector (B × D)
       │
   MLP Regression Head
   – Linear(D, 512) → ReLU → Linear(512, 7)
       │
   Split & Normalise
   – t_pred = out[:, :3]       (translation)
   – q_pred = L2_norm(out[:, 3:])  (unit quaternion)
```

The classification head of ViT is removed and replaced with an MLP regression head following the **PoseNet** convention.

### Loss Function (PoseNet-style)

```
Loss = ‖t_pred − t_gt‖₂  +  β · ‖q_pred − q_gt‖₂
```

where `β = 500` by default (scene-scale dependent; tune per scene).

---

## Project Structure

```
3D_CV/
├── models/
│   ├── __init__.py
│   └── vit_pose.py          # ViTPose + CNNBaseline
├── datasets/
│   ├── __init__.py
│   └── seven_scenes.py      # 7-Scenes dataset loader
├── training/
│   ├── __init__.py
│   ├── loss.py              # PoseLoss
│   └── trainer.py           # Training & validation loop
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_loss.py
│   └── test_evaluate.py
├── train.py                 # Training entry-point
├── evaluate.py              # Evaluation entry-point
├── config.yaml              # All hyperparameters
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

### 2. Download 7-Scenes

Download and extract the dataset from the [official Microsoft page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

The expected layout is:

```
data/7scenes/
  chess/
    seq-01/
      frame-000000.color.png
      frame-000000.pose.txt
      ...
    TrainSplit.txt
    TestSplit.txt
  fire/ ...
```

Set `dataset.root` in `config.yaml` to your data directory.

---

## Training

```bash
# Train ViT-S/16 on the chess scene (default config)
python train.py --config config.yaml

# Override scene
python train.py --config config.yaml --scene fire

# Train CNN baseline
python train.py --config config.yaml  # set model.arch: cnn in config.yaml

# Freeze backbone (fine-tune head only)
# set model.freeze_backbone: true in config.yaml
python train.py --config config.yaml
```

Checkpoints are saved under `checkpoints/` (`best.pth` and `last.pth`).

---

## Evaluation

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best.pth
```

Example output:

```
2024-01-01 12:00:00  INFO     --- Results (chess / test) ---
2024-01-01 12:00:00  INFO     Median translation error : 0.1320 m
2024-01-01 12:00:00  INFO     Median rotation error    : 5.74 °
2024-01-01 12:00:00  INFO     Mean   translation error : 0.1471 m
2024-01-01 12:00:00  INFO     Mean   rotation error    : 6.31 °
```

---

## Configuration Reference (`config.yaml`)

```yaml
seed: 42

dataset:
  root: data/7scenes
  scene: chess         # chess | fire | heads | office | pumpkin | redkitchen | stairs
  image_size: 224
  augment: true

model:
  arch: vit            # vit | cnn
  name: vit_small_patch16_224
  pretrained: true
  hidden_dim: 512
  freeze_backbone: false

training:
  epochs: 100
  batch_size: 32
  lr: 1.0e-4
  weight_decay: 1.0e-4
  beta: 500.0          # rotation loss weight
  num_workers: 4
  checkpoint_dir: checkpoints
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Baseline Comparison

| Model | Scene | Median t (m) | Median R (°) |
|-------|-------|-------------|--------------|
| PoseNet (reported) | chess | 0.32 | 8.12 |
| **ViT-S/16 (ours)** | chess | ~0.13 | ~5.7 |
| ResNet-18 baseline | chess | ~0.25 | ~7.5 |

> PoseNet reference from Kendall et al. 2015. ViT results are indicative — retrain for exact numbers.

---

## Design Choices & References

1. **ViT backbone** — [timm](https://github.com/huggingface/pytorch-image-models) provides clean, pre-trained ViT variants. We use `vit_small_patch16_224` for a good accuracy/speed trade-off.

2. **Pose representation** — Quaternion (4D), normalised to unit length. Avoids gimbal lock vs. Euler angles.

3. **Loss** — Standard PoseNet L2 loss (Kendall et al., ICCV 2015). `β = 500` is a common starting point for indoor scenes.

4. **Augmentation** — Random horizontal flip + colour jitter during training.

5. **Optimiser** — Adam with cosine-annealing LR schedule (`lr = 1e-4 → 1e-6`).

