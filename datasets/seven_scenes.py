"""
7-Scenes dataset loader.

Directory layout expected on disk
(as distributed by Microsoft Research):

    <root>/
      chess/
        seq-01/
          frame-000000.color.png
          frame-000000.pose.txt
          ...
        seq-02/ ...
        TrainSplit.txt   ← lists sequence numbers used for training
        TestSplit.txt
      fire/
        ...

Each pose file is a 4x4 camera-to-world transformation matrix.

Usage example
-------------
    from datasets import SevenScenesDataset
    from torch.utils.data import DataLoader

    train_ds = SevenScenesDataset(root="data/7scenes", scene="chess", split="train")
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    for images, t, q in loader:
        ...
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ImageNet mean/std used for ViT pre-training
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [qx, qy, qz, qw]."""
    # Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-10)


def _load_pose(pose_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 4×4 pose matrix and return (translation, quaternion)."""
    mat = np.loadtxt(pose_path, dtype=np.float32)  # (4, 4)
    t = mat[:3, 3]                                  # (3,)
    q = _rotation_matrix_to_quaternion(mat[:3, :3]) # (4,)  [qx,qy,qz,qw]
    return t, q


class SevenScenesDataset(Dataset):
    """PyTorch Dataset for the 7-Scenes benchmark.

    Args:
        root: Path to the root 7-Scenes directory.
        scene: One of chess / fire / heads / office / pumpkin /
               redkitchen / stairs.
        split: "train" or "test".
        image_size: Resize images to (image_size, image_size).
        augment: Apply random horizontal flip + colour jitter (train only).
    """

    SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

    def __init__(
        self,
        root: str | os.PathLike,
        scene: str = "chess",
        split: str = "train",
        image_size: int = 224,
        augment: bool = True,
    ) -> None:
        super().__init__()
        if scene not in self.SCENES:
            raise ValueError(f"Unknown scene '{scene}'. Choose from {self.SCENES}.")
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'.")

        self.root = Path(root)
        self.scene = scene
        self.split = split

        # Build transforms
        base = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        if augment and split == "train":
            aug = [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
            self.transform = T.Compose(aug + base)
        else:
            self.transform = T.Compose(base)

        self.samples = self._collect_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sequence_ids(self) -> list[int]:
        """Read sequence IDs from TrainSplit.txt / TestSplit.txt.

        Handles both ``sequence1`` and ``sequence 1`` formats.
        """
        split_file = "TrainSplit.txt" if self.split == "train" else "TestSplit.txt"
        path = self.root / self.scene / split_file
        ids: list[int] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                m = re.search(r"sequence\s*(\d+)", line, re.IGNORECASE)
                if m:
                    ids.append(int(m.group(1)))
        return ids

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        """Return list of (image_path, pose_path) tuples."""
        samples: list[tuple[Path, Path]] = []
        scene_dir = self.root / self.scene
        seq_ids = self._sequence_ids()
        for sid in seq_ids:
            seq_dir = scene_dir / f"seq-{sid:02d}"
            if not seq_dir.is_dir():
                continue
            # Frames are numbered 000000, 000001, ...
            for img_path in sorted(seq_dir.glob("*.color.png")):
                pose_path = img_path.with_suffix("").with_suffix(".pose.txt")
                if pose_path.exists():
                    samples.append((img_path, pose_path))
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, pose_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        t, q = _load_pose(pose_path)
        return image, torch.from_numpy(t), torch.from_numpy(q)
