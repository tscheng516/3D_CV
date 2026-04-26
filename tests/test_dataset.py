"""Tests for the 7-Scenes dataset loader (offline, using synthetic data)."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets import SevenScenesDataset


# ---------------------------------------------------------------------------
# Helpers to build a minimal synthetic 7-Scenes directory
# ---------------------------------------------------------------------------

def _write_pose(path: Path, R: np.ndarray, t: np.ndarray) -> None:
    """Write a 4×4 camera-to-world pose matrix to a text file."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3] = t
    np.savetxt(path, mat)


def _identity_rotation() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory) -> Path:
    """Create a minimal synthetic 7-Scenes tree with two sequences."""
    root = tmp_path_factory.mktemp("7scenes")
    scene_dir = root / "chess"
    scene_dir.mkdir()

    # Two sequences: seq-01 (3 frames) and seq-02 (2 frames)
    frame_counts = {1: 3, 2: 2}
    for sid, n_frames in frame_counts.items():
        seq_dir = scene_dir / f"seq-{sid:02d}"
        seq_dir.mkdir()
        for i in range(n_frames):
            # Colour image (random RGB, 640×480)
            img = Image.fromarray(
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            )
            img.save(seq_dir / f"frame-{i:06d}.color.png")
            # Pose file
            t = np.random.randn(3).astype(np.float32)
            _write_pose(seq_dir / f"frame-{i:06d}.pose.txt", _identity_rotation(), t)

    # Split files
    (scene_dir / "TrainSplit.txt").write_text("sequence1\nsequence2\n")
    (scene_dir / "TestSplit.txt").write_text("sequence2\n")

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dataset_length_train(synthetic_root):
    ds = SevenScenesDataset(root=synthetic_root, scene="chess", split="train", augment=False)
    assert len(ds) == 5, f"Expected 5 samples (3+2), got {len(ds)}"


def test_dataset_length_test(synthetic_root):
    ds = SevenScenesDataset(root=synthetic_root, scene="chess", split="test", augment=False)
    assert len(ds) == 2, f"Expected 2 samples, got {len(ds)}"


def test_item_shapes(synthetic_root):
    ds = SevenScenesDataset(
        root=synthetic_root, scene="chess", split="train", image_size=224, augment=False
    )
    img, t, q = ds[0]
    assert img.shape == (3, 224, 224), f"Image shape mismatch: {img.shape}"
    assert t.shape == (3,), f"Translation shape mismatch: {t.shape}"
    assert q.shape == (4,), f"Quaternion shape mismatch: {q.shape}"


def test_quaternion_unit_norm(synthetic_root):
    ds = SevenScenesDataset(root=synthetic_root, scene="chess", split="train", augment=False)
    import torch
    for i in range(len(ds)):
        _, _, q = ds[i]
        norm = float(torch.norm(q, p=2))
        assert abs(norm - 1.0) < 1e-5, f"Sample {i}: quaternion norm {norm:.6f} != 1"


def test_invalid_scene_raises(synthetic_root):
    with pytest.raises(ValueError, match="Unknown scene"):
        SevenScenesDataset(root=synthetic_root, scene="nonexistent", split="train")


def test_invalid_split_raises(synthetic_root):
    with pytest.raises(ValueError, match="split must be"):
        SevenScenesDataset(root=synthetic_root, scene="chess", split="val")
