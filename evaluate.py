"""
Evaluation script — computes median translation and rotation errors.

Usage
-----
    python evaluate.py --config config.yaml --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import SevenScenesDataset
from models import ViTPose, CNNBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> np.ndarray:
    """Per-sample L2 translation error in metres (shape: (N,))."""
    return np.linalg.norm(t_pred - t_gt, axis=1)


def rotation_error_deg(q_pred: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    """Per-sample rotation error in degrees using quaternion dot-product.

    angle = 2 * arccos(|q_pred · q_gt|)

    Args:
        q_pred: (N, 4) predicted unit quaternions [qx, qy, qz, qw].
        q_gt:   (N, 4) ground-truth unit quaternions.

    Returns:
        errors: (N,) rotation errors in degrees.
    """
    dot = np.sum(q_pred * q_gt, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    return np.degrees(angle_rad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pose estimator on 7-Scenes")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.scene:
        cfg["dataset"]["scene"] = args.scene

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Data
    data_cfg = cfg.get("dataset", {})
    ds = SevenScenesDataset(
        root=data_cfg["root"],
        scene=data_cfg.get("scene", "chess"),
        split=args.split,
        image_size=data_cfg.get("image_size", 224),
        augment=False,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    logger.info("Evaluating on %d samples (%s / %s)", len(ds), data_cfg.get("scene"), args.split)

    # Model
    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("arch", "vit")
    if arch == "cnn":
        model = CNNBaseline(pretrained=False, hidden_dim=model_cfg.get("hidden_dim", 512))
    else:
        model = ViTPose(
            model_name=model_cfg.get("name", "vit_small_patch16_224"),
            pretrained=False,
            hidden_dim=model_cfg.get("hidden_dim", 512),
        )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_t_pred, all_t_gt = [], []
    all_q_pred, all_q_gt = [], []

    with torch.no_grad():
        for images, t_gt, q_gt in loader:
            images = images.to(device)
            t_pred, q_pred = model(images)
            all_t_pred.append(t_pred.cpu().numpy())
            all_t_gt.append(t_gt.numpy())
            all_q_pred.append(q_pred.cpu().numpy())
            all_q_gt.append(q_gt.numpy())

    all_t_pred = np.concatenate(all_t_pred)
    all_t_gt = np.concatenate(all_t_gt)
    all_q_pred = np.concatenate(all_q_pred)
    all_q_gt = np.concatenate(all_q_gt)

    t_errs = translation_error(all_t_pred, all_t_gt)
    r_errs = rotation_error_deg(all_q_pred, all_q_gt)

    logger.info("--- Results (%s / %s) ---", data_cfg.get("scene"), args.split)
    logger.info("Median translation error : %.4f m", np.median(t_errs))
    logger.info("Median rotation error    : %.4f °", np.median(r_errs))
    logger.info("Mean   translation error : %.4f m", np.mean(t_errs))
    logger.info("Mean   rotation error    : %.4f °", np.mean(r_errs))


if __name__ == "__main__":
    main()
