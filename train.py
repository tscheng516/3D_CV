"""
Main training entry-point.

Example
-------
    python train.py --config config.yaml --scene chess --model_name vit_small_patch16_224
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import SevenScenesDataset
from models import ViTPose, CNNBaseline
from training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("arch", "vit")
    if arch == "cnn":
        logger.info("Building CNN baseline (ResNet-18)")
        return CNNBaseline(
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 512),
        )
    logger.info("Building ViT pose model: %s", model_cfg.get("name", "vit_small_patch16_224"))
    return ViTPose(
        model_name=model_cfg.get("name", "vit_small_patch16_224"),
        pretrained=model_cfg.get("pretrained", True),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViT pose estimator on 7-Scenes")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--scene", default=None, help="Override config scene")
    parser.add_argument("--model_name", default=None, help="Override config model name")
    parser.add_argument("--data_root", default=None, help="Override config data root")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Command-line overrides
    if args.scene:
        cfg["dataset"]["scene"] = args.scene
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    if args.data_root:
        cfg["dataset"]["root"] = args.data_root
    if args.checkpoint_dir:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Data
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})

    train_ds = SevenScenesDataset(
        root=data_cfg["root"],
        scene=data_cfg.get("scene", "chess"),
        split="train",
        image_size=data_cfg.get("image_size", 224),
        augment=data_cfg.get("augment", True),
    )
    val_ds = SevenScenesDataset(
        root=data_cfg["root"],
        scene=data_cfg.get("scene", "chess"),
        split="test",
        image_size=data_cfg.get("image_size", 224),
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=(device == "cuda"),
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # Model
    model = build_model(cfg)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
    )
    trainer.train()

    logger.info("Training complete. Best val loss: %.4f", trainer.best_val_loss)


if __name__ == "__main__":
    main()
