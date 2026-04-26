"""
Training loop for the ViT pose estimator.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .loss import PoseLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Encapsulates training and validation loops.

    Args:
        model: The pose estimation model.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        cfg: Config dict (from config.yaml, parsed by OmegaConf / yaml).
        device: torch device string ("cuda" or "cpu").
        checkpoint_dir: Where to save model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        training_cfg = cfg.get("training", {})
        self.criterion = PoseLoss(beta=training_cfg.get("beta", 500.0))
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_cfg.get("lr", 1e-4),
            weight_decay=training_cfg.get("weight_decay", 1e-4),
        )
        self.epochs = training_cfg.get("epochs", 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.best_val_loss = math.inf
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "t_error": [],
            "q_error": [],
        }

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float, float]:
        """Run one epoch.

        Returns:
            avg_loss, avg_t_loss, avg_q_loss (all in metres / raw units).
        """
        self.model.train(train)
        total_loss = total_t = total_q = 0.0
        n = 0
        with torch.set_grad_enabled(train):
            for images, t_gt, q_gt in loader:
                images = images.to(self.device)
                t_gt = t_gt.to(self.device)
                q_gt = q_gt.to(self.device)

                t_pred, q_pred = self.model(images)
                loss, t_loss, q_loss = self.criterion(t_pred, q_pred, t_gt, q_gt)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                bs = images.size(0)
                total_loss += loss.item() * bs
                total_t += t_loss.item() * bs
                total_q += q_loss.item() * bs
                n += bs

        return total_loss / n, total_t / n, total_q / n

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop."""
        for epoch in range(1, self.epochs + 1):
            train_loss, t_tr, q_tr = self._run_epoch(self.train_loader, train=True)
            val_loss, t_val, q_val = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["t_error"].append(t_val)
            self.history["q_error"].append(q_val)

            logger.info(
                "Epoch %3d/%d | train %.4f | val %.4f | t_err %.4fm | q_err %.4f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                t_val,
                q_val,
            )

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, "best.pth")

        # Always save the final checkpoint
        self._save_checkpoint(self.epochs, val_loss, "last.pth")

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_loss: float, filename: str) -> None:
        path = self.ckpt_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)
