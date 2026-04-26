"""
PoseNet-style loss function.

Loss = ||t_pred - t_gt||_2  +  beta * ||q_pred - q_gt||_2

where beta scales the rotation term relative to translation.
"""

import torch
import torch.nn as nn


class PoseLoss(nn.Module):
    """Weighted L2 loss on translation and quaternion.

    Args:
        beta: Weight for the rotation term. A common default is 500
              (scene-scale dependent; tune per dataset).
    """

    def __init__(self, beta: float = 500.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        t_pred: torch.Tensor,
        q_pred: torch.Tensor,
        t_gt: torch.Tensor,
        q_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            t_pred: (B, 3) predicted translations.
            q_pred: (B, 4) predicted (unit) quaternions.
            t_gt:   (B, 3) ground-truth translations.
            q_gt:   (B, 4) ground-truth (unit) quaternions.

        Returns:
            total_loss, t_loss, q_loss (all scalar tensors).
        """
        t_loss = torch.mean(torch.norm(t_pred - t_gt, p=2, dim=1))
        q_loss = torch.mean(torch.norm(q_pred - q_gt, p=2, dim=1))
        total = t_loss + self.beta * q_loss
        return total, t_loss, q_loss
