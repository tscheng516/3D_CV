"""
ViT-based 6-DoF camera pose regression model.

Architecture:
  - Backbone: ViT (via timm), classification head removed
  - Head: MLP regression head → 7 outputs (3 translation + 4 quaternion)
"""

import torch
import torch.nn as nn
import timm


class PoseHead(nn.Module):
    """MLP regression head: hidden_dim → ReLU → 7 outputs."""

    def __init__(self, in_features: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ViTPose(nn.Module):
    """ViT backbone + MLP regression head for 6-DoF pose estimation.

    Outputs a 7-dim vector: [tx, ty, tz, qx, qy, qz, qw].
    The quaternion is L2-normalised during inference.

    Args:
        model_name: timm model name (default: vit_small_patch16_224).
        pretrained: load ImageNet-pretrained weights.
        hidden_dim: width of the hidden layer in the pose head.
        freeze_backbone: if True, backbone weights are frozen.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        hidden_dim: int = 512,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load backbone, strip the classifier head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        self.head = PoseHead(feature_dim, hidden_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images.

        Returns:
            t_pred: (B, 3) translation prediction.
            q_pred: (B, 4) unit-quaternion prediction [qx, qy, qz, qw].
        """
        features = self.backbone(x)          # (B, feature_dim)
        out = self.head(features)            # (B, 7)
        t_pred = out[:, :3]
        q_pred = out[:, 3:]
        q_pred = nn.functional.normalize(q_pred, p=2, dim=1)
        return t_pred, q_pred


class CNNBaseline(nn.Module):
    """Lightweight ResNet-18 baseline for comparison.

    Uses the same output convention as ViTPose.
    """

    def __init__(self, pretrained: bool = True, hidden_dim: int = 512) -> None:
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.head = PoseHead(feature_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        out = self.head(features)
        t_pred = out[:, :3]
        q_pred = nn.functional.normalize(out[:, 3:], p=2, dim=1)
        return t_pred, q_pred
