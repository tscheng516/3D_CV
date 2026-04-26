"""Tests for PoseLoss and Trainer (using a tiny synthetic setup)."""

from __future__ import annotations

import math
import torch
import pytest

from training.loss import PoseLoss


# ---------------------------------------------------------------------------
# PoseLoss tests
# ---------------------------------------------------------------------------

def test_pose_loss_zero():
    """Loss should be zero when predictions match ground truth."""
    criterion = PoseLoss(beta=500.0)
    t = torch.zeros(4, 3)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)
    total, t_loss, q_loss = criterion(t, q, t, q)
    assert math.isclose(total.item(), 0.0, abs_tol=1e-6)
    assert math.isclose(t_loss.item(), 0.0, abs_tol=1e-6)
    assert math.isclose(q_loss.item(), 0.0, abs_tol=1e-6)


def test_pose_loss_shape():
    criterion = PoseLoss(beta=1.0)
    B = 8
    t_pred = torch.randn(B, 3)
    q_pred = torch.randn(B, 4)
    t_gt = torch.randn(B, 3)
    q_gt = torch.randn(B, 4)
    total, t_l, q_l = criterion(t_pred, q_pred, t_gt, q_gt)
    assert total.shape == torch.Size([])   # scalar
    assert t_l.shape == torch.Size([])
    assert q_l.shape == torch.Size([])


def test_pose_loss_beta_scaling():
    """Doubling beta should double the rotation contribution."""
    criterion1 = PoseLoss(beta=1.0)
    criterion2 = PoseLoss(beta=2.0)
    t_pred = torch.zeros(2, 3)
    t_gt = torch.zeros(2, 3)
    q_pred = torch.randn(2, 4)
    q_gt = torch.zeros(2, 4)
    total1, _, q_l = criterion1(t_pred, q_pred, t_gt, q_gt)
    total2, _, _ = criterion2(t_pred, q_pred, t_gt, q_gt)
    assert math.isclose(total2.item(), 2.0 * total1.item(), rel_tol=1e-5)


def test_pose_loss_positive():
    criterion = PoseLoss(beta=500.0)
    t_pred = torch.randn(4, 3)
    t_gt = torch.zeros(4, 3)
    q_pred = torch.randn(4, 4)
    q_gt = torch.zeros(4, 4)
    total, t_l, q_l = criterion(t_pred, q_pred, t_gt, q_gt)
    assert total.item() >= 0.0
    assert t_l.item() >= 0.0
    assert q_l.item() >= 0.0
