"""Tests for evaluate.py helper functions."""

import numpy as np
import pytest

from evaluate import translation_error, rotation_error_deg


def test_translation_error_zero():
    t = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    errs = translation_error(t, t)
    np.testing.assert_allclose(errs, 0.0, atol=1e-6)


def test_translation_error_known():
    t_pred = np.array([[3.0, 0.0, 0.0]])
    t_gt = np.array([[0.0, 0.0, 0.0]])
    errs = translation_error(t_pred, t_gt)
    np.testing.assert_allclose(errs, [3.0], atol=1e-6)


def test_rotation_error_zero():
    q = np.array([[0.0, 0.0, 0.0, 1.0]] * 4)
    errs = rotation_error_deg(q, q)
    np.testing.assert_allclose(errs, 0.0, atol=1e-5)


def test_rotation_error_180_degrees():
    """Opposite quaternions (same rotation) should give 0 degrees."""
    q = np.array([[0.0, 0.0, 0.0, 1.0]])
    q_neg = -q
    errs = rotation_error_deg(q, q_neg)
    np.testing.assert_allclose(errs, 0.0, atol=1e-5)


def test_rotation_error_90_degrees():
    """90-degree rotation around z: q = [0, 0, sin(45°), cos(45°)]."""
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    q_id = np.array([[0.0, 0.0, 0.0, 1.0]])
    q_90 = np.array([[0.0, 0.0, s, c]])
    errs = rotation_error_deg(q_id, q_90)
    np.testing.assert_allclose(errs, [90.0], atol=1e-4)
