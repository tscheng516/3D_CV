"""Tests for ViTPose and CNNBaseline models."""

import torch
import pytest

from models import ViTPose, CNNBaseline


BATCH = 2
IMG_SIZE = 224


@pytest.fixture(scope="module")
def vit_model():
    return ViTPose(model_name="vit_tiny_patch16_224", pretrained=False, hidden_dim=128)


@pytest.fixture(scope="module")
def cnn_model():
    return CNNBaseline(pretrained=False, hidden_dim=128)


@pytest.fixture
def dummy_input():
    return torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)


# ------------------------------------------------------------------
# ViT model tests
# ------------------------------------------------------------------

def test_vit_output_shapes(vit_model, dummy_input):
    vit_model.eval()
    with torch.no_grad():
        t, q = vit_model(dummy_input)
    assert t.shape == (BATCH, 3), f"Expected (B,3), got {t.shape}"
    assert q.shape == (BATCH, 4), f"Expected (B,4), got {q.shape}"


def test_vit_quaternion_unit_norm(vit_model, dummy_input):
    vit_model.eval()
    with torch.no_grad():
        _, q = vit_model(dummy_input)
    norms = torch.norm(q, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5), \
        f"Quaternions are not unit-norm: {norms}"


def test_vit_forward_backward(vit_model, dummy_input):
    vit_model.train()
    t, q = vit_model(dummy_input)
    loss = t.sum() + q.sum()
    loss.backward()    # should not raise


def test_vit_freeze_backbone():
    model = ViTPose(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        freeze_backbone=True,
    )
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone param {name} should be frozen"
    for name, param in model.head.named_parameters():
        assert param.requires_grad, f"Head param {name} should require grad"


# ------------------------------------------------------------------
# CNN baseline tests
# ------------------------------------------------------------------

def test_cnn_output_shapes(cnn_model, dummy_input):
    cnn_model.eval()
    with torch.no_grad():
        t, q = cnn_model(dummy_input)
    assert t.shape == (BATCH, 3)
    assert q.shape == (BATCH, 4)


def test_cnn_quaternion_unit_norm(cnn_model, dummy_input):
    cnn_model.eval()
    with torch.no_grad():
        _, q = cnn_model(dummy_input)
    norms = torch.norm(q, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)
