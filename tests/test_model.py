"""Smoke tests for EfficientNetV2-S model factory."""

from __future__ import annotations

import torch

from waste_classification.model import create_model


def test_create_model_forward():
    model = create_model(model_name="tf_efficientnetv2_s", num_classes=3, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 300, 300)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 3)
