"""EfficientNetV2-S classifier via timm."""

from __future__ import annotations

import timm
import torch.nn as nn


def create_model(
    model_name: str = "tf_efficientnetv2_s",
    num_classes: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a classification model with a 3-class output head.

    Softmax is applied at inference/metrics time; training uses CrossEntropyLoss
    which expects raw logits.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
