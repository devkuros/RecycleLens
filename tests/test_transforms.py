"""Smoke tests for Albumentations transforms."""

from __future__ import annotations

import numpy as np

from waste_classification.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)


def test_train_transform_output_shape():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    transform = get_train_transforms(image_size=300)
    out = transform(image=image)["image"]
    assert out.shape == (3, 300, 300)


def test_val_and_test_transforms_match_shape():
    image = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
    val_out = get_val_transforms(224)(image=image)["image"]
    test_out = get_test_transforms(224)(image=image)["image"]
    assert val_out.shape == (3, 224, 224)
    assert test_out.shape == (3, 224, 224)
