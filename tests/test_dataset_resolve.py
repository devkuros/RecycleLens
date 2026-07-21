"""Tests for test-image path resolution from template ids."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from waste_classification.dataset import resolve_image_path


def test_resolve_numeric_id_with_jpg(tmp_path: Path):
    img_path = tmp_path / "1.jpg"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)
    assert resolve_image_path(tmp_path, "1") == img_path


def test_resolve_zero_padded(tmp_path: Path):
    img_path = tmp_path / "0007.jpg"
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(img_path)
    assert resolve_image_path(tmp_path, "7") == img_path


def test_resolve_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_image_path(tmp_path, "999")
