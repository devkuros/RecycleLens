"""Tests for labeled train folder discovery."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from waste_classification.dataset import (
    _class_id_from_folder_name,
    discover_train_samples,
)


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(path)


def test_class_id_from_prefixed_folder_names():
    assert _class_id_from_folder_name("0_Recyclable") == 0
    assert _class_id_from_folder_name("1_Electronic") == 1
    assert _class_id_from_folder_name("2_Organic") == 2
    assert _class_id_from_folder_name("Recyclable") == 0
    assert _class_id_from_folder_name("0") == 0


def test_discover_prefixed_class_folders(tmp_path: Path):
    _touch_jpg(tmp_path / "0_Recyclable" / "a.jpg")
    _touch_jpg(tmp_path / "1_Electronic" / "b.jpg")
    _touch_jpg(tmp_path / "2_Organic" / "c.jpg")
    paths, labels = discover_train_samples(tmp_path)
    assert len(paths) == 3
    assert sorted(labels) == [0, 1, 2]
