"""Tests for fold checkpoint discovery used by inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from waste_classification.infer import discover_fold_checkpoints


def test_discover_fold_checkpoints_sorted(tmp_path: Path):
    (tmp_path / "fold_2.pth").write_bytes(b"x")
    (tmp_path / "fold_0.pth").write_bytes(b"x")
    (tmp_path / "fold_10.pth").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    paths = discover_fold_checkpoints(tmp_path)
    assert [p.name for p in paths] == ["fold_0.pth", "fold_2.pth", "fold_10.pth"]


def test_discover_fold_checkpoints_empty_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No fold checkpoints"):
        discover_fold_checkpoints(tmp_path)


def test_discover_fold_checkpoints_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Models directory not found"):
        discover_fold_checkpoints(tmp_path / "missing")
