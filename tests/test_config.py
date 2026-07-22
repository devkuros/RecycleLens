"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

from waste_classification.config import load_config


def test_load_default_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml", root=root)
    assert cfg.model.name == "tf_efficientnetv2_s"
    assert cfg.model.image_size == 300
    assert cfg.train.n_folds == 5
    assert cfg.train.num_workers == 2
    assert cfg.train.patience == 3
    assert cfg.train.max_samples == 0
    assert cfg.train.torch_threads is None
    assert cfg.infer.batch_size == 64
    assert cfg.infer.num_workers == 2
    assert cfg.num_classes == 3
    assert cfg.paths.train_dir == root / "data" / "train"


def test_load_fast_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "fast.yaml", root=root)
    assert cfg.model.image_size == 224
    assert cfg.train.n_folds == 2
    assert cfg.train.epochs == 3
    assert cfg.train.batch_size == 48
    assert cfg.train.num_workers == 0
    assert cfg.train.patience == 2
    assert cfg.train.max_samples == 8000
    assert cfg.infer.batch_size == 96
    assert cfg.infer.num_workers == 0


def test_load_colab_config():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "colab.yaml", root=root)
    assert cfg.model.image_size == 300
    assert cfg.train.n_folds == 5
    assert cfg.train.epochs == 10
    assert cfg.train.batch_size == 8
    assert cfg.infer.batch_size == 16
