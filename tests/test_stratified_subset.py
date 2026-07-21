"""Tests for stratified training subset helper."""

from __future__ import annotations

from pathlib import Path

from waste_classification.train import stratified_subset


def test_stratified_subset_reduces_count():
    paths = [Path(f"img_{i}.jpg") for i in range(30)]
    labels = [0] * 10 + [1] * 10 + [2] * 10
    sub_paths, sub_labels = stratified_subset(paths, labels, max_samples=12, seed=42)
    assert len(sub_paths) == 12
    assert len(sub_labels) == 12
    assert set(sub_labels) == {0, 1, 2}


def test_stratified_subset_noop_when_disabled():
    paths = [Path(f"img_{i}.jpg") for i in range(9)]
    labels = [0, 1, 2] * 3
    sub_paths, sub_labels = stratified_subset(paths, labels, max_samples=0, seed=42)
    assert sub_paths == paths
    assert sub_labels == labels


def test_stratified_subset_noop_when_larger_than_data():
    paths = [Path(f"img_{i}.jpg") for i in range(6)]
    labels = [0, 1, 2, 0, 1, 2]
    sub_paths, sub_labels = stratified_subset(paths, labels, max_samples=100, seed=0)
    assert len(sub_paths) == 6
    assert sub_labels == labels
