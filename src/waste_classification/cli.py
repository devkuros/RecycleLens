"""CLI entry points for training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from waste_classification.config import load_config
from waste_classification.infer import run_inference
from waste_classification.train import run_training


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train waste classification model (5-fold)")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config(),
        help="Path to YAML config",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_training(cfg)


def predict_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference → submission.csv")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config(),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="submission.csv",
        help="Filename under outputs/submissions/",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_inference(cfg, output_name=args.output_name)
