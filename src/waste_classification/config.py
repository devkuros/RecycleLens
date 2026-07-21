"""Configuration loading for the waste classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "tf_efficientnetv2_s"
    pretrained: bool = True
    image_size: int = 300


@dataclass
class TrainConfig:
    n_folds: int = 5
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    min_lr: float = 1e-6
    target_macro_f1: float = 0.92
    patience: int = 3
    torch_threads: int | None = None
    max_samples: int = 0


@dataclass
class InferConfig:
    batch_size: int = 64
    num_workers: int = 0


@dataclass
class PathsConfig:
    train_dir: Path = Path("data/train")
    test_dir: Path = Path("data/test")
    template_csv: Path = Path("data/template.csv")
    models_dir: Path = Path("outputs/models")
    logs_dir: Path = Path("outputs/logs")
    submissions_dir: Path = Path("outputs/submissions")


@dataclass
class SubmissionConfig:
    expected_rows: int = 1458
    id_column: str = "id"
    label_column: str = "predicted"


@dataclass
class Config:
    seed: int = 42
    num_classes: int = 3
    class_names: dict[int, str] = field(
        default_factory=lambda: {0: "Recyclable", 1: "Electronic", 2: "Organic"}
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    root: Path = field(default_factory=lambda: Path.cwd())

    def resolve_paths(self) -> None:
        """Resolve relative paths against project root."""
        for attr in (
            "train_dir",
            "test_dir",
            "template_csv",
            "models_dir",
            "logs_dir",
            "submissions_dir",
        ):
            value = getattr(self.paths, attr)
            path = Path(value)
            if not path.is_absolute():
                setattr(self.paths, attr, self.root / path)


def _as_int_keys(mapping: dict[Any, Any]) -> dict[int, str]:
    return {int(k): str(v) for k, v in mapping.items()}


def load_config(path: str | Path, root: str | Path | None = None) -> Config:
    """Load YAML config into a typed Config object."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    project_root = Path(root) if root else config_path.resolve().parent.parent

    class_names = _as_int_keys(raw.get("class_names", {0: "Recyclable", 1: "Electronic", 2: "Organic"}))
    model_raw = raw.get("model", {})
    train_raw = raw.get("train", {})
    infer_raw = raw.get("infer", {})
    paths_raw = raw.get("paths", {})
    submission_raw = raw.get("submission", {})

    cfg = Config(
        seed=int(raw.get("seed", 42)),
        num_classes=int(raw.get("num_classes", 3)),
        class_names=class_names,
        model=ModelConfig(
            name=str(model_raw.get("name", "tf_efficientnetv2_s")),
            pretrained=bool(model_raw.get("pretrained", True)),
            image_size=int(model_raw.get("image_size", 300)),
        ),
        train=TrainConfig(
            n_folds=int(train_raw.get("n_folds", 5)),
            epochs=int(train_raw.get("epochs", 10)),
            batch_size=int(train_raw.get("batch_size", 32)),
            num_workers=int(train_raw.get("num_workers", 2)),
            learning_rate=float(train_raw.get("learning_rate", 1e-4)),
            weight_decay=float(train_raw.get("weight_decay", 1e-2)),
            min_lr=float(train_raw.get("min_lr", 1e-6)),
            target_macro_f1=float(train_raw.get("target_macro_f1", 0.92)),
            patience=int(train_raw.get("patience", 3)),
            torch_threads=(
                None
                if train_raw.get("torch_threads", None) is None
                else int(train_raw["torch_threads"])
            ),
            max_samples=int(train_raw.get("max_samples", 0)),
        ),
        infer=InferConfig(
            batch_size=int(infer_raw.get("batch_size", 64)),
            num_workers=int(infer_raw.get("num_workers", 0)),
        ),
        paths=PathsConfig(
            train_dir=Path(paths_raw.get("train_dir", "data/train")),
            test_dir=Path(paths_raw.get("test_dir", "data/test")),
            template_csv=Path(paths_raw.get("template_csv", "data/template.csv")),
            models_dir=Path(paths_raw.get("models_dir", "outputs/models")),
            logs_dir=Path(paths_raw.get("logs_dir", "outputs/logs")),
            submissions_dir=Path(paths_raw.get("submissions_dir", "outputs/submissions")),
        ),
        submission=SubmissionConfig(
            expected_rows=int(submission_raw.get("expected_rows", 1458)),
            id_column=str(submission_raw.get("id_column", "id")),
            label_column=str(submission_raw.get("label_column", "predicted")),
        ),
        root=project_root,
    )
    cfg.resolve_paths()
    return cfg
