"""Modul 3 — Ensemble inference and submission exporter."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from waste_classification.config import Config
from waste_classification.dataset import WasteTestDataset
from waste_classification.model import create_model
from waste_classification.runtime import configure_cpu_threads, dataloader_kwargs
from waste_classification.seed import set_seed
from waste_classification.transforms import get_test_transforms
from waste_classification.validate_submission import validate_submission

_FOLD_RE = re.compile(r"^fold_(\d+)\.pth$", re.IGNORECASE)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_fold_checkpoints(models_dir: Path) -> list[Path]:
    """Return existing fold_*.pth paths sorted by fold index."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    found: list[tuple[int, Path]] = []
    for path in models_dir.glob("fold_*.pth"):
        match = _FOLD_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))

    if not found:
        raise FileNotFoundError(
            f"No fold checkpoints (fold_*.pth) found under {models_dir}"
        )
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def _load_checkpoint(path: Path, cfg: Config, device: torch.device) -> tuple[torch.nn.Module, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model_name = checkpoint.get("model_name", cfg.model.name)
    num_classes = checkpoint.get("num_classes", cfg.num_classes)
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _predict_proba(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    probs_list: list[np.ndarray] = []
    non_blocking = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, leave=False, desc="infer"):
            images = batch["image"].to(device, non_blocking=non_blocking)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    if not probs_list:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(probs_list, axis=0)


def run_inference(cfg: Config, output_name: str = "submission.csv") -> Path:
    """
    Run fold ensemble inference on test images ordered by template.csv.

    Uses all available fold_*.pth under models_dir (1+ folds).
    Guardrail: this is the only path that reads test images.
    """
    set_seed(cfg.seed)
    device = _device()
    threads = configure_cpu_threads(cfg.train.torch_threads)
    print(f"Device: {device} | torch_threads={threads}")

    checkpoint_paths = discover_fold_checkpoints(cfg.paths.models_dir)
    print(f"Using {len(checkpoint_paths)} fold checkpoint(s): "
          + ", ".join(p.name for p in checkpoint_paths))

    first_model, first_ckpt = _load_checkpoint(checkpoint_paths[0], cfg, device)
    image_size = int(first_ckpt.get("image_size", cfg.model.image_size))
    print(f"Inference image_size={image_size}")

    test_ds = WasteTestDataset(
        template_csv=cfg.paths.template_csv,
        test_dir=cfg.paths.test_dir,
        id_column=cfg.submission.id_column,
        transform=get_test_transforms(image_size),
    )
    loader_kwargs = dataloader_kwargs(
        cfg.infer.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.infer.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"Loading {checkpoint_paths[0].name}")
    ensemble = _predict_proba(first_model, test_loader, device)
    del first_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for path in checkpoint_paths[1:]:
        print(f"Loading {path.name}")
        model, _ = _load_checkpoint(path, cfg, device)
        probs = _predict_proba(model, test_loader, device)
        ensemble += probs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble /= len(checkpoint_paths)
    labels = np.argmax(ensemble, axis=1).astype(int)

    ids = test_ds.ids
    if len(ids) != len(labels):
        raise RuntimeError("Mismatch between template IDs and prediction count")

    submission = pd.DataFrame(
        {
            cfg.submission.id_column: ids,
            cfg.submission.label_column: labels,
        }
    )

    cfg.paths.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.paths.submissions_dir / output_name
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    result = validate_submission(
        submission_csv=out_path,
        template_csv=cfg.paths.template_csv,
        expected_rows=cfg.submission.expected_rows,
        id_column=cfg.submission.id_column,
        label_column=cfg.submission.label_column,
    )
    print(f"Submission QA passed: {result}")
    return out_path
