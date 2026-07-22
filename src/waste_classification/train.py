"""Modul 2 — Training engine with Stratified 5-Fold Cross-Validation."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from waste_classification.config import Config
from waste_classification.dataset import WasteTrainDataset, discover_train_samples
from waste_classification.metrics import macro_f1
from waste_classification.model import create_model
from waste_classification.runtime import configure_cpu_threads, dataloader_kwargs
from waste_classification.seed import set_seed
from waste_classification.transforms import get_train_transforms, get_val_transforms


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stratified_subset(
    paths: list[Path],
    labels: list[int],
    max_samples: int,
    seed: int,
) -> tuple[list[Path], list[int]]:
    """
    Stratified downsample to at most max_samples.

    max_samples <= 0 or >= len(paths) keeps the full set.
    """
    n = len(paths)
    if max_samples <= 0 or max_samples >= n:
        return list(paths), list(labels)

    labels_arr = np.asarray(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=max_samples,
        random_state=seed,
    )
    keep_idx, _ = next(splitter.split(np.arange(n), labels_arr))
    keep_idx = np.sort(keep_idx)
    return [paths[i] for i in keep_idx], labels_arr[keep_idx].tolist()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    non_blocking = device.type == "cuda"
    use_amp = device.type == "cuda"

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            images = batch["image"].to(device, non_blocking=non_blocking)
            labels = batch["label"].to(device, non_blocking=non_blocking)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            losses.append(float(loss.item()))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    f1 = macro_f1(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, f1


def train_fold(
    cfg: Config,
    fold: int,
    train_paths: list[Path],
    train_labels: list[int],
    val_paths: list[Path],
    val_labels: list[int],
    device: torch.device,
) -> dict:
    """Train a single fold and save the best checkpoint by validation macro F1."""
    set_seed(cfg.seed + fold)

    train_ds = WasteTrainDataset(
        train_paths,
        train_labels,
        transform=get_train_transforms(cfg.model.image_size),
    )
    val_ds = WasteTrainDataset(
        val_paths,
        val_labels,
        transform=get_val_transforms(cfg.model.image_size),
    )

    loader_kwargs = dataloader_kwargs(
        cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = create_model(
        model_name=cfg.model.name,
        num_classes=cfg.num_classes,
        pretrained=cfg.model.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=cfg.train.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    epochs_without_improve = 0
    best_path = cfg.paths.models_dir / f"fold_{fold}.pth"
    history: list[dict] = []

    try:
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss, train_f1 = _run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
                scaler=scaler,
            )
            val_loss, val_f1 = _run_epoch(
                model, val_loader, criterion, None, device, train=False, scaler=None
            )
            scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "lr": float(scheduler.get_last_lr()[0]),
            }
            history.append(record)
            print(
                f"[fold {fold}] epoch {epoch}/{cfg.train.epochs} "
                f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_without_improve = 0
                cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_macro_f1": best_f1,
                        "model_name": cfg.model.name,
                        "num_classes": cfg.num_classes,
                        "image_size": cfg.model.image_size,
                    },
                    best_path,
                )
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= cfg.train.patience:
                    print(
                        f"[fold {fold}] early stopping at epoch {epoch} "
                        f"(patience={cfg.train.patience})"
                    )
                    break
    finally:
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        _clear_cuda()

    passed = best_f1 >= cfg.train.target_macro_f1
    return {
        "fold": fold,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(best_path),
        "passed_target": passed,
        "history": history,
    }


def run_training(cfg: Config) -> dict:
    """
    Full Stratified K-Fold training loop.

    Guardrail: only uses labeled train data — never reads test images/labels.
    """
    set_seed(cfg.seed)
    device = _device()
    threads = configure_cpu_threads(cfg.train.torch_threads)
    print(f"Device: {device} | torch_threads={threads}")

    paths, labels = discover_train_samples(cfg.paths.train_dir)
    print(f"Found {len(paths)} training images")

    paths, labels = stratified_subset(
        paths,
        labels,
        max_samples=cfg.train.max_samples,
        seed=cfg.seed,
    )
    if cfg.train.max_samples > 0:
        print(f"Using stratified subset of {len(paths)} images (max_samples={cfg.train.max_samples})")

    labels_arr = np.asarray(labels)
    paths_list = list(paths)

    skf = StratifiedKFold(
        n_splits=cfg.train.n_folds,
        shuffle=True,
        random_state=cfg.seed,
    )

    fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths_list, labels_arr)):
        print(f"\n=== Fold {fold}/{cfg.train.n_folds - 1} ===")
        train_paths = [paths_list[i] for i in train_idx]
        train_labels = labels_arr[train_idx].tolist()
        val_paths = [paths_list[i] for i in val_idx]
        val_labels = labels_arr[val_idx].tolist()

        result = train_fold(
            cfg,
            fold,
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            device,
        )
        fold_results.append(result)
        status = "PASS" if result["passed_target"] else "BELOW TARGET"
        print(
            f"Fold {fold} best macro F1={result['best_val_macro_f1']:.4f} "
            f"(target {cfg.train.target_macro_f1:.2f}) [{status}]"
        )

    mean_f1 = float(np.mean([r["best_val_macro_f1"] for r in fold_results]))
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "model": cfg.model.name,
        "image_size": cfg.model.image_size,
        "n_folds": cfg.train.n_folds,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.train.learning_rate,
        "weight_decay": cfg.train.weight_decay,
        "patience": cfg.train.patience,
        "max_samples": cfg.train.max_samples,
        "torch_threads": threads,
        "mean_best_val_macro_f1": mean_f1,
        "target_macro_f1": cfg.train.target_macro_f1,
        "folds": fold_results,
    }

    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.paths.logs_dir / "train_summary.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nMean best val macro F1: {mean_f1:.4f}")
    print(f"Training log written to {log_path}")
    return summary
