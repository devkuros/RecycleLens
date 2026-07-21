"""Shared CPU / DataLoader runtime helpers for train and infer."""

from __future__ import annotations

import os


def configure_cpu_threads(torch_threads: int | None) -> int:
    """Set PyTorch intra-op threads; None → use all logical CPUs."""
    import torch

    threads = torch_threads if torch_threads is not None else (os.cpu_count() or 4)
    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    return threads


def dataloader_kwargs(num_workers: int, pin_memory: bool) -> dict:
    """Common DataLoader kwargs; enables prefetch when workers > 0."""
    kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs
