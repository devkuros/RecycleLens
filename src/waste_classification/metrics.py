"""Evaluation metrics — Macro F1-Score."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray | list[int], y_pred: np.ndarray | list[int]) -> float:
    """Compute macro-averaged F1 score over classes 0, 1, 2."""
    return float(
        f1_score(
            np.asarray(y_true),
            np.asarray(y_pred),
            average="macro",
            labels=[0, 1, 2],
            zero_division=0,
        )
    )
