"""Tests for submission CSV integrity checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from waste_classification.validate_submission import (
    SubmissionValidationError,
    validate_submission,
)


def _write_pair(tmp_path: Path, n: int = 5, shuffle_ids: bool = False, bad_label: int | None = None):
    ids = [f"img_{i:04d}.jpg" for i in range(n)]
    tpl = pd.DataFrame({"id": ids, "predicted": [0] * n})
    labels = [i % 3 for i in range(n)]
    if bad_label is not None:
        labels[0] = bad_label
    sub_ids = list(reversed(ids)) if shuffle_ids else ids
    sub = pd.DataFrame({"id": sub_ids, "predicted": labels if not shuffle_ids else labels})
    tpl_path = tmp_path / "template.csv"
    sub_path = tmp_path / "submission.csv"
    tpl.to_csv(tpl_path, index=False)
    sub.to_csv(sub_path, index=False)
    return sub_path, tpl_path


def test_validate_submission_ok(tmp_path: Path):
    sub_path, tpl_path = _write_pair(tmp_path, n=5)
    result = validate_submission(
        sub_path, tpl_path, expected_rows=5, id_column="id", label_column="predicted"
    )
    assert result["ok"] is True
    assert result["rows"] == 5


def test_validate_submission_wrong_order(tmp_path: Path):
    sub_path, tpl_path = _write_pair(tmp_path, n=5, shuffle_ids=True)
    with pytest.raises(SubmissionValidationError, match="order"):
        validate_submission(sub_path, tpl_path, expected_rows=5)


def test_validate_submission_bad_label(tmp_path: Path):
    sub_path, tpl_path = _write_pair(tmp_path, n=5, bad_label=9)
    with pytest.raises(SubmissionValidationError, match=r"\{0, 1, 2\}"):
        validate_submission(sub_path, tpl_path, expected_rows=5)


def test_validate_submission_wrong_row_count(tmp_path: Path):
    sub_path, tpl_path = _write_pair(tmp_path, n=3)
    with pytest.raises(SubmissionValidationError, match="Expected"):
        validate_submission(sub_path, tpl_path, expected_rows=1458)
