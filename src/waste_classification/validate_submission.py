"""QA validation for submission.csv against template.csv and SDLC rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class SubmissionValidationError(ValueError):
    """Raised when submission.csv fails integrity checks."""


def validate_submission(
    submission_csv: Path | str,
    template_csv: Path | str,
    expected_rows: int = 1458,
    id_column: str = "id",
    label_column: str = "predicted",
) -> dict:
    """
    Verify submission integrity:
    - exact expected_rows
    - id order matches template.csv 100%
    - labels are integers in {0, 1, 2}
    - no null/NaN values
    """
    submission_path = Path(submission_csv)
    template_path = Path(template_csv)

    sub = pd.read_csv(submission_path)
    tpl = pd.read_csv(template_path)

    errors: list[str] = []

    for col in (id_column, label_column):
        if col not in sub.columns:
            errors.append(f"Missing column '{col}' in submission")
    if id_column not in tpl.columns:
        errors.append(f"Missing column '{id_column}' in template")

    if errors:
        raise SubmissionValidationError("; ".join(errors))

    if len(sub) != expected_rows:
        errors.append(
            f"Expected {expected_rows} rows, got {len(sub)}"
        )

    if len(tpl) != expected_rows:
        errors.append(
            f"Template has {len(tpl)} rows, expected {expected_rows}"
        )

    sub_ids = sub[id_column].astype(str).tolist()
    tpl_ids = tpl[id_column].astype(str).tolist()
    if sub_ids != tpl_ids:
        errors.append("ID order does not match template.csv exactly")

    if sub[label_column].isna().any() or sub[id_column].isna().any():
        errors.append("Null/NaN values found in submission")

    labels = sub[label_column]
    try:
        as_float = labels.astype(float)
        if not (as_float == as_float.astype(int)).all():
            errors.append("Labels must be integers")
        label_ints = as_float.astype(int)
        if not label_ints.isin([0, 1, 2]).all():
            errors.append("Labels must be in {0, 1, 2}")
    except (ValueError, TypeError):
        errors.append("Labels must be integers in {0, 1, 2}")

    if errors:
        raise SubmissionValidationError("; ".join(errors))

    return {
        "ok": True,
        "rows": len(sub),
        "expected_rows": expected_rows,
        "id_column": id_column,
        "label_column": label_column,
    }
