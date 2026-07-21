#!/usr/bin/env python
"""Predict entrypoint: python scripts/predict.py --config configs/default.yaml"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waste_classification.cli import predict_main

if __name__ == "__main__":
    predict_main()
