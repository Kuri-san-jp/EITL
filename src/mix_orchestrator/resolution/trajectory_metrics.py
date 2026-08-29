"""Trajectory-level descriptors over per-window scores."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def summarize_trajectory(values: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float32)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": 0.0, "drift": 0.0, "p95_minus_p10": 0.0}
    drift = float(arr[-1] - arr[0]) if len(arr) >= 2 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "drift": drift,
        "p95_minus_p10": float(np.percentile(arr, 95) - np.percentile(arr, 10)),
    }
