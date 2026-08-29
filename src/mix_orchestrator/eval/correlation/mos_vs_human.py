"""RQ7: MOS predictors vs human ratings — Spearman / Pearson + plot.

Input: per-mix MOS scores (auto) + human MUSHRA mean scores.
Output: correlation table by ear, identifies best MOS for mixing.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ without scipy."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ra = _rank(a); rb = _rank(b)
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", required=True, help="JSON: {song_system_id: {ear: score}}")
    ap.add_argument("--human", required=True, help="JSON: {song_system_id: human_mean}")
    args = ap.parse_args()

    auto = json.loads(Path(args.auto).read_text("utf-8"))
    human = json.loads(Path(args.human).read_text("utf-8"))

    common = sorted(set(auto) & set(human))
    if len(common) < 5:
        print(f"WARN: only {len(common)} shared items — results unreliable")
    # collect per-ear vectors
    per_ear: Dict[str, List[float]] = defaultdict(list)
    h_arr: List[float] = []
    for k in common:
        h_arr.append(float(human[k]))
        for ear, s in auto[k].items():
            per_ear[ear].append(float(s))
    h_np = np.array(h_arr)

    print(f"\nMOS↔human correlation (n={len(common)})\n")
    print(f"{'ear':<28} {'spearman':<10} {'pearson':<10}")
    print("-" * 50)
    rows = []
    for ear, scores in per_ear.items():
        if len(scores) != len(common):
            continue
        a = np.array(scores)
        rho = spearman(a, h_np)
        pearson = float(np.corrcoef(a, h_np)[0, 1]) if a.std() > 0 else float("nan")
        rows.append((ear, rho, pearson))
    rows.sort(key=lambda r: -abs(r[1]) if not np.isnan(r[1]) else 0)
    for ear, rho, p in rows:
        print(f"{ear:<28} {rho:>8.3f}   {p:>8.3f}")


if __name__ == "__main__":
    main()
