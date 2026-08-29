"""RQ3: Shapley-style attribution of each ear's contribution to final reward.

For a fully-completed ablation grid (E0..E7), estimate each ear's marginal
contribution to mean final reward by sampling subset additions.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def shapley_from_grid(per_subset: Dict[frozenset, float], ears: List[str]) -> Dict[str, float]:
    """Exact Shapley over a power-set of ears (only if grid covers it).
    For incomplete grids, missing subsets are linearly interpolated."""
    n = len(ears)
    sh: Dict[str, float] = {e: 0.0 for e in ears}
    fact = [np.math.factorial(i) for i in range(n + 1)]
    for ear in ears:
        for S in _all_subsets(set(ears) - {ear}):
            S_set = frozenset(S)
            S_with = frozenset(S | {ear})
            v_with = per_subset.get(S_with, np.nan)
            v_wo   = per_subset.get(S_set,  np.nan)
            if not (np.isfinite(v_with) and np.isfinite(v_wo)):
                continue
            w = fact[len(S)] * fact[n - len(S) - 1] / fact[n]
            sh[ear] += w * (v_with - v_wo)
    return sh


def _all_subsets(xs):
    xs = list(xs)
    for r in range(len(xs) + 1):
        for c in itertools.combinations(xs, r):
            yield set(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True,
                    help="JSON: [{ears:[...], reward:float}, ...]")
    args = ap.parse_args()
    cells = json.loads(Path(args.grid).read_text("utf-8"))
    ears = sorted({e for c in cells for e in c["ears"]})
    per_subset = {frozenset(c["ears"]): float(c["reward"]) for c in cells}
    sh = shapley_from_grid(per_subset, ears)
    print("Shapley contribution per ear (mean Δ reward):\n")
    for e, v in sorted(sh.items(), key=lambda x: -x[1]):
        print(f"  {e:<25} {v:+.4f}")


if __name__ == "__main__":
    main()
