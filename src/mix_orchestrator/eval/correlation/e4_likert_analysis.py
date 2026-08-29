"""Analyze E4 Likert ratings — per-system mean ± CI + inter-rater agreement."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings", help="ratings.jsonl from diagnosis_likert_gradio")
    args = ap.parse_args()
    recs = [json.loads(l) for l in Path(args.ratings).read_text("utf-8").splitlines() if l.strip()]
    by_system: Dict[str, List[int]] = defaultdict(list)
    for r in recs:
        if r.get("score") is None:
            continue
        by_system[r["system"]].append(int(r["score"]))

    print(f"\nE4 Likert results (n={len(recs)} ratings)\n")
    print(f"{'system':<24} {'n':<5} {'mean':<8} {'std':<8} {'95%CI':<12}")
    print("-" * 60)
    for s, vals in sorted(by_system.items(), key=lambda kv: -np.mean(kv[1])):
        arr = np.array(vals, dtype=np.float64)
        ci = 1.96 * arr.std() / np.sqrt(max(1, len(arr)))
        print(f"{s:<24} {len(arr):<5d} {arr.mean():<8.2f} {arr.std():<8.2f} ±{ci:<.2f}")

    # ---- Cohen's κ across pairs of raters on shared items ----
    by_rater_item: Dict[tuple, int] = {}
    for r in recs:
        if r.get("score") is None:
            continue
        by_rater_item[(r["rater_id"], r["item_id"])] = int(r["score"])
    raters = sorted({k[0] for k in by_rater_item})
    if len(raters) < 2:
        return
    print("\nPair-wise rater agreement (Spearman, on shared items):")
    for i in range(len(raters)):
        for j in range(i+1, len(raters)):
            shared = [(by_rater_item[(raters[i], it)], by_rater_item[(raters[j], it)])
                       for (rt, it) in by_rater_item if rt == raters[i]
                       and (raters[j], it) in by_rater_item]
            if len(shared) < 3:
                continue
            a = np.array([x[0] for x in shared])
            b = np.array([x[1] for x in shared])
            rho = float(np.corrcoef(np.argsort(np.argsort(a)),
                                    np.argsort(np.argsort(b)))[0, 1])
            print(f"  {raters[i]} ↔ {raters[j]} (n_shared={len(shared)})  ρ={rho:.3f}")


if __name__ == "__main__":
    main()
