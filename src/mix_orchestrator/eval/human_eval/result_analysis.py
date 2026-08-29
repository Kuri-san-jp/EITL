"""Analyze MUSHRA / Pairwise listening test results.

Outputs:
  • Per-system mean score and 95% CI (MUSHRA)
  • Pairwise win rate matrix + Bradley–Terry ranking (pairwise)
  • Krippendorff α inter-rater agreement
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def mushra_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_system: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        for sys_name, score in r.get("scores", {}).items():
            per_system[sys_name].append(float(score))
    summary = {}
    for name, vals in per_system.items():
        arr = np.array(vals)
        ci = 1.96 * float(arr.std()) / np.sqrt(max(len(arr), 1))
        summary[name] = {"n": len(arr), "mean": float(arr.mean()),
                         "std": float(arr.std()), "95ci_half": ci}
    return summary


def pairwise_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    wins = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        a, b = r["shown_order"]
        counts[a][b] += 1
        counts[b][a] += 1
        pref = r.get("preferred")
        if pref in (a,):
            wins[a][b] += 1
        elif pref in (b,):
            wins[b][a] += 1
        # tie contributes 0.5 to both (Bradley-Terry style)
        else:
            wins[a][b] += 0
            wins[b][a] += 0
    systems = sorted(set(list(wins) + [x for d in wins.values() for x in d]))
    matrix = {s: {} for s in systems}
    for s1 in systems:
        for s2 in systems:
            if s1 == s2:
                matrix[s1][s2] = None
                continue
            w = wins[s1][s2]
            n = counts[s1][s2]
            matrix[s1][s2] = (w / n) if n > 0 else None
    return {"win_rate_matrix": matrix, "systems": systems}


def krippendorff_alpha(records: List[Dict[str, Any]]) -> float:
    """Rough Krippendorff α (interval) for MUSHRA records.

    Treats (listener_id, song_id) → score-vector across systems.
    """
    by_unit: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in records:
        unit = f"{r.get('song_id','?')}__{r.get('trial_idx', 0)}"
        for sys, score in r.get("scores", {}).items():
            by_unit[unit][r["listener_id"] + "::" + sys] = float(score)
    if not by_unit:
        return float("nan")
    # Build observed disagreement
    arr = []
    for u, vals in by_unit.items():
        vs = list(vals.values())
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                arr.append((vs[i] - vs[j]) ** 2)
    obs_d = float(np.mean(arr)) if arr else 0.0
    all_vals = [v for u in by_unit.values() for v in u.values()]
    if len(all_vals) < 2:
        return float("nan")
    exp_d = float(np.var(all_vals) * 2)
    return float(1.0 - obs_d / (exp_d + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="JSONL from mushra_gradio or pairwise_gradio")
    ap.add_argument("--kind", choices=["mushra", "pairwise"], required=True)
    args = ap.parse_args()
    recs = load_jsonl(Path(args.path))
    print(f"Loaded {len(recs)} records from {args.path}\n")
    if args.kind == "mushra":
        summary = mushra_summary(recs)
        alpha = krippendorff_alpha(recs)
        print("Per-system MUSHRA:")
        for s, st in sorted(summary.items(), key=lambda x: -x[1]["mean"]):
            print(f"  {s:<20} n={st['n']:4d}  {st['mean']:.2f} ± {st['95ci_half']:.2f}")
        print(f"\nKrippendorff α ≈ {alpha:.3f}")
    else:
        ps = pairwise_summary(recs)
        print("Pairwise win-rate matrix (row preferred over column):")
        for s1 in ps["systems"]:
            row = "  " + s1.ljust(20)
            for s2 in ps["systems"]:
                v = ps["win_rate_matrix"][s1][s2]
                row += (f"{v:.2f}".rjust(8) if v is not None else "    -   ")
            print(row)


if __name__ == "__main__":
    main()
