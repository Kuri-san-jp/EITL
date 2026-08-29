#!/usr/bin/env python3
"""Recompute paired statistics for the reward-form contrasts (min vs pq_only / min vs mean) from the raw JSON.

CPU only. For scorer consistency we also report the subset of songs with
duration ≤300s (above 300s, the chunked scoring introduced on  and
full-forward scoring are mixed across runs).
After the SB rescoring of pqonly (sb_rescore), the sb values in songs/*.json are
updated, so simply re-running this script yields the final n=100 numbers.

Run: python3 experiments/recompute_rewardform_contrasts.py
Output: outputs/analysis/rewardform_contrasts.json + stdout
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "outputs" / "runs"
DUR_CSV = ROOT / "outputs" / "analysis" / "fig2_drift_song_durations.csv"

BASE = "the main random-proposer run"          # reward = min (proposed)
ARMS = ["the pqonly ablation run", "the mean ablation run", "the sbonly ablation run"]


def load_durations() -> dict[str, float]:
    durs: dict[str, float] = {}
    with open(DUR_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            durs[row["song_id"]] = float(row["dur_sec"])
    return durs


def song_metrics(run: str) -> dict[str, dict[str, float]]:
    """song_id -> {pq, sb} (mean over the 2 proposed_rand seeds, done songs only).

    A seed-count mismatch is treated as fail-fast. If the arm has 1 seed while the
    baseline has 2, the paired structure is broken, yet simply averaging would let
    it through without any warning (the same failure mode as the silent truncation
    that caused real damage before; raised in the  review).
    """
    out: dict[str, dict[str, float]] = {}
    for p in (RUNS / run / "songs").glob("*.json"):
        rec = json.loads(p.read_text("utf-8"))
        if rec.get("status") != "done":
            continue
        vals = [s["proposed_rand"] for s in rec.get("seeds", [])
                if s.get("proposed_rand")]
        if not vals:
            continue
        n_expect = int(rec.get("random_seeds") or len(rec.get("seeds", [])) or 0)
        if n_expect and len(vals) != n_expect:
            raise SystemExit(
                f"[contrast] seed count mismatch: {run}/{rec['song_id']} has "
                f"proposed_rand for only {len(vals)}/{n_expect} seeds."
                f" A partial re-run may not have been merged in, so aborting.")
        out[rec["song_id"]] = {
            "pq": sum(v["pq"] for v in vals) / len(vals),
            "sb": sum(v["sb"] for v in vals) / len(vals),
        }
    return out


def wilcoxon_p(diffs: list[float]) -> float:
    from scipy.stats import wilcoxon
    return float(wilcoxon(diffs, alternative="two-sided").pvalue)


def contrast(base: dict, arm: dict, songs: list[str], label: str) -> dict:
    d_pq = [arm[s]["pq"] - base[s]["pq"] for s in songs]
    d_sb = [arm[s]["sb"] - base[s]["sb"] for s in songs]
    res = {
        "label": label, "n": len(songs),
        "d_pq_mean": sum(d_pq) / len(d_pq),
        "d_sb_mean": sum(d_sb) / len(d_sb),
        "pq_win_arm": sum(x > 0 for x in d_pq) / len(d_pq),
        "sb_win_arm": sum(x > 0 for x in d_sb) / len(d_sb),
        "p_pq": wilcoxon_p(d_pq), "p_sb": wilcoxon_p(d_sb),
    }
    print(f"  {label:24} n={res['n']:3}  "
          f"ΔPQ(arm−min)={res['d_pq_mean']:+.4f} (p={res['p_pq']:.2e}, "
          f"arm win rate {res['pq_win_arm']:.0%})  "
          f"ΔSB={res['d_sb_mean']:+.4f} (p={res['p_sb']:.2e}, "
          f"arm win rate {res['sb_win_arm']:.0%})")
    return res


def main() -> int:
    import hashlib

    durs = load_durations()
    base = song_metrics(BASE)
    results = []
    for arm_name in ARMS:
        arm = song_metrics(arm_name)
        common = sorted(set(base) & set(arm))
        # Do not swallow a mismatch in the song sets (we once had an incident where
        # comparing mismatched song sets inflated the effect size by 2x).
        # If there is a difference, print the song names and warn.
        only_base, only_arm = sorted(set(base) - set(arm)), sorted(set(arm) - set(base))
        if only_base or only_arm:
            print(f"[contrast] WARN song sets differ: {BASE}-only {len(only_base)} songs "
                  f"{only_base[:3]}, {arm_name}-only {len(only_arm)} songs {only_arm[:3]}")
        # Check that songs missing from the duration CSV do not silently drop out
        # of the ≤300s subset
        missing_dur = [s for s in common if s not in durs]
        if missing_dur:
            print(f"[contrast] WARN duration unknown, excluded from the ≤300s set: {missing_dur[:3]}")
        short = [s for s in common if durs.get(s, 1e9) <= 300.0]
        print(f"[contrast] {arm_name} vs {BASE}")
        results.append(contrast(base, arm, common, "all-common"))
        results.append(contrast(base, arm, short, "≤300s (scorer-consistent)"))
        for r in results[-2:]:
            r["arm"] = arm_name
            # provenance: lets us verify later which song set the numbers came from
            r["song_set_sha256_12"] = hashlib.sha256(
                "\n".join(common).encode()).hexdigest()[:12]
            r["n_only_base"], r["n_only_arm"] = len(only_base), len(only_arm)
    out = ROOT / "outputs" / "analysis" / "rewardform_contrasts.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    print(f"[contrast] saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
