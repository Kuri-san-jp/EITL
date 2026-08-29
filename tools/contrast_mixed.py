"""Evaluate an arbitrary 2-condition contrast with the same variance structure
as the main analysis .

The mixed model of pilot_analysis has the decomposition
SE^2 = tau_LC^2/L + tau_SC^2/S + sigma_d^2/(L*S_per)
(L=number of listeners, S=number of songs, S_per=songs per listener).
So that contrasts other than the pre-registered ones (H1-H3, S1) can be
evaluated with the same formula, we estimate the variance components from the
per-song / per-listener mean differences and report a one-sided p.

Sanity check: confirm that H2 (min vs pq_only) reproduces the SE=2.66 of the
main analysis.
"""
import json
import math
import os
import statistics as st
import sys

from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = os.path.join(ROOT, "experiments/subjective/pilot_v2")
S_PER = 9


def load():
    rep = json.load(open(os.path.join(
        ROOT, "outputs/analysis/main_interim/interim_main2")))
    kept = set(rep["kept"])
    priv = json.load(open(os.path.join(PILOT, "stimuli/manifest.json")))
    bmap, song_of = {}, {}
    for t in priv["trials"]:
        for cond, c in t["clips"].items():
            bmap[c["blind_id"]] = cond
            song_of[c["blind_id"]] = t["song_id"]
    rows = json.load(open(os.path.join(
        ROOT, "outputs/analysis/main_interim/db_dump.json")))
    recs = []          # (listener, song, {cond: score})
    for r in rows:
        if r.get("type") != "mushra_trial" or r["pid"] not in kept:
            continue
        pl = json.loads(r["payload"])
        pl = pl.get("payload") or pl
        conv, song = {}, None
        for key, v in (pl.get("ratings") or {}).items():
            bid = os.path.basename(str(key)).replace(".opus", "")
            if bid in bmap:
                conv[bmap[bid]] = float(v)
                song = song_of[bid]
        if len(conv) == 8:
            recs.append((r["pid"], song, conv))
    return recs


def contrast(recs, a, b):
    d = [(l, s, sc[a] - sc[b]) for l, s, sc in recs if a in sc and b in sc]
    mu = st.mean(x for _, _, x in d)
    L = len({l for l, _, _ in d})
    S = len({s for _, s, _ in d})
    by_l, by_s = {}, {}
    for l, s, x in d:
        by_l.setdefault(l, []).append(x)
        by_s.setdefault(s, []).append(x)
    tau_L = st.stdev([st.mean(v) for v in by_l.values()]) if L > 1 else 0.0
    tau_S = st.stdev([st.mean(v) for v in by_s.values()]) if S > 1 else 0.0
    resid = [x - st.mean(by_l[l]) - st.mean(by_s[s]) + mu for l, s, x in d]
    sig = st.stdev(resid) if len(resid) > 1 else 0.0
    se = math.sqrt(tau_L**2 / L + tau_S**2 / S + sig**2 / (L * S_PER))
    p1 = 1 - norm.cdf(abs(mu) / se) if se else float("nan")
    # per-listener paired comparison (the less conservative version, which treats
    # song as a fixed effect)
    lm = [st.mean(v) for v in by_l.values()]
    se_l = st.stdev(lm) / math.sqrt(len(lm))
    p_l = 2 * (1 - norm.cdf(abs(st.mean(lm)) / se_l))
    return mu, se, p1, se_l, p_l, L, S


def main() -> int:
    recs = load()
    print(f"data: {len(recs)} screens, listeners "
          f"{len({l for l, _, _ in recs})}, songs {len({s for _, s, _ in recs})}")
    print()
    print(f"{'contrast':26s} {'Δ':>6} {'mixedSE':>7} {'p_1side':>7} | "
          f"{'pairSE':>6} {'p_2side':>7}")
    pairs = [
        ("proposed", "pq_only", "min - PQ-only (H2)"),
        ("proposed", "sb_only", "min - SB-only"),
        ("proposed", "megami", "min - MEGAMI"),
        ("proposed", "fxnorm", "min - FxNorm (H3)"),
        ("proposed", "mean_reward", "min - mean (S1)"),
        ("proposed", "random_dry", "min - corrupted"),
        ("professional", "proposed", "professional - min"),
    ]
    for a, b, lbl in pairs:
        mu, se, p1, se_l, p_l, L, S = contrast(recs, a, b)
        mark = " *" if p1 < 0.05 else ""
        print(f"  {lbl:24s} {mu:+6.2f} {se:7.2f} {p1:7.4f}{mark} | "
              f"{se_l:6.2f} {p_l:7.4f}")
    print("\n* = mixed-model one-sided p<0.05 (before Holm correction)")
    print("mixedSE includes the song random effect (generalisation to unseen songs).")
    print("pairSE is per-listener (conditional on this set of songs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
