"""Recompute the "pre/post search change" per reward form in z-calibrated units .

Response to reviewer comments (new codex/fugu review):
  - the Δ in Table 1 of the paper is "form − min", not a pre/post search difference
  - the raw differences in PQ and SB are on different scales and cannot be compared
  - to support "loss matches or exceeds the gain", pre->post has to be recomputed
    in the units of the per-song calibration (the z of Eq. (1))

Recovering the calibration (mu, sigma):
  The song JSON does not store mu/sigma, but the reward of the pq_only arm is
  exactly z_PQ = (pq - mu_PQ)/sigma_PQ, so it can be solved linearly from two
  (pq, reward) points of best_chain (likewise for sb_only). Even for runs
  without init_reward, the initial z can be recovered as
  best_chain[0].reward - best_chain[0].delta.
  The calibration pool is built deterministically from the song and should be
  shared across all arms -- verify this with reward of the min arm
  ≈ min(z_PQ, z_SB).

Definition of pre/post:
  pre  = the KB init state (= the initial state of the search). z recovered from init_reward.
  post = excerpt score of the last accepted proposal (pre if nothing was accepted).
  Both are in the search domain (12-second excerpt), i.e. the change in the
  region the reward lives in.

Output: outputs/analysis/zdelta_rewardform.json + stdout
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "min": "the main random-proposer run",
    "pq_only": "the pqonly ablation run",
    "mean": "the mean ablation run",
    "sb_only": "the sbonly ablation run",
}


def load_run(run):
    """song_id -> seed_idx -> {chain, init_reward}

    For runs where init_reward is not stored, recover it from best_chain[0]'s
    reward - delta (delta = the reward improvement at that step).
    A song-seed with zero acceptances cannot have its init recovered, so it is
    returned with chain=[].
    """
    out = {}
    for p in glob.glob(os.path.join(ROOT, "outputs/runs", run, "songs/*.json")):
        j = json.load(open(p))
        if j.get("status") != "done":
            continue
        seeds = []
        for s in j.get("seeds", []):
            pr = s.get("proposed_rand")
            if pr is None:
                continue
            chain = pr.get("best_chain") or []
            init = pr.get("init_reward")
            if init is None and chain:
                init = chain[0]["reward"] - chain[0]["delta"]
            seeds.append({"chain": chain, "init_reward": init})
        if seeds:
            out[j["song_id"]] = seeds
    return out


def fit_axis(log, axis):
    """Recover mu, sigma from (axis_value, reward) pairs by least squares.

    reward = (v - mu)/sigma is linear in v. Regress over all points for
    robustness against numerical error, and also return the maximum residual
    (used to verify that the calibration is shared).
    """
    xs = [r[axis] for r in log]
    ys = [r["reward"] for r in log]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    if slope <= 0:
        return None
    sigma = 1.0 / slope
    mu = mx - my * sigma
    resid = max(abs((x - mu) / sigma - y) for x, y in zip(xs, ys))
    return mu, sigma, resid


def main() -> int:
    data = {a: load_run(r) for a, r in RUNS.items()}
    songs = set.intersection(*(set(v) for v in data.values()))
    print(f"songs in common: {len(songs)}")

    rows = []
    bad_fit = 0
    for song in sorted(songs):
        n_seeds = min(len(data[a][song]) for a in RUNS)
        for si in range(n_seeds):
            # Recover the calibration (from the accepted chains of the single-axis arms)
            cp = data["pq_only"][song][si]
            cs = data["sb_only"][song][si]
            if len(cp["chain"]) < 2 or len(cs["chain"]) < 2:
                bad_fit += 1
                continue
            fp = fit_axis(cp["chain"], "pq")
            fs = fit_axis(cs["chain"], "sb")
            if not fp or not fs or fp[2] > 0.05 or fs[2] > 0.05:
                bad_fit += 1
                continue
            mu_p, sg_p, _ = fp
            mu_s, sg_s, _ = fs

            # Check: reward of the min arm ≈ min(z_pq, z_sb)
            ok = True
            for r in data["min"][song][si]["chain"][:10]:
                z = min((r["pq"] - mu_p) / sg_p, (r["sb"] - mu_s) / sg_s)
                if abs(z - r["reward"]) > 0.1:
                    ok = False
                    break
            if not ok:
                bad_fit += 1
                continue

            # The init state is shared by all arms (same corruption + KB init)
            pq0 = mu_p + sg_p * cp["init_reward"]
            sb0 = mu_s + sg_s * cs["init_reward"]
            for arm in RUNS:
                sd = data[arm][song][si]
                la = sd["chain"][-1] if sd["chain"] else None
                pq1 = la["pq"] if la else pq0
                sb1 = la["sb"] if la else sb0
                rows.append({
                    "song": song, "seed": si, "arm": arm,
                    "dz_pq": (pq1 - pq0) / sg_p,
                    "dz_sb": (sb1 - sb0) / sg_s,
                    "d_pq_raw": pq1 - pq0, "d_sb_raw": sb1 - sb0,
                })

    print(f"excluded (bad fit / calibration mismatch): {bad_fit} song-seed")
    print()
    summary = {}
    print(f"{'arm':8s} {'Δz_PQ':>8} {'Δz_SB':>8} | {'Δraw_PQ':>8} {'Δraw_SB':>8}  (pre->post search, excerpt domain)")
    for arm in RUNS:
        rs = [r for r in rows if r["arm"] == arm]
        m = {k: st.mean(r[k] for r in rs)
             for k in ("dz_pq", "dz_sb", "d_pq_raw", "d_sb_raw")}
        summary[arm] = {**m, "n": len(rs)}
        print(f"{arm:8s} {m['dz_pq']:+8.2f} {m['dz_sb']:+8.2f} | "
              f"{m['d_pq_raw']:+8.3f} {m['d_sb_raw']:+8.3f}  (n={len(rs)})")

    # Check the central claim: on the single-axis arms, is |loss| >= gain (in z units)?
    for arm, opt, unmon in (("pq_only", "dz_pq", "dz_sb"),
                            ("sb_only", "dz_sb", "dz_pq")):
        rs = [r for r in rows if r["arm"] == arm]
        gain = st.mean(r[opt] for r in rs)
        loss = st.mean(r[unmon] for r in rs)
        frac = sum(1 for r in rs if -r[unmon] >= r[opt]) / len(rs)
        print(f"\n{arm}: monitored axis +{gain:.2f}σ / unmonitored axis {loss:+.2f}σ "
              f"(ratio {-loss/gain:.2f}) | loss>=gain on {frac:.0%} of song-seeds")
        summary[arm]["loss_over_gain"] = -loss / gain
        summary[arm]["frac_loss_ge_gain"] = frac

    out = os.path.join(ROOT, "outputs/analysis/zdelta_rewardform.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
    print("\n->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
