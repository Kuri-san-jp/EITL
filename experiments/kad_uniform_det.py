#!/usr/bin/env python3
"""Measure KAD/FAD for the 3 reward-form arms through one deterministic pipeline (GPU).

## Background (blocker from the  experiment code review)
The previous kad_eval.py builds embeddings with CLAP's rand_trunc (unseeded random
10 s crop), so it is non-deterministic: KAD wobbles by sd≈0.0012 even for bit-identical
input. The between-arm difference the paper needs (~0.001) is buried in that noise.

## How this script is made deterministic
- Each wav is converted to 48 kHz mono and split into consecutive 10 s windows
  (480000 samples; a tail shorter than 5 s is dropped), each window is CLAP-embedded
  and the embeddings are averaged. rand_trunc never fires (input is ≤10 s).
- The reference (MUSDB train, 50 songs, 20 s → 2 windows) is computed once in a single
  process and shared across all arms.
- Song-level bootstrap (B=500, fixed seed) gives 95% CIs for each arm's KAD and for the
  between-arm differences.
- random_dry is measured in all 3 runs and used to verify determinism
  (cross-run |ΔKAD| ≈ 0).

Run (GPU): via scripts/kad_uniform_det.sh.
Output: outputs/analysis/kad_uniform_det.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _common import enumerate_split, load_one           # noqa: E402
from kad_eval import _clap_embed_fn, _frechet, _mmd_gaussian  # noqa: E402

RUNS = ["the main random-proposer run", "the pqonly ablation run", "the mean ablation run"]
METHODS = ["proposed_rand", "random_dry"]
EXCLUDE = "Lushlife"
WIN = 480000          # 10 s @ 48 kHz
MIN_TAIL = 240000     # drop a trailing remainder shorter than 5 s
BOOT_B = 500
BOOT_SEED = 20260815


def canon(s: str) -> str:
    """Matching key for song_ids whose spelling varies across runs (the fxnorm/megami runs use raw MUSDB names)."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def det_embed_windows(embed_raw, audio, sr):
    """Deterministic embedding: convert to 48k mono -> consecutive 10 s windows -> embed each window -> average."""
    import torch, torchaudio
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    x = torch.from_numpy(np.asarray(mono, np.float32)).unsqueeze(0)
    if sr != 48000:
        x = torchaudio.functional.resample(x, sr, 48000)
    x = x.squeeze(0).numpy()
    n = len(x)
    wins = [x[i:i + WIN] for i in range(0, n - MIN_TAIL + 1, WIN)]
    if not wins:
        wins = [x]
    embs = [embed_raw(w[None, :], 48000) for w in wins]
    return np.mean(embs, axis=0)


def song_key(fname: str) -> str:
    m = re.match(r"^[a-z_]+__(.+)__seed\d+\.wav$", fname)
    return m.group(1) if m else fname


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="",
                    help="Comma-separated run:method pairs (e.g. the fxnorm baseline run:fxnorm). "
                         "If omitted, defaults to 3 runs x 2 methods.")
    ap.add_argument("--restrict-common", action="store_true",
                    help="Restrict the comparison to the song set common to all specs (matched by canon).")
    ap.add_argument("--songs-json", default="",
                    help="JSON array of song IDs; restrict to that set. "
                         "For re-measuring on the same song set as a past run (added . "
                         "Used to cross-check values against an arm whose wavs were deleted.")
    ap.add_argument("--out", default="kad_uniform_det.json",
                    help="Output file name under outputs/analysis/.")
    args = ap.parse_args()

    if args.specs:
        specs = [tuple(s.split(":", 1)) for s in args.specs.split(",") if s]
    else:
        specs = [(r, m) for r in RUNS for m in METHODS]

    np.random.seed(0)
    embed_raw = _clap_embed_fn()

    # --- reference (computed once, shared across all arms) ---
    ref = []
    for r in enumerate_split("train", tiers=None, limit=50):
        stems, sr = load_one(dataset="musdb18", song_id=r["song_id"],
                             duration_sec=20.0, split="train")
        mix = stems.get("mixture")
        if mix is None:
            mix = np.sum([v for k, v in stems.items() if k != "mixture"], axis=0)
        ref.append(det_embed_windows(embed_raw, np.asarray(mix, np.float32), sr))
    ref = np.array(ref)
    print(f"[det] reference n={len(ref)}", flush=True)

    # --- embeddings per spec (run:method), kept per song for the bootstrap ---
    import soundfile as sf
    by = {}  # (run, method) -> {canon_song: [emb, ...]}
    for run, method in specs:
        d = defaultdict(list)
        for f in sorted(glob.glob(str(ROOT / "outputs/runs" / run / "mixes"
                                      / f"{method}__*.wav"))):
            name = Path(f).name
            if EXCLUDE in name:
                continue
            audio, sr = sf.read(f, dtype="float32", always_2d=True)
            d[canon(song_key(name))].append(det_embed_windows(embed_raw, audio.T, sr))
        by[(run, method)] = d
        print(f"[det] {run}:{method} embeddings done ({len(d)} songs)", flush=True)

    # --- restrict to an explicitly given song set (for cross-checking against a past run) ---
    if args.songs_json:
        want = {canon(s_) for s_ in json.load(open(args.songs_json))}
        before = {k: len(v) for k, v in by.items()}
        by = {k: defaultdict(list, {s_: v for s_, v in d.items() if s_ in want})
              for k, d in by.items()}
        for k, d in by.items():
            if len(d) != len(want):
                raise SystemExit(
                    f"[det] {k} does not cover the requested song set: "
                    f"{len(d)}/{len(want)} songs (was {before[k]}). "
                    "Missing wavs are suspected, aborting.")
        print(f"[det] restricted to the requested song set: {len(want)} songs", flush=True)

    # --- restrict to the common song set (fair comparison when runs cover different songs) ---
    if args.restrict_common:
        common = set.intersection(*(set(d) for d in by.values()))
        print(f"[det] restricted to the common song set: {len(common)} songs", flush=True)
        by = {k: defaultdict(list, {s: v for s, v in d.items() if s in common})
              for k, d in by.items()}

    def kad_of(songs_dict, song_subset=None):
        songs = song_subset if song_subset is not None else sorted(songs_dict)
        X = np.array([e for s in songs for e in songs_dict[s]])
        return _mmd_gaussian(X, ref)

    rng = np.random.default_rng(BOOT_SEED)
    out = {"reference_n": len(ref), "boot_B": BOOT_B, "runs": {}}
    boots = {}
    for (run, method), d in sorted(by.items()):
        songs = sorted(d)
        point = kad_of(d)
        X = np.array([e for s in songs for e in d[s]])
        fad = _frechet(X, ref)
        # song-level bootstrap (rng re-initialized per group so every arm sees the same resamples)
        rs = np.random.default_rng(BOOT_SEED)
        bs = [kad_of(d, [songs[i] for i in rs.integers(0, len(songs), len(songs))])
              for _ in range(BOOT_B)]
        boots[(run, method)] = np.array(bs)
        out["runs"].setdefault(run, {})[method] = {
            "KAD": point, "FAD": fad, "n_songs": len(songs), "n_wavs": len(X),
            "KAD_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        }
        print(f"[det] {run:22} {method:14} KAD={point:.5f} "
              f"CI[{np.percentile(bs, 2.5):.5f},{np.percentile(bs, 97.5):.5f}] FAD={fad:.3f}",
              flush=True)

    # --- bootstrap CI for between-spec differences (paired bootstrap, only pairs with identical song sets) ---
    spec_keys = sorted(by)
    rs = np.random.default_rng(BOOT_SEED)
    diffs = {}
    comparable = [(a, b) for i, a in enumerate(spec_keys) for b in spec_keys[i + 1:]
                  if sorted(by[a]) == sorted(by[b])]
    if comparable:
        songs0 = sorted(by[comparable[0][0]])
        for _ in range(BOOT_B):
            idx = rs.integers(0, len(songs0), len(songs0))
            sub = [songs0[i] for i in idx]
            k = {sk: kad_of(by[sk], sub) for sk in spec_keys
                 if sorted(by[sk]) == songs0}
            for a, b in comparable:
                key = f"{a[0]}:{a[1]} - {b[0]}:{b[1]}"
                if k.get(a) is not None and k.get(b) is not None:
                    diffs.setdefault(key, []).append(k[a] - k[b])
    out["pairwise_diff_ci95"] = {
        key: {"mean": float(np.mean(v)),
              "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
              "excludes_zero": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0)}
        for key, v in diffs.items()}

    # --- determinism check: random_dry measured in several runs must agree ---
    dry = [out["runs"][r]["random_dry"]["KAD"] for r in {rk for rk, mk in by if mk == "random_dry"}
           if "random_dry" in out["runs"].get(r, {})]
    if len(dry) >= 2:
        out["determinism_check"] = {"random_dry_kads": dry,
                                    "max_abs_diff": float(max(dry) - min(dry))}
        print(f"[det] determinism check: random_dry KAD spread = {max(dry) - min(dry):.2e}", flush=True)
    for key, v in out["pairwise_diff_ci95"].items():
        print(f"[det] Δ {key}: {v['mean']:+.5f} CI{v['ci95']} "
              f"{'excludes 0 = significant' if v['excludes_zero'] else 'includes 0 = not significant'}", flush=True)

    (ROOT / "outputs/analysis" / args.out).write_text(json.dumps(out, indent=1), "utf-8")
    print(f"[det] saved: outputs/analysis/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
