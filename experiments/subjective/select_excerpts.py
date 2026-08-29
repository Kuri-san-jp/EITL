"""Automatically select excerpt segments for the listening test from the dry
stems of MUSDB18-HQ.

CPU only. It never uses the GPU (run it directly, without `the cluster job wrapper`).

--------------------------------------------------------------------------
Design
--------------------------------------------------------------------------
Perceptual differences in mixing quality are largest in segments where all
stems sound simultaneously (balance, masking and spatial placement can all be
judged there). Conversely, in segments where only the vocal sounds, or in the
intro/outro, differences between conditions cannot appear in principle.

So for a window w of length L seconds, define the presence of each stem k as

    a_k(w) = mean_{t in w} 20*log10(RMS_k(t)) - p95_k         [dB, <= 0]

    where p95_k = the 95th percentile of the frame RMS of stem k over the whole
    song (= the reference level at which that stem is "clearly sounding")

and use **the weakest stem** as the objective:

    score(w) = min_k a_k(w) - lambda * std_{t in w}[ 20*log10(RMS_mix(t)) ]

Taking the min follows the same idea as reward = min(z_PQ, z_SB): it rejects
windows in which even one stem is missing. The second term penalizes dynamic
variation within the window (structural boundaries, fades) and picks stable
segments that are easy for non-experts to judge.

In addition:
  * candidate windows are restricted to [head_frac*dur, (1-tail_frac)*dur - L]
    (excluding intro/outro/fade)
  * the start of the best window is snapped to the librosa beat grid (a
    musically natural start)
  * the window is **decided from the dry stems, so it is identical for all
    conditions regardless of condition / seed**

--------------------------------------------------------------------------
Output
--------------------------------------------------------------------------
JSON (--out):
  {"config": {...},
   "songs": [{"song_id", "split", "duration_sec", "start_sec", "end_sec",
              "beat_snapped", "score_db", "min_stem_db", "stem_db": {...},
              "mix_rms_std_db", "n_active_stems", "vocal_active",
              "excluded", "exclude_reason"}, ...]}

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
  # 1) enumerate the CRN-matched pool (also verifies which conditions really
  #    exist in which run)
  python3 experiments/subjective/select_excerpts.py --list-pool

  # 2) select the excerpt segments (analysis only; no audio is written)
  python3 experiments/subjective/select_excerpts.py \
      --pool-from-runs --limit 0 --length 12.0 \
      --out outputs/analysis/subjective_excerpts.json

  # 3) render the stimuli from the condition wavs that exist, using the
  #    selection JSON
  python3 experiments/subjective/select_excerpts.py --render \
      --excerpts outputs/analysis/subjective_excerpts.json \
      --songs-file <selected_songs.txt> \
      --stimuli-dir experiments/subjective/stimuli
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MUSDB = ROOT / "data" / "musdb18hq"
STEMS = ("bass", "drums", "other", "vocals")

# The (condition -> run, wav prefix) map needed to span the paper's main
# contrasts. That the CRN (common random numbers) agree across runs is verified
# separately.
CONDITION_SOURCES: Dict[str, Tuple[str, str]] = {
    "proposed":   ("the main random-proposer run", "proposed_rand"),
    "random_dry": ("the main random-proposer run", "random_dry"),
    "pq_only":    ("the pqonly ablation run", "proposed_rand"),
    "mean_reward": ("the mean ablation run", "proposed_rand"),
    "fxnorm":     ("the fxnorm baseline run", "fxnorm"),
    "megami":     ("the megami baseline run", "megami"),
    # pro_mix does not depend on the corruption seed (it is the original MUSDB18
    # mixture itself), so it is generated directly from data/musdb18hq rather
    # than from a run.
    "pro_mix":    ("__musdb_mixture__", "pro_mix"),
}
# The runs for which CRN parity is required (pro_mix excluded)
CRN_RUNS = ["the main random-proposer run", "the pqonly ablation run", "the mean ablation run",
            "the fxnorm baseline run", "the megami baseline run"]
BASE_RUN = "the main random-proposer run"


# ----------------------------------------------------------------------
# run metadata / CRN verification
# ----------------------------------------------------------------------
def load_run_songs(run: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for f in glob.glob(str(ROOT / "outputs" / "runs" / run / "songs" / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:                                   # noqa: BLE001
            continue
        if d.get("status") != "done":
            continue
        out[d["song_id"]] = d
    return out


def crn_identical(a: dict, b: dict, n_seeds: int = 2) -> bool:
    """Strictly decide whether the same song in two runs is the same corruption
    realization."""
    try:
        for i in range(n_seeds):
            sa, sb = a["seeds"][i], b["seeds"][i]
            if sa["seed"] != sb["seed"]:
                return False
            if sa.get("common_gain") is None or sb.get("common_gain") is None:
                return False
            if abs(sa["common_gain"] - sb["common_gain"]) > 1e-9:
                return False
            for k in sa["stem_gains"]:
                if abs(sa["stem_gains"][k] - sb["stem_gains"][k]) > 1e-12:
                    return False
            if abs(sa["random_dry"]["pq"] - sb["random_dry"]["pq"]) > 1e-6:
                return False
            if a.get("normalize_corrupted_lufs") != b.get("normalize_corrupted_lufs"):
                return False
    except (KeyError, IndexError, TypeError):
        return False
    return True


def mix_path(run: str, prefix: str, song_id: str, seed: int) -> Optional[Path]:
    """Find a mix wav inside a run. Depending on the run, the naming uses either
    spaces or underscores."""
    d = ROOT / "outputs" / "runs" / run / "mixes"
    for sid in (song_id, song_id.replace(" ", "_").replace("'", "_")):
        p = d / f"{prefix}__{sid}__seed{seed}.wav"
        if p.exists():
            return p
    # Last resort: loose match with glob
    stem = song_id.replace(" ", "?").replace("'", "?")
    hits = sorted(d.glob(f"{prefix}__{stem}__seed{seed}.wav"))
    return hits[0] if hits else None


def build_pool(verbose: bool = True) -> List[dict]:
    """Return the pool of songs whose CRN agree across all runs and for which the
    wavs of every required condition really exist."""
    runs = {r: load_run_songs(r) for r in CRN_RUNS}
    base = runs[BASE_RUN]
    pool: List[dict] = []
    reasons: Dict[str, int] = {}

    def bump(k: str) -> None:
        reasons[k] = reasons.get(k, 0) + 1

    for song_id, meta in sorted(base.items()):
        ok = True
        for r in CRN_RUNS[1:]:
            if song_id not in runs[r]:
                bump(f"missing_in_{r}"); ok = False; break
            if not crn_identical(meta, runs[r][song_id]):
                bump(f"crn_break_{r}"); ok = False; break
        if not ok:
            continue
        # check that the wavs exist
        missing = []
        for cond, (run, prefix) in CONDITION_SOURCES.items():
            if run == "__musdb_mixture__":
                if not find_song_dir(song_id):
                    missing.append(cond)
                continue
            for seed in (0, 1):
                if mix_path(run, prefix, song_id, seed) is None:
                    missing.append(f"{cond}@seed{seed}")
        if missing:
            bump("missing_wav"); continue
        pool.append({"song_id": song_id, "split": meta["split"]})

    if verbose:
        print(f"[pool] base run {BASE_RUN}: {len(base)} done songs", file=sys.stderr)
        for k, v in sorted(reasons.items()):
            print(f"[pool]   dropped {v:3d}  ({k})", file=sys.stderr)
        print(f"[pool] usable pool: {len(pool)} songs", file=sys.stderr)
    return pool


def find_song_dir(song_id: str) -> Optional[Path]:
    for split in ("train", "test"):
        p = MUSDB / split / song_id
        if p.is_dir():
            return p
    return None


# ----------------------------------------------------------------------
# excerpt selection
# ----------------------------------------------------------------------
def frame_rms_db(x: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> np.ndarray:
    """Return the frame RMS of a mono signal in dBFS."""
    win = max(1, int(round(win_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 0:
        return np.array([-120.0])
    # O(N) via a cumulative sum of squares
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange(n) * hop
    e = (cs[idx + win] - cs[idx]) / win
    return 10.0 * np.log10(np.maximum(e, 1e-14))


def read_mono(path: Path, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Read a stereo wav as mono, decimated by an integer factor to near
    target_sr (fast, and no librosa resample needed)."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    m = x.mean(axis=1)
    step = max(1, int(round(sr / target_sr)))
    if step > 1:
        # Anti-aliasing is unnecessary for an RMS envelope (even if out-of-band
        # energy folds back, the ordering of the envelope is essentially
        # unchanged). Swap in scipy.signal.decimate if strictness is needed.
        m = m[::step]
        sr = sr // step
    return m, sr


def analyse_song(song_id: str, split: str, length: float,
                 hop_sec: float = 0.25, win_sec: float = 0.5,
                 head_frac: float = 0.15, tail_frac: float = 0.12,
                 lam: float = 0.25, activity_floor_db: float = -18.0,
                 beat_snap: bool = True, max_end_sec: float = 240.0) -> dict:
    """max_end_sec: constrain the end of the excerpt not to exceed this.

    The FxNorm output is **cut off at 240 s** (measured: truncated for 48/94
    songs). Since the same segment has to be cut for every condition, candidate
    windows are capped at the length common to all conditions.
    """
    d = find_song_dir(song_id)
    if d is None:
        return {"song_id": song_id, "split": split, "excluded": True,
                "exclude_reason": "musdb_dir_not_found"}

    stem_db: Dict[str, np.ndarray] = {}
    p95: Dict[str, float] = {}
    sr_ref = None
    for k in STEMS:
        p = d / f"{k}.wav"
        if not p.exists():
            return {"song_id": song_id, "split": split, "excluded": True,
                    "exclude_reason": f"missing_stem_{k}"}
        m, sr = read_mono(p)
        sr_ref = sr if sr_ref is None else sr_ref
        db = frame_rms_db(m, sr, win_sec, hop_sec)
        stem_db[k] = db
        p95[k] = float(np.percentile(db, 95))

    mix_m, sr_mix = read_mono(d / "mixture.wav")
    mix_db = frame_rms_db(mix_m, sr_mix, win_sec, hop_sec)
    dur = len(mix_m) / sr_mix

    n_fr = min(min(len(v) for v in stem_db.values()), len(mix_db))
    for k in STEMS:
        stem_db[k] = stem_db[k][:n_fr]
    mix_db = mix_db[:n_fr]

    # Stem presence (dB, relative to that stem's p95. 0 = clearly sounding)
    rel = {k: np.minimum(stem_db[k] - p95[k], 0.0) for k in STEMS}

    w_fr = int(round(length / hop_sec))
    if n_fr <= w_fr + 2:
        return {"song_id": song_id, "split": split, "duration_sec": dur,
                "excluded": True, "exclude_reason": "too_short"}

    lo = int(round(head_frac * n_fr))
    hi = int(round((1.0 - tail_frac) * n_fr)) - w_fr
    if max_end_sec and max_end_sec > length:
        hi = min(hi, int(math.floor((max_end_sec - length) / hop_sec)))
    if hi <= lo:
        lo = 0
        hi = min(n_fr - w_fr,
                 int(math.floor((max_end_sec - length) / hop_sec))
                 if max_end_sec and max_end_sec > length else n_fr - w_fr)
    if hi <= lo:
        return {"song_id": song_id, "split": split, "duration_sec": dur,
                "excluded": True, "exclude_reason": "no_valid_window"}

    # Compute all window means at once with a moving average
    def movmean(v: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(v)])
        return (cs[w_fr:] - cs[:-w_fr]) / w_fr

    rel_mean = {k: movmean(rel[k]) for k in STEMS}
    mix_mean = movmean(mix_db)
    mix_sq = movmean(mix_db ** 2)
    mix_std = np.sqrt(np.maximum(mix_sq - mix_mean ** 2, 0.0))

    min_stem = np.min(np.vstack([rel_mean[k] for k in STEMS]), axis=0)
    score = min_stem - lam * mix_std

    cand = np.full_like(score, -1e9)
    cand[lo:hi] = score[lo:hi]
    best = int(np.argmax(cand))
    start = best * hop_sec

    beat_snapped = False
    if beat_snap:
        try:
            import librosa
            tempo, beats = librosa.beat.beat_track(
                y=mix_m.astype(np.float32), sr=sr_mix, hop_length=512, units="time")
            if len(beats):
                # Snap to the nearest beat within +/-0.6 s of the start position
                j = int(np.argmin(np.abs(beats - start)))
                if abs(beats[j] - start) <= 0.6 and beats[j] + length <= dur:
                    start = float(beats[j]); beat_snapped = True
        except Exception:                                   # noqa: BLE001
            pass

    stem_vals = {k: float(rel_mean[k][best]) for k in STEMS}
    n_active = sum(1 for v in stem_vals.values() if v >= activity_floor_db)
    excluded, reason = False, ""
    if float(min_stem[best]) < activity_floor_db - 12.0:
        excluded, reason = True, (
            f"weakest_stem_too_quiet(min={float(min_stem[best]):.1f}dB)")
    if n_active < 3:
        excluded, reason = True, f"only_{n_active}_active_stems"

    return {
        "song_id": song_id, "split": split, "duration_sec": round(dur, 2),
        "start_sec": round(float(start), 3),
        "end_sec": round(float(start + length), 3),
        "beat_snapped": beat_snapped,
        "score_db": round(float(score[best]), 3),
        "min_stem_db": round(float(min_stem[best]), 3),
        "stem_db": {k: round(v, 2) for k, v in stem_vals.items()},
        "mix_rms_std_db": round(float(mix_std[best]), 3),
        "mix_rms_mean_db": round(float(mix_mean[best]), 2),
        "n_active_stems": n_active,
        "vocal_active": bool(stem_vals["vocals"] >= activity_floor_db),
        "excluded": excluded, "exclude_reason": reason,
    }


# ----------------------------------------------------------------------
# stimulus rendering (optional)
# ----------------------------------------------------------------------
def render_stimuli(excerpts: List[dict], songs: List[str], stimuli_dir: Path,
                   length: float, target_lufs: float, peak_ceiling_db: float,
                   fade_in_ms: float, fade_out_ms: float, seeds=(0,),
                   fmt: str = "flac") -> List[dict]:
    import soundfile as sf
    import pyloudnorm as pyln

    by_id = {e["song_id"]: e for e in excerpts}
    report: List[dict] = []
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    for song_id in songs:
        e = by_id.get(song_id)
        if e is None or e.get("excluded"):
            print(f"[render] skip {song_id} (no excerpt)", file=sys.stderr)
            continue
        outdir = stimuli_dir / song_id.replace(" ", "_").replace("'", "_")
        outdir.mkdir(parents=True, exist_ok=True)
        for cond, (run, prefix) in CONDITION_SOURCES.items():
            for seed in (seeds if run != "__musdb_mixture__" else (0,)):
                if run == "__musdb_mixture__":
                    src = find_song_dir(song_id)
                    src = None if src is None else src / "mixture.wav"
                else:
                    src = mix_path(run, prefix, song_id, seed)
                if src is None or not Path(src).exists():
                    print(f"[render] MISSING {cond} {song_id} seed{seed}", file=sys.stderr)
                    continue
                info = sf.info(str(src))
                s0 = int(round(e["start_sec"] * info.samplerate))
                n = int(round(length * info.samplerate))
                x, sr = sf.read(str(src), start=s0, frames=n,
                                dtype="float64", always_2d=True)
                if x.shape[0] < n:
                    x = np.pad(x, ((0, n - x.shape[0]), (0, 0)))
                meter = pyln.Meter(sr)
                lufs_pre = float(meter.integrated_loudness(x))
                # Re-normalize after excerpting (mandatory, since LUFS changes
                # per window)
                y = pyln.normalize.loudness(x, lufs_pre, target_lufs)
                pk = float(np.abs(y).max())
                ceiling = 10 ** (peak_ceiling_db / 20.0)
                clipped_db = 0.0
                if pk > ceiling:
                    clipped_db = 20 * math.log10(pk / ceiling)
                    y = y * (ceiling / pk)
                # fade
                fi = int(round(fade_in_ms * 1e-3 * sr)); fo = int(round(fade_out_ms * 1e-3 * sr))
                if fi > 0:
                    y[:fi] *= np.linspace(0, 1, fi)[:, None] ** 2
                if fo > 0:
                    y[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
                lufs_post = float(meter.integrated_loudness(y))
                out = outdir / f"{cond}__seed{seed}.{fmt}"
                sf.write(str(out), y, sr,
                         subtype="PCM_16" if fmt == "flac" else "PCM_24")
                report.append({"song_id": song_id, "cond": cond, "seed": seed,
                               "path": str(out.relative_to(ROOT)),
                               "lufs_pre": round(lufs_pre, 2),
                               "lufs_post": round(lufs_post, 2),
                               "peak_trim_db": round(clipped_db, 2)})
                print(f"[render] {song_id[:28]:28s} {cond:11s} s{seed} "
                      f"LUFS {lufs_pre:7.2f} -> {lufs_post:7.2f} "
                      f"(peak trim {clipped_db:.2f} dB)", file=sys.stderr)
    return report


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-pool", action="store_true",
                    help="print the pool of songs with matching CRN and existing wavs for every condition, then exit")
    ap.add_argument("--pool-from-runs", action="store_true",
                    help="take the analysis targets from the CRN-matched pool (default)")
    ap.add_argument("--songs-file", type=str, default=None,
                    help="file with one song_id per line (used instead of the pool)")
    ap.add_argument("--limit", type=int, default=0, help="first N songs only (0=all)")
    ap.add_argument("--length", type=float, default=12.0, help="excerpt length [s]")
    ap.add_argument("--head-frac", type=float, default=0.15)
    ap.add_argument("--tail-frac", type=float, default=0.12)
    ap.add_argument("--lam", type=float, default=0.25,
                    help="penalty coefficient on the RMS variation within the window")
    ap.add_argument("--activity-floor-db", type=float, default=-18.0,
                    help="lower bound, relative to p95, for treating a stem as active [dB]")
    ap.add_argument("--no-beat-snap", action="store_true")
    ap.add_argument("--max-end-sec", type=float, default=240.0,
                    help="upper bound on the excerpt end [s]. Default 240 because "
                         "the FxNorm output is truncated at 240 s. 0 disables it.")
    ap.add_argument("--out", type=str,
                    default="outputs/analysis/subjective_excerpts.json")
    # render
    ap.add_argument("--render", action="store_true", help="write out the stimulus wavs")
    ap.add_argument("--excerpts", type=str, default=None,
                    help="for --render: an existing excerpt JSON")
    ap.add_argument("--stimuli-dir", type=str,
                    default="experiments/subjective/stimuli")
    ap.add_argument("--target-lufs", type=float, default=-16.0,
                    help="common loudness target after excerpting [LUFS]")
    ap.add_argument("--peak-ceiling-db", type=float, default=-1.0)
    ap.add_argument("--fade-in-ms", type=float, default=20.0)
    ap.add_argument("--fade-out-ms", type=float, default=150.0)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--format", type=str, default="flac", choices=["flac", "wav"])
    args = ap.parse_args()

    if args.list_pool:
        pool = build_pool()
        import collections
        print(json.dumps({"n": len(pool),
                          "by_split": dict(collections.Counter(p["split"] for p in pool)),
                          "songs": pool}, indent=1, ensure_ascii=False))
        return 0

    if args.render:
        if not args.excerpts:
            print("--render requires --excerpts", file=sys.stderr); return 2
        data = json.load(open(args.excerpts))
        excerpts = data["songs"]
        if args.songs_file:
            songs = [l.strip() for l in open(args.songs_file) if l.strip()
                     and not l.startswith("#")]
        else:
            songs = [e["song_id"] for e in excerpts if not e.get("excluded")]
        seeds = tuple(int(s) for s in args.seeds.split(","))
        rep = render_stimuli(excerpts, songs, ROOT / args.stimuli_dir,
                             data["config"]["length"], args.target_lufs,
                             args.peak_ceiling_db, args.fade_in_ms,
                             args.fade_out_ms, seeds, args.format)
        outp = ROOT / args.stimuli_dir / "render_report.json"
        json.dump({"config": vars(args), "files": rep}, open(outp, "w"), indent=1)
        print(f"[render] wrote {len(rep)} files, report -> {outp}", file=sys.stderr)
        return 0

    if args.songs_file:
        ids = [l.strip() for l in open(args.songs_file) if l.strip()
               and not l.startswith("#")]
        base = load_run_songs(BASE_RUN)
        targets = [{"song_id": s, "split": base.get(s, {}).get("split", "?")}
                   for s in ids]
    else:
        targets = build_pool()
    if args.limit:
        targets = targets[:args.limit]

    rows: List[dict] = []
    for i, t in enumerate(targets, 1):
        r = analyse_song(t["song_id"], t["split"], args.length,
                         head_frac=args.head_frac, tail_frac=args.tail_frac,
                         lam=args.lam, activity_floor_db=args.activity_floor_db,
                         beat_snap=not args.no_beat_snap,
                         max_end_sec=args.max_end_sec)
        rows.append(r)
        flag = "EXCL" if r.get("excluded") else "    "
        print(f"[{i:3d}/{len(targets)}] {flag} {t['song_id'][:38]:38s} "
              f"{r.get('start_sec', -1):7.2f}s  min_stem={r.get('min_stem_db', 0):6.1f}dB "
              f"act={r.get('n_active_stems', 0)} {r.get('exclude_reason', '')}",
              file=sys.stderr, flush=True)

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": {"length": args.length, "hop_sec": 0.25, "win_sec": 0.5,
                          "head_frac": args.head_frac, "tail_frac": args.tail_frac,
                          "lam": args.lam,
                          "activity_floor_db": args.activity_floor_db,
                          "beat_snap": not args.no_beat_snap,
                          "max_end_sec": args.max_end_sec,
                          "conditions": {k: list(v) for k, v in CONDITION_SOURCES.items()}},
               "songs": rows}, open(outp, "w"), indent=1, ensure_ascii=False)

    n_ok = sum(1 for r in rows if not r.get("excluded"))
    print(f"\n[done] {n_ok}/{len(rows)} songs usable -> {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
