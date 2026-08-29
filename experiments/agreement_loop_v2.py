"""Agreement-reward (R = min(z_PQ, z_SB)) driven mixing loop v2 — extended action space.

Step 2, first half : mechanism validation (Qwen is not used yet).
v1 (agreement_loop_smoke.py) only used gain/pan/EQ as perturbation candidates.
v2 draws candidates from the **full space of the 24 handlers** that the DSP
extension added to action_tools:

  static : gain / pan / width / EQ / compressor
  insert : reverb / delay / saturation / de-esser   <- new FX
  (section / master / automation / dynamic families need temporal structure or
   sidechain, so they are excluded from perturbation in this full-song 1-shot
   greedy search. They are handled at the LLM stage.)

Every FX stays inside the Field constraints in schemas.py:
  reverb : wet_level <= 0.5, dry_level <= 1.0, room/damp/width in [0,1], hpf <= 1000
  delay  : delay_seconds <= 1.0, feedback <= 0.6, mix <= 0.5
  saturation : drive_db <= 12
  de-esser   : center 5-9 kHz, range_db <= 12 (proposed for vocals only)

Pipeline (identical to v1):
  1. Initial mix = knowledge_base_mix with eq=False (LUFS balance only).
  2. per-song calibration: score {dry-sum, P2(analysis_driven), initial KB mix
     (eq=False), a few random candidates (FX included)} with PQ and SongBench
     -> z reference (fit_calibration).
  3. eval-gated greedy search (25-30 candidates). Stack 1 action onto the best
     state, render the full song -> normalize_for_eval -> PQ + SB ->
     reward = min(z_PQ, z_SB).
  4. Accept only candidates that improved the reward.
  5. Save the final mix after acceptance and the initial mix. Save per-stem
     LUFS, per-candidate timings, and per-action-type proposal/acceptance counts.

Differences from v1:
  - Proposal space extended to 9 kinds (gain/pan/width/eq/comp +
    reverb/delay/saturation/deesser).
  - Per-action-type proposed/accepted/rejected counts are aggregated and logged.
  - The extra render cost of the new FX is observed individually (per-candidate
    time broken down by action kind).
  - **partial saving**: the log JSON is overwritten after every scored candidate
    (partial results survive a timeout or a crash mid-run).

Run (GPU required; PQ=audiobox and SongBench=MuQ are used in one process):
  IMAGE_NAME=songbench-image:latest JOB_TIME=01:00:00 \
    scripts/the local job wrapper python experiments/agreement_loop_v2.py

Hard policy (memory: feedback_no_proxy_no_experiment): no proxies. If the real
PQ/SB models cannot be loaded, raise and stop.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# experiments/_common puts ROOT/src on sys.path and configures the HF cache.
from _common import ROOT, load_one  # noqa: E402

from mix_orchestrator.dsp.mix_state import MixState  # noqa: E402
from mix_orchestrator.dsp.renderer import render  # noqa: E402
from mix_orchestrator.dsp.loudness_norm import _measure_lufs, normalize_for_eval  # noqa: E402
from mix_orchestrator.tools import action_tools as A  # noqa: E402
from mix_orchestrator.strategies.knowledge_base_mix import (  # noqa: E402
    build_knowledge_base_mix, classify_role,
)


OUT_DIR = ROOT / "outputs" / "runs" / "agreement_loop_v2"


# ============================================================================
# Extended action proposal space (random proposals in place of an LLM)
# ----------------------------------------------------------------------------
# Out of the full 24 handlers, only the **per-stem static / insert** families
# that are meaningful for a full-song 1-shot greedy search are perturbed. Each
# proposal returns (handler_name, args). Ranges always stay inside the Field
# constraints in schemas.py.
# ============================================================================
_EQ_FREQS = [80.0, 120.0, 200.0, 300.0, 500.0, 800.0, 1500.0,
             3000.0, 5000.0, 8000.0, 11000.0]

# Proposal kinds and their probabilities. EQ stays somewhat heavy while the new
# FX get enough probability mass.
_ACTION_KINDS = ["gain", "pan", "width", "eq", "comp",
                 "reverb", "delay", "saturation", "deesser"]
_ACTION_PROBS = [0.14, 0.08, 0.08, 0.24, 0.10,
                 0.10, 0.08, 0.10, 0.08]


# ============================================================================
# gain automation on the random side (added 
# ----------------------------------------------------------------------------
# **Why it is needed (fairness)**: if the LLM side gets an automation tool while
# random can only play the 9 static kinds, only the LLM can make time-varying
# moves, and "LLM > random" could be explained by an **action-space difference**
# rather than a capability difference. The comparison only holds once random has
# the same action space.
#
# **Default is off**. When off, this block is never touched and the behaviour of
# ``_propose_action`` is **bit-identical** to existing runs
# (tools/check_propose_action_bitexact.py checks the proposal sequence, the rng
# state, and the kind distribution).
#
#   off  : the 9 static kinds only, as before
#   gain : adds apply_gain_automation as a 10th kind
#
# ----------------------------------------------------------------------------
# How breakpoints are built, and why
# ----------------------------------------------------------------------------
# 1. **Beat handling is mechanically tied to the LLM side (this is the core of
#    the fairness argument).**
#    The LLM gets a beat grid in its prompt only when ``--excerpt-beats`` is
#    passed (the beat times that agreement_loop_all.excerpt_beat_times detects
#    with librosa, in seconds relative to the start of the excerpt). So random
#    quantizes to the same grid **only under the same flag**:
#
#      without --excerpt-beats : neither LLM nor random knows the beats
#                                -> times are U(0, D)
#      with --excerpt-beats    : LLM and random hold the same beat times
#                                -> snap to the nearest beat
#
#    The grid is passed in by the caller (agreement_loop_all._propose_pipeline)
#    from ``args.excerpt_beat_times_rel``. It is **the same array that goes into
#    the LLM prompt**, so a state where only one side knows the beats cannot
#    arise by construction. In both modes the rng consumption is the same
#    (n_int draws of U(0, D)); quantization acts only as a downstream mapping.
#
# 2. **Both endpoints are pinned to 0 dB.**
#    (a) It removes the overlap with static gain. With free endpoints you could
#        draw an envelope of roughly +3 dB across the whole span, which is a
#        degraded apply_static_gain. Accepting it would not be evidence that
#        "automation worked".
#    (b) It makes the transfer back in excerpt mode meaningful. np.interp clamps
#        out-of-range values to the endpoint values, so with 0 dB at both ends
#        there is zero gain change outside the excerpt window, i.e. "the
#        envelope was applied only to the 12 s that were scored". With non-zero
#        endpoints a constant gain would remain over the 288 s outside the
#        window, and the full song would sound different from what was measured
#        during the search.
#
# 3. **1-3 interior breakpoints, with times drawn U(0, D) and sorted ascending.**
#    D = length of the search domain (12 s in excerpt mode). With 1 point you get
#    a peak/dip, with 2-3 a swell or a two-stage contour. Times are continuous to
#    match the fact that the LLM can write any time it likes (rounding to a grid
#    would reduce only random's expressiveness).
#
# 4. **Gain range is U(-4, +4) dB.** Same range as apply_static_gain (the same
#    values as the existing _ACTION_KINDS 'gain' branch). The maximum deviation
#    of the envelope never exceeds the maximum deviation of static gain, so the
#    confound "automation won because it can move things further" does not enter.
# ============================================================================
RANDOM_AUTOMATION_ENV = "AUTOMIX_RANDOM_AUTOMATION"
RANDOM_AUTOMATION_MODES = ("off", "gain")

#: Probability assigned to the automation kind. The existing 9 kinds are merely
#: scaled by (1 - this value), so **the relative ratios among the 9 are exactly
#: preserved** (the design of e.g. a heavy eq share is just thinned uniformly).
_AUTOMATION_P = 0.10
_AUTOMATION_KIND = "gain_auto"
#: Range of the interior breakpoint count [lo, hi) — passed to rng.integers. 1-3 points.
_AUTOMATION_N_INTERIOR = (1, 4)
#: Gain range of the interior breakpoints [dB]. Same as apply_static_gain.
_AUTOMATION_GAIN_DB = (-4.0, 4.0)


def _mix_in_probs(base: List[float], p_new: float) -> List[float]:
    """Assign probability ``p_new`` to a new kind, keeping the relative ratios of the existing kinds.

    ``rng.choice(p=...)`` raises ValueError if the sum is not exactly 1 in
    floating point (atol=sqrt(eps)), so normalize by the total at the end.
    """
    scaled = [float(p) * (1.0 - float(p_new)) for p in base] + [float(p_new)]
    s = float(sum(scaled))
    return [p / s for p in scaled]


_ACTION_KINDS_GAINAUTO = _ACTION_KINDS + [_AUTOMATION_KIND]
_ACTION_PROBS_GAINAUTO = _mix_in_probs(_ACTION_PROBS, _AUTOMATION_P)


def random_automation_mode(mode: Optional[str] = None) -> str:
    """Automation mode of the random proposer. Defaults to the env var, or 'off' if unset."""
    v = (mode if mode is not None
         else (os.environ.get(RANDOM_AUTOMATION_ENV) or "off"))
    v = str(v).strip().lower()
    if v not in RANDOM_AUTOMATION_MODES:
        raise ValueError(f"unknown {RANDOM_AUTOMATION_ENV}={v!r}; "
                         f"valid={RANDOM_AUTOMATION_MODES}")
    return v


def _interior_beat_grid(beat_times: Optional[List[float]],
                        dur: float) -> "np.ndarray":
    """Return, as an ascending array, only the beat times strictly inside the excerpt, i.e. not colliding with the endpoints (0, dur).

    The excerpt window is cut snapped to the beat grid, so ``beat_times[0]`` is
    close to 0.0. Keeping it as an interior breakpoint candidate would create a
    point at the same time as the pinned endpoint (0, 0 dB), and np.interp would
    produce a vertical step (= a click). Exclude it.
    """
    if not beat_times:
        return np.empty(0, dtype=np.float64)
    b = np.asarray([float(t) for t in beat_times], dtype=np.float64)
    b = b[(b > 0.0) & (b < dur)]
    return np.sort(b)


def _propose_gain_automation(rng: np.random.Generator, tracks: List[str],
                             duration_sec: Optional[float],
                             beat_times: Optional[List[float]] = None
                             ) -> Tuple[str, Dict[str, Any]]:
    """Propose one gain envelope with both endpoints pinned to 0 dB.

    The rng consumption order is fixed as track -> point count -> times -> gains
    ("track first", aligned with the other kinds). The presence of
    ``beat_times`` **does not change the rng consumption** (quantization is a
    mapping applied after the draws, so the with-beats and without-beats
    conditions see the same sequence under the same seed = CRN is preserved).
    """
    if duration_sec is None or not (float(duration_sec) > 0.0):
        raise ValueError(
            f"{RANDOM_AUTOMATION_ENV} is not 'off', so duration_sec "
            f"(length of the search domain in seconds) is required (got {duration_sec!r}). "
            "The caller must pass max(v.shape[-1] for v in stems.values())/sr.")
    dur = float(duration_sec)
    track = _pick_track(rng, tracks)
    n_int = int(rng.integers(_AUTOMATION_N_INTERIOR[0], _AUTOMATION_N_INTERIOR[1]))
    ts = np.sort(rng.uniform(0.0, dur, size=n_int))
    gs = rng.uniform(_AUTOMATION_GAIN_DB[0], _AUTOMATION_GAIN_DB[1], size=n_int)

    grid = _interior_beat_grid(beat_times, dur)
    if grid.size:
        # Snap to the nearest beat. grid is ascending, so the snap is monotone
        # non-decreasing = the ordering is preserved.
        idx = np.abs(grid[None, :] - ts[:, None]).argmin(axis=1)
        ts = grid[idx]

    bp: List[Tuple[float, float]] = [(0.0, 0.0)]
    seen: set = set()
    for t, g in zip(ts, gs):
        tf = float(t)
        # Drop duplicate times collapsed by the snap (two points at the same
        # time would be a vertical step).
        if tf in seen or tf <= 0.0 or tf >= dur:
            continue
        seen.add(tf)
        bp.append((tf, float(g)))
    if len(bp) == 1:
        # All points collapsed onto the endpoints (e.g. no beat falls inside).
        # Keep one point at its pre-quantization time to avoid an identity
        # envelope (= wasting a move on a proposal that does nothing).
        bp.append((float(ts[0]) if 0.0 < float(ts[0]) < dur else dur * 0.5,
                   float(gs[0])))
    bp.append((dur, 0.0))
    return "apply_gain_automation", {"track": track, "breakpoints": bp}


def _pick_track(rng: np.random.Generator, tracks: List[str],
                prefer: List[str] | None = None) -> str:
    """If a role name listed in prefer exists among tracks, sample from those preferentially."""
    if prefer:
        cands = [t for t in tracks if classify_role(t) in prefer or t in prefer]
        if cands:
            return str(rng.choice(cands))
    return str(rng.choice(tracks))


def _propose_action(rng: np.random.Generator, tracks: List[str],
                    duration_sec: Optional[float] = None,
                    automation: Optional[str] = None,
                    beat_times: Optional[List[float]] = None
                    ) -> Tuple[str, Dict[str, Any]]:
    """Propose a single random action.

    By default (``automation`` = None and the env var
    ``AUTOMIX_RANDOM_AUTOMATION`` unset or 'off') this returns **only the 9
    static / insert kinds, as before**, and is bit-identical to existing runs
    down to the rng consumption order, the amount consumed, and the return value
    (checked by tools/check_propose_action_bitexact.py).

    Args:
        duration_sec: Length of the search domain (the excerpt in excerpt mode)
            in seconds. Only read when automation is enabled; never looked at
            when off.
        automation: 'off' / 'gain'. If None, the env var is read
            (:func:`random_automation_mode`).
        beat_times: Beat times in seconds, relative to the start of the excerpt.
            Pass **the same array that goes into the LLM prompt**
            (``args.excerpt_beat_times_rel`` in agreement_loop_all). If None or
            empty, no quantization is applied.
    """
    mode = random_automation_mode(automation)
    if mode == "off":
        kind = str(rng.choice(_ACTION_KINDS, p=_ACTION_PROBS))
    else:
        kind = str(rng.choice(_ACTION_KINDS_GAINAUTO, p=_ACTION_PROBS_GAINAUTO))
        if kind == _AUTOMATION_KIND:
            return _propose_gain_automation(rng, tracks, duration_sec,
                                            beat_times=beat_times)

    if kind == "gain":
        # Conservative -4..+4 dB trim (schemas: -18..+12)
        return "apply_static_gain", {
            "track": _pick_track(rng, tracks),
            "gain_db": float(rng.uniform(-4.0, 4.0))}

    if kind == "pan":
        return "apply_static_pan", {
            "track": _pick_track(rng, tracks),
            "pan": float(rng.uniform(-0.7, 0.7))}

    if kind == "width":
        # 0..2 (1=neutral). Widening other/drums and narrowing vocals/bass is the
        # usual search direction.
        return "apply_static_width", {
            "track": _pick_track(rng, tracks),
            "width": float(rng.uniform(0.7, 1.6))}

    if kind == "eq":
        freq = float(rng.choice(_EQ_FREQS))
        gain_db = float(rng.uniform(-5.0, 5.0))     # schemas: -12..+12
        q = float(rng.uniform(0.6, 2.0))            # schemas: >0..10
        return "apply_static_eq", {
            "track": _pick_track(rng, tracks),
            "freq": freq, "gain_db": gain_db, "q": q}

    if kind == "comp":
        # Light glue compression. schemas static: thr -60..0, ratio 1..20.
        return "apply_static_compressor", {
            "track": _pick_track(rng, tracks),
            "threshold_db": float(rng.uniform(-30.0, -10.0)),
            "ratio": float(rng.uniform(1.5, 4.0)),
            "attack_ms": float(rng.uniform(5.0, 30.0)),
            "release_ms": float(rng.uniform(80.0, 300.0)),
            "knee_db": 6.0}

    if kind == "reverb":
        # wet <= 0.5 anti-cheat. Natural to apply on vocals/other.
        return "apply_reverb", {
            "track": _pick_track(rng, tracks, prefer=["vocals", "other"]),
            "room_size": float(rng.uniform(0.3, 0.8)),
            "damping": float(rng.uniform(0.3, 0.7)),
            "wet_level": float(rng.uniform(0.10, 0.35)),   # <=0.5
            "dry_level": float(rng.uniform(0.7, 1.0)),
            "width": float(rng.uniform(0.7, 1.0)),
            "highpass_hz": float(rng.choice([0.0, 150.0, 300.0]))}

    if kind == "delay":
        # delay_seconds <=1.0, feedback <=0.6, mix <=0.5.
        return "apply_delay", {
            "track": _pick_track(rng, tracks, prefer=["vocals", "other"]),
            "delay_seconds": float(rng.uniform(0.12, 0.45)),
            "feedback": float(rng.uniform(0.1, 0.45)),
            "mix": float(rng.uniform(0.08, 0.30))}

    if kind == "saturation":
        # drive <=12. Adds light harmonics to drums/bass.
        return "apply_saturation", {
            "track": _pick_track(rng, tracks, prefer=["drums", "bass"]),
            "drive_db": float(rng.uniform(2.0, 9.0))}

    # deesser: sibilance band. Proposed for vocals only (meaningless elsewhere).
    return "apply_deesser", {
        "track": _pick_track(rng, tracks, prefer=["vocals"]),
        "center_hz": float(rng.uniform(6000.0, 8500.0)),  # 5000..9000
        "threshold_db": float(rng.uniform(-35.0, -20.0)),
        "ratio": float(rng.uniform(2.0, 6.0)),
        "range_db": float(rng.uniform(3.0, 10.0)),         # <=12
        "attack_ms": float(rng.uniform(0.5, 3.0)),
        "release_ms": float(rng.uniform(30.0, 120.0))}


def _apply_action(state: MixState, name: str, args: Dict[str, Any]) -> MixState:
    handler = A.ACTION_HANDLERS[name]
    return handler(state, args)


# ============================================================================
# per-stem LUFS observation (vocals/bass/drums/other)
# ============================================================================
def _per_stem_lufs(state: MixState, sr: int) -> Dict[str, Dict[str, float]]:
    from mix_orchestrator.strategies.knowledge_base_mix import _render_single_stem
    out: Dict[str, Dict[str, float]] = {}
    for name in state.stems:
        try:
            single = _render_single_stem(state, name)
            lufs = _measure_lufs(single, sr)
        except Exception:                                       # noqa: BLE001
            lufs = float("nan")
        out[name] = {"role": classify_role(name), "lufs": float(lufs)}
    return out


# ============================================================================
# scoring / render helpers (with timing)
# ============================================================================
def _score_candidate(pq_scorer, sb_ear, audio: np.ndarray, sr: int
                     ) -> Tuple[float, float, float, float]:
    t0 = time.perf_counter()
    pq = float(pq_scorer.score(audio, sr))
    t1 = time.perf_counter()
    sb = float(sb_ear.score(audio, sr))
    t2 = time.perf_counter()
    return pq, sb, (t1 - t0), (t2 - t1)


def _render_and_norm(state: MixState, sr: int,
                     target_lufs: float) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    raw = render(state)
    audio = normalize_for_eval(raw, sr, target_lufs=target_lufs)
    return audio, (time.perf_counter() - t0)


def _stat(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"),
                "total": 0.0, "n": 0}
    a = np.asarray(xs, float)
    return {"mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3), "total": round(float(a.sum()), 3),
            "n": int(a.size)}


def _save_log(log: Dict[str, Any]) -> None:
    """Atomically overwrite the log JSON (used for partial saving)."""
    tmp = OUT_DIR / "agreement_loop_v2_log.json.tmp"
    final = OUT_DIR / "agreement_loop_v2_log.json"
    tmp.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(final)


# ============================================================================
# main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musdb18")
    ap.add_argument("--split", default="train")
    ap.add_argument("--song-index", type=int, default=0,
                    help="index into the enumerated song list (default 0 = train song 0)")
    ap.add_argument("--song", default=None,
                    help="specify song_id directly (--song-index is ignored when given)")
    ap.add_argument("--duration-sec", type=float, default=0.0,
                    help="<=0 for the full song (default). A positive number of seconds shortens smoke runs.")
    ap.add_argument("--n-random-calib", type=int, default=5,
                    help="number of random candidates added to the calibration pool (FX included)")
    ap.add_argument("--n-search", type=int, default=28,
                    help="number of search candidates (25-30 recommended)")
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    wall_t0 = time.perf_counter()

    # ---- resolve song id ----
    song_id = args.song
    if song_id is None and args.dataset != "synthetic":
        from _common import enumerate_songs
        songs = enumerate_songs(args.dataset, limit=args.song_index + 1,
                                split=args.split)
        songs = [s for s in songs if s is not None]
        if not songs:
            from mix_orchestrator.data.loaders import list_songs
            songs = list_songs(args.dataset, split=args.split)
        if args.song_index >= len(songs):
            raise IndexError(f"song-index {args.song_index} >= {len(songs)} songs")
        song_id = songs[args.song_index]
    print(f"[loop-v2] dataset={args.dataset} split={args.split} song={song_id} "
          f"duration_sec={args.duration_sec}")

    # ---- load stems (full song) ----
    t0 = time.perf_counter()
    stems, sr = load_one(args.dataset, song_id=song_id,
                         duration_sec=args.duration_sec, seed=args.seed,
                         split=args.split)
    load_sec = time.perf_counter() - t0
    main_stems = {k: v for k, v in stems.items() if k != "mixture"}
    dur = max(v.shape[-1] for v in main_stems.values()) / sr
    print(f"[loop-v2] loaded {len(main_stems)} stems sr={sr} "
          f"duration={dur:.1f}s in {load_sec:.1f}s: {list(main_stems)}")

    # ---- scorers (real models; both loaded in one process) ----
    from mix_orchestrator.eval.audiobox_pq import AudioboxPQScorer
    from mix_orchestrator.ears.tier6_music_reward import SongBenchMixingEar
    from mix_orchestrator.eval.agreement_reward import (
        fit_calibration, compute_agreement, SongCalibration)
    pq_scorer = AudioboxPQScorer()       # use_proxy_if_missing=False (no proxies)
    sb_ear = SongBenchMixingEar()
    print("[loop-v2] scorers constructed (models lazy-load on first score)")

    # ========================================================================
    # 1. Initial mix = KB mix with eq=False (LUFS balance only)
    # ========================================================================
    t0 = time.perf_counter()
    init_raw, init_meta = build_knowledge_base_mix(main_stems, sr, eq=False)
    kb_build_sec = time.perf_counter() - t0
    init_state = MixState.initial_from_stems(main_stems, sr)
    for name, dec in init_meta["per_stem"].items():
        g = float(dec.get("gain_db", 0.0))
        if abs(g) > 1e-9:
            init_state = A.apply_static_gain(init_state, {"track": name, "gain_db": g})
    init_audio, _ = _render_and_norm(init_state, sr, args.target_lufs)
    init_lufs_per_stem = _per_stem_lufs(init_state, sr)
    print(f"[loop-v2] initial KB mix (eq=False) built in {kb_build_sec:.1f}s; "
          f"per-stem LUFS:")
    for nm, info in init_lufs_per_stem.items():
        print(f"        {nm:12s} role={info['role']:7s} LUFS={info['lufs']:.2f}")

    # ========================================================================
    # 2. per-song calibration pool
    # ========================================================================
    calib_audios: List[np.ndarray] = []
    calib_labels: List[str] = []

    dry_state = MixState.initial_from_stems(main_stems, sr)
    dry_audio, _ = _render_and_norm(dry_state, sr, args.target_lufs)
    calib_audios.append(dry_audio); calib_labels.append("dry_sum")

    try:
        import asyncio
        from mix_orchestrator.strategies.analysis_driven import build_analysis_driven_mix
        p2_raw, _ = asyncio.run(build_analysis_driven_mix(main_stems, sr))
        p2_audio = normalize_for_eval(p2_raw, sr, target_lufs=args.target_lufs)
        calib_audios.append(p2_audio); calib_labels.append("p2_analysis_driven")
    except Exception as ex:                                     # noqa: BLE001
        print(f"[loop-v2] WARN: P2 baseline skipped: {ex!r}")

    calib_audios.append(init_audio); calib_labels.append("kb_initial_eq_false")

    tracks = list(main_stems.keys())
    for i in range(args.n_random_calib):
        nm, aargs = _propose_action(rng, tracks)
        try:
            st = _apply_action(init_state, nm, aargs)
            au, _ = _render_and_norm(st, sr, args.target_lufs)
            calib_audios.append(au); calib_labels.append(f"rand_calib_{i}_{nm}")
        except Exception as ex:                                 # noqa: BLE001
            print(f"[loop-v2] WARN: calib rand {i} ({nm}) skipped: {ex!r}")

    print(f"[loop-v2] calibrating on {len(calib_audios)} pool members ...")
    calib_pq: List[float] = []
    calib_sb: List[float] = []
    calib_rows: List[Dict[str, Any]] = []
    calib_t0 = time.perf_counter()
    for label, au in zip(calib_labels, calib_audios):
        pq, sb, pq_s, sb_s = _score_candidate(pq_scorer, sb_ear, au, sr)
        calib_pq.append(pq); calib_sb.append(sb)
        calib_rows.append({"label": label, "pq": round(pq, 4), "sb": round(sb, 4),
                           "pq_sec": round(pq_s, 3), "sb_sec": round(sb_s, 3)})
        print(f"        {label:30s} PQ={pq:.4f} SB={sb:.4f} "
              f"(pq={pq_s:.2f}s sb={sb_s:.2f}s)")
    calib_total_sec = time.perf_counter() - calib_t0
    calib: SongCalibration = fit_calibration(calib_pq, calib_sb)
    print(f"[loop-v2] calibration: PQ mean={calib.pq_mean:.4f} std={calib.pq_std:.4f} | "
          f"SB mean={calib.sb_mean:.4f} std={calib.sb_std:.4f} | "
          f"n={calib.n_samples} ({calib_total_sec:.1f}s)")

    init_pq = calib_pq[calib_labels.index("kb_initial_eq_false")]
    init_sb = calib_sb[calib_labels.index("kb_initial_eq_false")]
    init_reward = compute_agreement(init_pq, init_sb, calib)
    print(f"[loop-v2] INITIAL reward = min(z_PQ={calib.z_pq(init_pq):.3f}, "
          f"z_SB={calib.z_sb(init_sb):.3f}) = {init_reward:.4f}")

    # ========================================================================
    # 3-4. Search (eval-gated greedy: accept only candidates that improved)
    # ========================================================================
    best_state = init_state
    best_reward = init_reward
    best_pq, best_sb = init_pq, init_sb
    accepted: List[Dict[str, Any]] = []
    cand_rows: List[Dict[str, Any]] = []
    render_secs: List[float] = []
    pq_secs: List[float] = []
    sb_secs: List[float] = []
    # proposed/accepted/rejected/invalid counts per action type
    by_kind: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"proposed": 0, "accepted": 0, "rejected": 0, "invalid": 0})
    # render cost per action type (to observe the extra cost of the new FX)
    render_by_kind: Dict[str, List[float]] = defaultdict(list)

    # log skeleton so partial results can be saved on every candidate
    log: Dict[str, Any] = {
        "schema": "agreement_loop_v2", "status": "running",
        "dataset": args.dataset, "split": args.split, "song_id": song_id,
        "duration_sec_arg": args.duration_sec, "actual_duration_sec": round(dur, 2),
        "sr": sr, "target_lufs": args.target_lufs, "seed": args.seed,
        "n_search": args.n_search, "n_random_calib": args.n_random_calib,
        "action_space": _ACTION_KINDS,
        "initial_mix": {
            "method": init_meta["method"], "eq": init_meta["eq"],
            "per_stem_decision": init_meta["per_stem"],
            "per_stem_lufs": init_lufs_per_stem,
            "pq": round(init_pq, 4), "sb": round(init_sb, 4),
            "reward": round(init_reward, 4),
        },
        "calibration": {
            "pq_mean": round(calib.pq_mean, 4), "pq_std": round(calib.pq_std, 4),
            "sb_mean": round(calib.sb_mean, 4), "sb_std": round(calib.sb_std, 4),
            "n_samples": calib.n_samples, "pool": calib_rows,
        },
        "candidates": cand_rows,
    }
    _save_log(log)

    search_t0 = time.perf_counter()
    for c in range(args.n_search):
        nm, aargs = _propose_action(rng, tracks)
        by_kind[nm]["proposed"] += 1
        try:
            cand_state = _apply_action(best_state, nm, aargs)
        except Exception as ex:                                 # noqa: BLE001
            by_kind[nm]["invalid"] += 1
            print(f"[cand {c:02d}] action {nm} rejected (invalid): {ex!r}")
            cand_rows.append({"cand": c, "action": nm, "args": aargs,
                              "invalid": True, "error": repr(ex)})
            _save_log(log)
            continue
        audio, render_sec = _render_and_norm(cand_state, sr, args.target_lufs)
        pq, sb, pq_s, sb_s = _score_candidate(pq_scorer, sb_ear, audio, sr)
        reward = compute_agreement(pq, sb, calib)
        render_secs.append(render_sec); pq_secs.append(pq_s); sb_secs.append(sb_s)
        render_by_kind[nm].append(render_sec)
        improved = reward > best_reward
        row = {
            "cand": c, "action": nm, "args": aargs,
            "pq": round(pq, 4), "sb": round(sb, 4),
            "z_pq": round(calib.z_pq(pq), 4), "z_sb": round(calib.z_sb(sb), 4),
            "reward": round(reward, 4),
            "improved": bool(improved),
            "render_sec": round(render_sec, 3),
            "pq_sec": round(pq_s, 3), "sb_sec": round(sb_s, 3),
        }
        cand_rows.append(row)
        tag = "ACCEPT" if improved else "reject"
        print(f"[cand {c:02d}] {nm:22s} {aargs} -> "
              f"PQ={pq:.3f} SB={sb:.3f} R={reward:.4f} "
              f"(best={best_reward:.4f}) [{tag}] "
              f"render={render_sec:.2f}s pq={pq_s:.2f}s sb={sb_s:.2f}s")
        if improved:
            by_kind[nm]["accepted"] += 1
            best_state = cand_state
            best_reward = reward
            best_pq, best_sb = pq, sb
            accepted.append(row)
        else:
            by_kind[nm]["rejected"] += 1
        _save_log(log)
    search_total_sec = time.perf_counter() - search_t0

    # ========================================================================
    # 5. Render and save the final mix
    # ========================================================================
    final_audio, _ = _render_and_norm(best_state, sr, args.target_lufs)
    final_lufs_per_stem = _per_stem_lufs(best_state, sr)

    import soundfile as sf
    sf.write(str(OUT_DIR / "initial_mix.wav"), init_audio.T, sr, subtype="FLOAT")
    sf.write(str(OUT_DIR / "final_mix.wav"), final_audio.T, sr, subtype="FLOAT")

    # ---- cost aggregation ----
    per_cand_sec = [r["render_sec"] + r["pq_sec"] + r["sb_sec"]
                    for r in cand_rows if "render_sec" in r]
    cost = {
        "render_sec": _stat(render_secs),
        "pq_sec": _stat(pq_secs),
        "sb_sec": _stat(sb_secs),
        "per_candidate_total_sec": _stat(per_cand_sec),
        "render_by_action_kind": {k: _stat(v) for k, v in render_by_kind.items()},
        "calibration_total_sec": round(calib_total_sec, 2),
        "search_total_sec": round(search_total_sec, 2),
        "stems_load_sec": round(load_sec, 2),
        "wall_total_sec": round(time.perf_counter() - wall_t0, 2),
    }
    per_song_loop_sec = calib_total_sec + search_total_sec
    cost["per_song_loop_sec"] = round(per_song_loop_sec, 2)
    cost["est_150_songs_hours"] = round(per_song_loop_sec * 150 / 3600.0, 2)

    # per-action-type counts (converted to a plain dict)
    by_kind_out = {k: dict(v) for k, v in by_kind.items()}
    # Pull out concrete examples of new FX being proposed/accepted/rejected
    new_fx = {"apply_reverb", "apply_delay", "apply_saturation", "apply_deesser"}
    fx_examples = {
        "accepted": [r for r in accepted if r["action"] in new_fx],
        "rejected": [r for r in cand_rows
                     if r.get("action") in new_fx and r.get("improved") is False],
        "invalid": [r for r in cand_rows
                    if r.get("action") in new_fx and r.get("invalid")],
    }

    log.update({
        "status": "done",
        "final_mix": {
            "per_stem_lufs": final_lufs_per_stem,
            "pq": round(best_pq, 4), "sb": round(best_sb, 4),
            "reward": round(best_reward, 4),
            "n_accepted": len(accepted),
            "accepted_actions": accepted,
        },
        "reward_delta": round(best_reward - init_reward, 4),
        "by_action_kind": by_kind_out,
        "new_fx_examples": fx_examples,
        "cost": cost,
    })
    _save_log(log)

    print("\n" + "=" * 72)
    print(f"[RESULT] initial reward = {init_reward:.4f} -> final reward = "
          f"{best_reward:.4f}  (Δ={best_reward - init_reward:+.4f})")
    print(f"[RESULT] initial PQ={init_pq:.3f} SB={init_sb:.3f} -> "
          f"final PQ={best_pq:.3f} SB={best_sb:.3f}")
    print(f"[RESULT] accepted {len(accepted)}/{len([r for r in cand_rows if 'render_sec' in r])} scored candidates")
    print("[RESULT] by action kind (proposed/accepted/rejected/invalid):")
    for k in _ACTION_KINDS:
        h = ("apply_static_" + k) if k in ("gain", "pan", "width", "eq") else \
            ("apply_static_compressor" if k == "comp" else "apply_" + k)
        v = by_kind_out.get(h, {"proposed": 0, "accepted": 0, "rejected": 0, "invalid": 0})
        print(f"        {k:11s} ({h:24s}) "
              f"{v['proposed']}/{v['accepted']}/{v['rejected']}/{v['invalid']}")
    print(f"[RESULT] new-FX accepted={len(fx_examples['accepted'])} "
          f"rejected={len(fx_examples['rejected'])} invalid={len(fx_examples['invalid'])}")
    print(f"[COST] per-candidate (render+PQ+SB) mean="
          f"{cost['per_candidate_total_sec']['mean']:.2f}s "
          f"(render={cost['render_sec']['mean']:.2f} "
          f"pq={cost['pq_sec']['mean']:.2f} sb={cost['sb_sec']['mean']:.2f})")
    print("[COST] render by action kind (mean s):")
    for h, st in cost["render_by_action_kind"].items():
        print(f"        {h:24s} mean={st['mean']:.2f}s n={st['n']}")
    print(f"[COST] 1-song loop (calib+search) = {per_song_loop_sec:.1f}s "
          f"-> 150 songs ~= {cost['est_150_songs_hours']:.1f} h")
    print(f"[COST] wall total = {cost['wall_total_sec']:.1f}s")
    print(f"[loop-v2] outputs -> {OUT_DIR}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
