"""informed-random search proposer (good prior + zero adaptivity control arm).

Required by the external audit : comparing against an LLM proposer
needs an "informed random" control that merely follows a good prior with zero
adaptivity. This module provides a search_proposer hook implementation that
statically samples (action, args) from an empirical prior
(experiments/informed_prior_stats.json) aggregated from the acceptance gains of
a measured uniform-random run (exc10k_random_pq, 22 songs x 10000 steps =
220k proposals). The hook calling convention matches
agreement_loop_all._propose_pipeline:

    propose(state, pq, sb, reward, calib, accepted, rejected,
            step, n_search, tracks, sr, rand_proposal=...) -> (name, args)

(agreement_loop_all.py lines 1521-1525).

Design invariants
-----------------
* **The action space is exactly identical to uniform-random
  (agreement_loop_v2._propose_action).** 9 actions, the legal argument ranges
  (the uniform ranges at agreement_loop_v2.py lines 280-348), the discrete EQ
  freq grid, reverb highpass_hz {0,150,300}, compressor knee_db fixed at 6.0.
  The only thing replaced is **the sampling distribution (the weights)**. At
  load time we check that the legal_range / discrete support on the stats side
  matches the uniform side, and every sampled value is clipped into the legal
  range (this prevents schema violations and asymmetric action spaces).
* **Static distribution only.** The sampling distribution does not depend on
  state / acceptance history / step / rand_proposal at all (no win-stay style
  adaptation). That is the point of the control arm: separating "how close a
  good prior alone gets to the LLM" from adaptivity.
* **The rng is a dedicated stream, independent of the pipeline rng.** The hook
  never touches the pipeline rng (touching it would break CRN). The pipeline
  keeps drawing a uniform proposal every step regardless of proposer type
  (agreement_loop_all.py line 1520), so rng consumption and the calibration
  pool match a uniform-random run with the same seed exactly, and the informed
  proposal takes the form of "discard the uniform proposal and substitute".
  **Calibration is therefore fit from the uniform proposals (not from the
  informed distribution).** The z reference being identical to the uniform arm
  is in fact an advantage for the paired comparison, but it must be stated
  explicitly in the report.
* **The EQ 3D table is re-smoothed with alpha=EQ3D_ALPHA.** The probs in stats
  use Laplace alpha=1.0, where the pseudo-count total of 176 dominates the raw
  weight total of 9.07 and makes the table nearly uniform (caveat 1 from the
  aggregation agent). As recommended, recompute it deterministically from
  raw_weight_pq with a small alpha, and record alpha in the provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: Laplace alpha for re-smoothing the EQ 3D table
#: (freq 11 x sign 2 x |gain| 8 = 176 cells).
#: The probs inside stats (alpha=1.0) are pseudo-count dominated and nearly
#: uniform, so they are not used; instead P = (W + alpha) / (sum(W) + alpha * 176)
#: is recomputed deterministically from raw_weight_pq. 0.01 is the value that
#: "keeps a probability floor even on zero-weight cells while preserving the
#: ratios of the measured weights almost as they are" (recommended by the
#: aggregation agent).
EQ3D_ALPHA = 0.01

#: Identifier of the proposer's dedicated rng stream. Mixed into the entropy of
#: the SeedSequence to make it explicit that it does not collide with the
#: pipeline rng (np.random.default_rng(seed)).
_RNG_STREAM_TAG = 0x1AF0

#: Legal argument ranges on the uniform-random side
#: (agreement_loop_v2._propose_action, lines 280-348). At load time we check
#: that the legal_range in stats matches these, and they are also used to clip
#: sampled values. Relying on the stats-side values alone would let a swapped
#: stats file silently widen the action space (duplicated here for fail-fast).
_V2_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "apply_static_gain": {"gain_db": (-4.0, 4.0)},
    "apply_static_pan": {"pan": (-0.7, 0.7)},
    "apply_static_width": {"width": (0.7, 1.6)},
    "apply_static_eq": {"q": (0.6, 2.0)},
    "apply_static_compressor": {"threshold_db": (-30.0, -10.0),
                                "ratio": (1.5, 4.0),
                                "attack_ms": (5.0, 30.0),
                                "release_ms": (80.0, 300.0)},
    "apply_reverb": {"room_size": (0.3, 0.8), "damping": (0.3, 0.7),
                     "wet_level": (0.10, 0.35), "dry_level": (0.7, 1.0),
                     "width": (0.7, 1.0)},
    "apply_delay": {"delay_seconds": (0.12, 0.45), "feedback": (0.1, 0.45),
                    "mix": (0.08, 0.30)},
    "apply_saturation": {"drive_db": (2.0, 9.0)},
    "apply_deesser": {"center_hz": (6000.0, 8500.0),
                      "threshold_db": (-35.0, -20.0), "ratio": (2.0, 6.0),
                      "range_db": (3.0, 10.0), "attack_ms": (0.5, 3.0),
                      "release_ms": (30.0, 120.0)},
}

#: Discrete grid of EQ freq (same values as agreement_loop_v2._EQ_FREQS).
_EQ_FREQS = [80.0, 120.0, 200.0, 300.0, 500.0, 800.0, 1500.0,
             3000.0, 5000.0, 8000.0, 11000.0]
#: Legal range of EQ gain_db (uniform uses U(-5,5)). Values reconstructed from
#: sign x |gain| in the 3D table are clipped into this range.
_EQ_GAIN_RANGE = (-5.0, 5.0)
#: Discrete support of reverb highpass_hz (uniform uses rng.choice([0,150,300])).
_REVERB_HIGHPASS = [0.0, 150.0, 300.0]
#: compressor knee_db is the constant 6.0 on the uniform side (also recorded
#: under constants in stats).
_COMP_KNEE_DB = 6.0

#: Sampling order of the continuous arguments (deterministic). The ordering
#: follows the dict order of the agreement_loop_v2 return value (for
#: readability; since the rng stream is independent of the pipeline, the only
#: requirement is that it is fixed). EQ is special-cased as the 3D table + q.
_ARG_ORDER: Dict[str, List[str]] = {
    "apply_static_gain": ["gain_db"],
    "apply_static_pan": ["pan"],
    "apply_static_width": ["width"],
    "apply_static_compressor": ["threshold_db", "ratio",
                                "attack_ms", "release_ms"],
    "apply_reverb": ["room_size", "damping", "wet_level",
                     "dry_level", "width"],
    "apply_delay": ["delay_seconds", "feedback", "mix"],
    "apply_saturation": ["drive_db"],
    "apply_deesser": ["center_hz", "threshold_db", "ratio",
                      "range_db", "attack_ms", "release_ms"],
}

#: The prefer rule of _pick_track on the uniform side (used as a fallback).
#: Only used when the track support in stats does not intersect the song's stem
#: names. It never fires on MUSDB (bass/drums/other/vocals).
_TRACK_PREFER: Dict[str, List[str]] = {
    "apply_reverb": ["vocals", "other"],
    "apply_delay": ["vocals", "other"],
    "apply_saturation": ["drums", "bass"],
    "apply_deesser": ["vocals"],
}


def _validate_stats(stats: Dict[str, Any]) -> None:
    """Check that the structure and legal ranges of the stats JSON match the
    uniform proposer.

    The stats file is experiment input data and can be swapped out, so we
    fail-fast the moment it is loaded (noticing only after even one search step
    has run throws away GPU/CPU time).
    """
    actions = set(_V2_RANGES)
    have_ap = set(stats.get("action_probs") or {})
    have_pa = set(stats.get("per_action") or {})
    if have_ap != actions or have_pa != actions:
        raise ValueError(
            f"the action set in stats disagrees with the uniform proposer: "
            f"action_probs={sorted(have_ap)} per_action={sorted(have_pa)} "
            f"expected={sorted(actions)}")
    total = sum(float(v["prob"]) for v in stats["action_probs"].values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"action_probs do not sum to 1: {total!r}")
    for act, ranges in _V2_RANGES.items():
        pa = stats["per_action"][act]
        for arg, (lo, hi) in ranges.items():
            if act == "apply_static_eq" and arg != "q":
                continue
            h = (pa.get("args") or {}).get(arg)
            if h is None:
                raise ValueError(f"no histogram for {act}.{arg} in stats")
            lr = [float(x) for x in h.get("legal_range", [])]
            if lr != [lo, hi]:
                raise ValueError(
                    f"{act}.{arg} legal_range {lr} disagrees with the uniform range "
                    f"({lo}, {hi})")
            edges = h.get("bin_edges") or []
            probs = h.get("probs") or []
            if len(edges) != len(probs) + 1:
                raise ValueError(f"{act}.{arg}: length mismatch between bin_edges and probs")
    eq = stats["per_action"]["apply_static_eq"].get("eq_3d") or {}
    axes = eq.get("axes") or {}
    if [float(f) for f in axes.get("freq_hz", [])] != _EQ_FREQS:
        raise ValueError("the freq axis of eq_3d disagrees with _EQ_FREQS")
    if list(axes.get("gain_sign", [])) != ["neg", "pos"]:
        raise ValueError("the gain_sign axis of eq_3d is not ['neg','pos']")
    g_edges = [float(x) for x in axes.get("abs_gain_db_bin_edges", [])]
    if not g_edges or g_edges[0] != 0.0 or g_edges[-1] != _EQ_GAIN_RANGE[1]:
        raise ValueError(f"the |gain| axis of eq_3d is not [0, {_EQ_GAIN_RANGE[1]}]")
    w = np.asarray(eq.get("raw_weight_pq"), dtype=np.float64)
    if w.shape != (len(_EQ_FREQS), 2, len(g_edges) - 1):
        raise ValueError(f"invalid shape {w.shape} for eq_3d.raw_weight_pq")
    hp = (stats["per_action"]["apply_reverb"].get("args") or {}).get(
        "highpass_hz") or {}
    if sorted(float(x) for x in hp.get("support", [])) != _REVERB_HIGHPASS:
        raise ValueError("the support of reverb highpass_hz is not {0,150,300}")
    knee = (stats["per_action"]["apply_static_compressor"].get("constants")
            or {}).get("knee_db")
    if float(knee) != _COMP_KNEE_DB:
        raise ValueError(f"the compressor knee_db constant is not {_COMP_KNEE_DB}: {knee!r}")


def load_action_stats(path: str) -> Tuple[Dict[str, Any], str]:
    """Read the empirical prior JSON and return (stats, sha256 hex).

    The sha256 is taken over **the bytes of the file** (for provenance and the
    resume consistency check: even at the same path, replaced contents are
    detected as a different file).
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[informed] no stats JSON: {p}")
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    stats = json.loads(raw.decode("utf-8"))
    _validate_stats(stats)
    return stats, sha


def build_provenance(stats: Dict[str, Any], stats_path: str, stats_sha256: str,
                     base_seed: int) -> Dict[str, Any]:
    """Build the provenance dict written into the per-song JSON.

    Makes "which version of which stats file this distribution came from"
    traceable from the run JSON alone (audit requirement). The policy of not
    writing the key at all outside this mode is enforced by the caller
    (size_sweep_run.main).
    """
    meta = stats.get("meta") or {}
    return {
        "stats_path": str(stats_path),
        "stats_sha256": str(stats_sha256),
        "source_run": meta.get("source_run"),
        "n_source_proposals": meta.get("n_total_proposals"),
        "weights_version": meta.get("generated_date"),
        "eq3d_alpha": float(EQ3D_ALPHA),
        "base_seed": int(base_seed),
        "rng": "SeedSequence((base_seed, song_index, seed_ordinal, 0x1AF0)); "
               "a dedicated stream independent of the pipeline rng",
        "calibration_source": "uniform (the pipeline fits the calib pool from the "
                              "uniform proposals, not from the informed distribution)",
    }


def assert_informed_stats_consistent(run_dir, stats_sha256: str) -> None:
    """On resume, check that the proposer / stats sha256 of already-done songs
    match this run.

    A fail-fast of the same shape as
    agreement_loop_all.assert_corruption_consistent. Resuming after swapping the
    stats file mixes **songs from a different prior** into a single run, and
    they cannot be told apart without opening proposer_provenance in the JSON.
    It also stops the accident of resuming a uniform-random run with informed.
    Since it is only called in informed_random mode, it has no effect whatsoever
    on the default behaviour.

    It can be disabled deliberately with the environment variable
    ``ALLOW_PROPOSER_STATS_MISMATCH=1``.
    """
    if os.environ.get("ALLOW_PROPOSER_STATS_MISMATCH") == "1":
        print("[guard] ALLOW_PROPOSER_STATS_MISMATCH=1, so skipping the informed "
              "stats consistency check", flush=True)
        return
    songs_dir = Path(run_dir) / "songs"
    if not songs_dir.is_dir():
        return
    mismatches: List[str] = []
    for p in sorted(songs_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("status") != "done":
            continue
        prop = rec.get("proposer")
        if prop != "informed_random":
            mismatches.append(
                f"  {rec.get('song_id')}: proposer existing={prop!r} "
                f"this run='informed_random'")
            continue
        have = (rec.get("proposer_provenance") or {}).get("stats_sha256")
        if have != stats_sha256:
            mismatches.append(
                f"  {rec.get('song_id')}: stats_sha256 existing={str(have)[:12]}... "
                f"this run={stats_sha256[:12]}...")
    if mismatches:
        head = mismatches[:10]
        more = (f"\n  ... {len(mismatches)-10} more"
                if len(mismatches) > 10 else "")
        raise SystemExit(
            f"[guard] the informed_random stats file / proposer disagrees with "
            f"the existing run ({len(mismatches)} songs). Resuming would mix in "
            f"songs from a different prior:\n" + "\n".join(head) + more
            + "\n  -> point --proposer-stats at the same file as the existing "
              "run, or use a different --run-name."
              "\n  -> set ALLOW_PROPOSER_STATS_MISMATCH=1 only if you intend to mix.")


def make_informed_random_proposer(stats: Dict[str, Any], base_seed: int,
                                  song_index: int):
    """Return a callable compatible with the search_proposer hook
    (static informed-random).

    Args:
        stats: the validated stats dict returned by load_action_stats.
        base_seed: the effective base seed (= args.seed + args.random_base_seed).
            Shifted by the same amount as the derivation of the corruption seed,
            so that a seed replication moves the proposer stream along with it.
        song_index: song index in the full plan (the same one as the corruption
            seed).

    Every time ``step == 0`` is detected, the rng advances the seed_ordinal held
    in the closure and reseeds with
    ``SeedSequence((base_seed, song_index, seed_ordinal, 0x1AF0))``
    (one pass of the seed_idx loop = one independent deterministic stream per
    search). Under ``--random-seeds 1`` this effectively happens once. It does
    not interfere with the pipeline rng.
    """
    actions = sorted(stats["action_probs"])
    aprobs = np.asarray([float(stats["action_probs"][a]["prob"])
                         for a in actions], dtype=np.float64)
    aprobs = aprobs / aprobs.sum()
    pa = stats["per_action"]

    # EQ 3D table: deterministically re-smoothed from raw_weight_pq with
    # alpha=EQ3D_ALPHA.
    eq = pa["apply_static_eq"]["eq_3d"]
    eq_w = np.asarray(eq["raw_weight_pq"], dtype=np.float64)
    eq_p = (eq_w + EQ3D_ALPHA) / (eq_w.sum() + EQ3D_ALPHA * eq_w.size)
    eq_p = (eq_p / eq_p.sum()).ravel()
    eq_shape = eq_w.shape
    eq_axes = eq["axes"]

    st: Dict[str, Any] = {"ordinal": -1, "rng": None}

    def _reseed() -> None:
        st["ordinal"] += 1
        st["rng"] = np.random.default_rng(np.random.SeedSequence(
            (int(base_seed), int(song_index), int(st["ordinal"]),
             _RNG_STREAM_TAG)))

    def _hist(rng: np.random.Generator, action: str, arg: str) -> float:
        """Continuous argument: bin ~ Categorical(probs), uniform within the bin.
        Clipped into the legal range."""
        h = pa[action]["args"][arg]
        probs = np.asarray(h["probs"], dtype=np.float64)
        probs = probs / probs.sum()
        i = int(rng.choice(len(probs), p=probs))
        edges = h["bin_edges"]
        v = float(rng.uniform(float(edges[i]), float(edges[i + 1])))
        lo, hi = _V2_RANGES[action][arg]
        return float(min(max(v, lo), hi))

    def _track(rng: np.random.Generator, action: str,
               tracks: List[str]) -> str:
        """track: the acceptance-weight distribution in stats, restricted to the
        song's stem set and renormalized."""
        tr = pa[action]["track"]
        avail = [t for t in tr["support"] if t in tracks]
        if avail:
            w = np.asarray([float(tr["probs"][t]) for t in avail],
                           dtype=np.float64)
            w = w / w.sum()
            return str(avail[int(rng.choice(len(avail), p=w))])
        # Only when the support is empty (e.g. non-standard stem names) do we
        # fall back to the prefer rule of the uniform side (unreachable on MUSDB).
        prefer = _TRACK_PREFER.get(action)
        cands = ([t for t in tracks if t in prefer] if prefer else [])
        cands = cands or list(tracks)
        return str(cands[int(rng.integers(len(cands)))])

    def propose(state, pq: float, sb: float, reward: float, calib,
                accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]],
                step: int, n_search: int, tracks: List[str], sr: int,
                rand_proposal=None) -> Tuple[Optional[str], Dict[str, Any]]:
        # Static distribution: the sampling distribution does not depend on
        # state / history / step.
        # rand_proposal (the uniform proposal the pipeline drew for CRN) is
        # accepted but discarded -- the pipeline's rng consumption is preserved,
        # so with the same seed the calibration pool matches a uniform run
        # exactly.
        if step == 0 or st["rng"] is None:
            _reseed()
        rng: np.random.Generator = st["rng"]

        name = actions[int(rng.choice(len(actions), p=aprobs))]
        args: Dict[str, Any] = {"track": _track(rng, name, list(tracks))}

        if name == "apply_static_eq":
            # (freq, sign, |gain| bin) ~ the re-smoothed 3D table; |gain| is
            # uniform within the bin.
            cell = int(rng.choice(eq_p.size, p=eq_p))
            fi, si, gi = np.unravel_index(cell, eq_shape)
            sign = -1.0 if str(eq_axes["gain_sign"][si]) == "neg" else 1.0
            g_edges = eq_axes["abs_gain_db_bin_edges"]
            mag = float(rng.uniform(float(g_edges[gi]), float(g_edges[gi + 1])))
            gain_db = float(min(max(sign * mag, _EQ_GAIN_RANGE[0]),
                                _EQ_GAIN_RANGE[1]))
            args.update({"freq": float(eq_axes["freq_hz"][fi]),
                         "gain_db": gain_db,
                         "q": _hist(rng, name, "q")})
            return name, args

        for arg in _ARG_ORDER[name]:
            args[arg] = _hist(rng, name, arg)
        if name == "apply_reverb":
            hp = pa[name]["args"]["highpass_hz"]
            keys = [float(k) for k in hp["support"]]
            w = np.asarray([float(hp["probs"][str(k)]) for k in hp["support"]],
                           dtype=np.float64)
            w = w / w.sum()
            args["highpass_hz"] = float(keys[int(rng.choice(len(keys), p=w))])
        if name == "apply_static_compressor":
            args["knee_db"] = _COMP_KNEE_DB
        return name, args

    return propose
