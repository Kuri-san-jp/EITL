"""Aggregate ear results into a single reward + consensus metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import EarResult


# Default reward weights — overridable from config. Negative means "lower is better".
#
# Allocation policy (starting point for the Stage 1 calibration):
#   - Tier 1 MOS family: ~50% of the total reward. The Audiobox 4 axes
#     (PQ/PC/CE/CU) carry it, with NISQA/UTMOS/DNSMOS/SingMOS/SCOREQ as
#     complements.
#   - Tier 2 reference family: ~20%. Keeps the mix from drifting off the
#     original mix.
#   - Tier 4 detector family: ~15%. Technical quality guards (clip / phase /
#     dyn range).
#   - Tier 5 task-grounded family: ~15%. Vocal intelligibility, genre
#     preservation, stem separability.
#
# Notes:
#   - The scores live on different scales (MOS 1-5, cosine 0-1, dB, unbounded
#     distance), so a plain linear sum is crude. These are placeholder values,
#     on the assumption that Stage 1 re-derives the normalization and the
#     weights from real mix samples.
#   - A negative weight means "lower is better" and acts as a penalty on
#     distance or clip metrics.
# =====================================================================
# Phase B validated no-reference music reward 
# =====================================================================
# Design rationale (internal notes, analyze_phaseB.py):
#   - In the Phase A broad search (6 songs x 49 variants), "the search best beats
#     the raw dry sum on 6/6 songs", i.e. the reward has generalizable structure.
#   - The ears that identify a generally good mix are audiobox PQ/CU/CE (a
#     coherent cluster, mutual Spearman 0.62-0.73) plus stereo_quality (an
#     orthogonal axis, uncorrelated with audiobox).
#   - Non-discriminative (excluded): audiobox.PC, genre, vocal_intel,
#     stereo_correlation, lra (~0 contribution to the ranking after z-scoring).
#   - The speech-MOS ears (nisqa/utmos/dnsmos/singmos/scoreq) sit at the floor on
#     music -> excluded.
#   - true_peak is not a reward term, only a hard safety gate (decision_policy).
#
# The weights apply to values **after z-score normalization** (see
# REWARD_CALIBRATION). The coherent aesthetic core (PQ/CU/CE) is weighted
# equally, and the orthogonal stereo_quality gets half the weight as a secondary
# objective. This avoids domination by a single ear (reward A: correlations of
# 0.72/0.82/0.78 with PQ/CU/CE, i.e. balanced).
#
# The reference family (CLAP/MERT/ViSQOL/CDPAM/FAD) and reseparation are 0
# because the dry setting has no reference. Re-enabling them is under
# consideration for Phase C (reference-based).
DEFAULT_WEIGHTS: Dict[str, float] = {
    # ---- coherent aesthetic core (z-scored) ----
    "audiobox.PQ":               1.0,
    "audiobox.CU":               1.0,
    "audiobox.CE":               1.0,
    # ---- orthogonal stereo image axis (z-scored, secondary objective) ----
    "stereo.stereo_quality":     0.5,
    # ---- below: unused in the reward (0); the ears are still computed and logged for reference ----
    "audiobox.PC":               0.0,   # production complexity: non-discriminative (2/6)
    "nisqa":                     0.0,   # speech-MOS (floors on music)
    "utmos":                     0.0,
    "dnsmos.ovrl_mos":           0.0,
    "singmos":                   0.0,
    "scoreq":                    0.0,
    "clap.clap_cosine":          0.0,   # reference family (revisit in Phase C)
    "mert.mert_distance":        0.0,
    "visqol_music.moslqo":       0.0,
    "cdpam.cdpam_distance":      0.0,
    "fad.fad":                   0.0,
    "qwen2_audio_judge":         0.0,   # judge is not trustworthy (Phase A2)
    "stereo.stereo_correlation": 0.0,   # non-discriminative ("more mono = better" is wrong)
    "true_peak.true_peak_db":    0.0,   # hard gate only
    "lra.lra":                   0.0,   # ~0 contribution to the ranking after z-scoring
    "vocal_intelligibility.intelligibility": 0.0,
    "vocal_intelligibility.1_minus_wer":     0.0,
    "genre.genre_confidence":                0.0,
    "reseparation_sdr.sdr_mean":             0.0,
}


def _load_calibration() -> Dict[str, Dict[str, float]]:
    """Read reward_calibration.json (the z-score constants). Empty dict if absent."""
    import json as _json
    from pathlib import Path as _Path
    p = (_Path(__file__).resolve().parents[3]
         / "configs/agent/reward_calibration.json")
    if not p.exists():
        return {}
    try:
        d = _json.loads(p.read_text("utf-8"))
        return {k: v for k, v in d.items() if isinstance(v, dict) and "mean" in v}
    except Exception:                                # noqa: BLE001
        return {}


# z-score normalization constants (Phase B, computed over the explore_mixing_v1 population)
REWARD_CALIBRATION: Dict[str, Dict[str, float]] = _load_calibration()


def flatten(results: Dict[str, EarResult]) -> Dict[str, float]:
    """`{ear_name: EarResult}` → `{"ear.key": value}` flat dict."""
    out: Dict[str, float] = {}
    for ear_name, r in results.items():
        if isinstance(r.score, dict):
            for k, v in r.score.items():
                out[f"{ear_name}.{k}"] = float(v)
        else:
            out[ear_name] = float(r.score)
    return out


def aggregate(flat: Dict[str, float], weights: Dict[str, float] = None,
              calibration: Dict[str, Dict[str, float]] = None) -> float:
    """Weighted reward. Ears that have a calibration are z-scored before weighting.

    Phase B : to absorb the scale differences (audiobox ~7,
    stereo_quality ~0.8, etc.), values are z-score normalized with the {mean,std}
    from REWARD_CALIBRATION before the linear combination. Keys without a
    calibration use the raw value (backward compatible; in the current reward
    those all have weight=0).
    """
    weights = weights or DEFAULT_WEIGHTS
    calibration = calibration if calibration is not None else REWARD_CALIBRATION
    total = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        if w == 0.0:
            continue
        if k in flat and np.isfinite(flat[k]):
            v = flat[k]
            c = calibration.get(k)
            if c and c.get("std", 0.0) > 1e-9:
                v = (v - c["mean"]) / c["std"]
            total += w * v
            weight_sum += abs(w)
    return total / weight_sum if weight_sum > 0 else 0.0


def divergence(prev: Dict[str, float], cur: Dict[str, float]) -> Dict[str, float]:
    """Per-key delta with NaN-safety."""
    out = {}
    for k in set(prev) | set(cur):
        a = prev.get(k, float("nan"))
        b = cur.get(k, float("nan"))
        if np.isfinite(a) and np.isfinite(b):
            out[k] = b - a
    return out
