"""Difficulty classifier for songs: Tier A (easy) / B (medium) / C (hard).

The tier is derived from cheap stem-level statistics so we can compute
it once per song and stratify the experiment grid. The "hard" tier is
where the multi-scope multi-resolution agent is expected to dominate
classical baselines — that's the β-plan core claim.

Features computed (cheap, all numpy):
  - n_stems
  - mean dynamic range (short-term LUFS p95 - p10 averaged across stems)
  - vocal_present (any stem name matches /vocal|voc|lead/)
  - masking_overlap (energy overlap between vocal band 200-3000Hz and others)
  - duration_sec

Heuristic decision tree (thresholds in configs/datasets/tier_thresholds.yaml):

    if n_stems ≤ 3 and dyn_range < DR_LOW: A
    elif n_stems ≥ 6 and (dyn_range ≥ DR_HIGH or masking ≥ MASK_HIGH): C
    else: B
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


@dataclass
class TierFeatures:
    n_stems: int
    duration_sec: float
    dynamic_range_db: float
    vocal_present: bool
    masking_overlap_db: float
    n_full_band_stems: int


@dataclass
class TierResult:
    tier: str            # "A" | "B" | "C"
    features: TierFeatures
    rule_fired: str      # which branch fired (for inspection)


# ---------- thresholds ----------


DEFAULT_THRESHOLDS = {
    "n_stems_easy_max": 3,
    "n_stems_hard_min": 6,
    "dyn_range_db_low": 4.0,     # below this = easy
    "dyn_range_db_high": 9.0,    # at/above this = hard candidate
    "masking_overlap_db_high": -3.0,   # vocal-band overlap with non-vocal stems
}


def load_thresholds(path: Optional[str] = None) -> Dict[str, float]:
    if not path:
        candidate = Path(__file__).resolve().parents[3] / "configs/datasets/tier_thresholds.yaml"
        if candidate.exists():
            path = str(candidate)
    if path and Path(path).exists():
        return {**DEFAULT_THRESHOLDS, **yaml.safe_load(Path(path).read_text("utf-8"))}
    return DEFAULT_THRESHOLDS


# ---------- feature extraction ----------


def _short_term_lufs(stem: np.ndarray, sr: int) -> List[float]:
    """Approximate short-term LUFS (3s windows). Returns dB values."""
    if stem.ndim == 2:
        mono = stem.mean(axis=0)
    else:
        mono = stem
    if len(mono) < sr:
        return []
    win = 3 * sr
    hop = sr
    out: List[float] = []
    for i in range(0, len(mono) - win, hop):
        seg = mono[i:i + win]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
        out.append(20.0 * np.log10(rms + 1e-12))
    return out


def _band_energy_db(audio: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    from numpy.fft import rfft, rfftfreq
    a = audio.mean(axis=0) if audio.ndim == 2 else audio
    a = a[: min(len(a), sr * 10)]
    spec = np.abs(rfft(a)) ** 2
    freqs = rfftfreq(len(a), 1 / sr)
    band = (freqs >= f_lo) & (freqs < f_hi)
    e = float(spec[band].sum())
    return 10.0 * np.log10(e + 1e-12)


def _full_band_energy_db(audio: np.ndarray, sr: int) -> float:
    a = audio.mean(axis=0) if audio.ndim == 2 else audio
    a = a[: min(len(a), sr * 10)]
    return 10.0 * np.log10(np.sum(a.astype(np.float64) ** 2) + 1e-12)


def _is_vocal_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("vocal", "voc", "lead", "sing"))


def _is_vocal_instrument(instrument: str) -> bool:
    """Match against MedleyDB / MoisesDB instrument labels."""
    if not instrument:
        return False
    s = instrument.lower()
    return any(k in s for k in (
        "vocal", "voice", "singer", "sing", "lead", "rap",
    ))


def _is_vocal(name: str, instrument_map: Optional[Dict[str, str]] = None) -> bool:
    """Check both stem name and (if available) instrument metadata."""
    if instrument_map and name in instrument_map:
        if _is_vocal_instrument(instrument_map[name]):
            return True
    return _is_vocal_name(name)


def _center_crop(audio: np.ndarray, max_samples: int) -> np.ndarray:
    """Center-crop a 1-D / (C,N) array to at most `max_samples`."""
    if audio.ndim == 1:
        n = audio.shape[0]
        if n <= max_samples:
            return audio
        start = (n - max_samples) // 2
        return audio[start:start + max_samples]
    # (C, N)
    n = audio.shape[-1]
    if n <= max_samples:
        return audio
    start = (n - max_samples) // 2
    return audio[..., start:start + max_samples]


def compute_features(stems: Dict[str, np.ndarray], sr: int,
                     instrument_map: Optional[Dict[str, str]] = None,
                     excerpt_sec: Optional[float] = None,
                     silence_floor_db: float = -50.0,
                     ) -> TierFeatures:
    """Compute tier-relevant features.

    `excerpt_sec` (e.g. 60.0) restricts analysis to a centered window —
    important on long full songs where silence padding inflates the
    dynamic-range estimate.

    `silence_floor_db` filters short-term LUFS values below this floor
    before computing the p95-p10 dynamic range, so long quiet
    intros/outros don't dominate.
    """
    if not stems:
        raise ValueError("empty stems dict")
    if excerpt_sec is not None and excerpt_sec > 0:
        max_n = int(excerpt_sec * sr)
        stems = {k: _center_crop(v, max_n) for k, v in stems.items()}

    duration = max(s.shape[-1] for s in stems.values()) / sr

    # dynamic range averaged across stems (silence-floored)
    dr_per_stem: List[float] = []
    full_band_count = 0
    for name, s in stems.items():
        lu = _short_term_lufs(s, sr)
        if len(lu) >= 2:
            arr = np.array([x for x in lu
                            if np.isfinite(x) and x > silence_floor_db],
                           dtype=np.float64)
            if len(arr) >= 2:
                dr = float(np.percentile(arr, 95) - np.percentile(arr, 10))
                dr_per_stem.append(dr)
        if _full_band_energy_db(s, sr) > -30:    # non-silent stem
            full_band_count += 1
    dyn_range = float(np.mean(dr_per_stem)) if dr_per_stem else 0.0

    vocal_present = any(_is_vocal(n, instrument_map) for n in stems)

    # Masking: vocal-band (200-3000 Hz) energy ratio across vocal vs other stems
    vocal_band = (200.0, 3000.0)
    voc_energy = -np.inf
    other_energy = -np.inf
    for name, s in stems.items():
        e = _band_energy_db(s, sr, *vocal_band)
        if _is_vocal(name, instrument_map):
            voc_energy = max(voc_energy, e)
        else:
            other_energy = max(other_energy, e)
    if voc_energy == -np.inf or other_energy == -np.inf:
        masking_overlap = float("-inf")
    else:
        # +ve = vocal dominates, -ve = vocal masked by others
        masking_overlap = float(voc_energy - other_energy)

    return TierFeatures(
        n_stems=len(stems),
        duration_sec=float(duration),
        dynamic_range_db=dyn_range,
        vocal_present=vocal_present,
        masking_overlap_db=masking_overlap,
        n_full_band_stems=full_band_count,
    )


def classify(stems: Dict[str, np.ndarray], sr: int,
             thresholds: Optional[Dict[str, float]] = None,
             instrument_map: Optional[Dict[str, str]] = None,
             excerpt_sec: Optional[float] = 60.0) -> TierResult:
    """Classify a song into tier A/B/C.

    `excerpt_sec` defaults to 60 s (centered) so full-length songs with
    long silent intros/outros do not inflate the dynamic-range
    estimate. Pass `None` to use the full song.
    """
    th = thresholds or load_thresholds()
    f = compute_features(stems, sr, instrument_map=instrument_map,
                         excerpt_sec=excerpt_sec)

    # ---- easy ----
    if (f.n_stems <= th["n_stems_easy_max"]
            and f.dynamic_range_db < th["dyn_range_db_low"]):
        return TierResult(tier="A", features=f,
                          rule_fired="n_stems_low_AND_dyn_low")

    # ---- hard ----
    hard_signals = 0
    fired: List[str] = []
    if f.n_stems >= th["n_stems_hard_min"]:
        hard_signals += 1
        fired.append("n_stems_high")
    if f.dynamic_range_db >= th["dyn_range_db_high"]:
        hard_signals += 1
        fired.append("dyn_range_high")
    # If vocal exists AND masking is negative (vocal masked by others), hard
    if f.vocal_present and np.isfinite(f.masking_overlap_db) \
            and f.masking_overlap_db <= th["masking_overlap_db_high"]:
        hard_signals += 1
        fired.append("vocal_masked")
    if hard_signals >= 2:
        return TierResult(tier="C", features=f, rule_fired="+".join(fired))

    # ---- medium (default) ----
    return TierResult(tier="B", features=f, rule_fired="default_medium")


def to_dict(r: TierResult) -> Dict[str, Any]:
    return {"tier": r.tier, "rule_fired": r.rule_fired,
            "features": asdict(r.features)}
