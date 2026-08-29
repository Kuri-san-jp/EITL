"""Shared fixed-structure action space.

All numerical baselines (BO, random) and the LLM agent (when running
in the same-budget comparison mode) sample from this space.

Per stem:
  - static gain in [-12, +6] dB    (continuous)
  - low_eq:  freq in [60, 400],  gain [-6, +6], q in [0.3, 2.0]
  - mid_eq:  freq in [400, 4000], gain [-6, +6], q in [0.3, 2.0]
  - hi_eq:   freq in [4000, 16000], gain [-6, +6], q in [0.3, 2.0]
  - compressor: threshold [-30, -3], ratio [1.0, 4.0],
                attack [3, 50], release [50, 500]
  - pan in [-0.5, +0.5]

Master:
  - master_eq: freq [60, 16000], gain [-3, +3], q [0.3, 2.0]
  - master_limiter ceiling [-3, -0.5] dBTP
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..dsp.mix_state import MixState
from ..dsp.effects import (
    StaticGain, StaticPan, StaticEQ, StaticCompressor,
    MasterEQ, MasterLimiter,
)


@dataclass
class StemParams:
    gain_db: float = 0.0
    pan: float = 0.0
    eq_low_freq:  float = 120.0
    eq_low_gain:  float = 0.0
    eq_low_q:     float = 0.7
    eq_mid_freq:  float = 1000.0
    eq_mid_gain:  float = 0.0
    eq_mid_q:     float = 1.0
    eq_hi_freq:   float = 8000.0
    eq_hi_gain:   float = 0.0
    eq_hi_q:      float = 0.7
    comp_threshold_db: float = -18.0
    comp_ratio: float = 1.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 200.0


@dataclass
class MasterParams:
    eq_freq: float = 1000.0
    eq_gain: float = 0.0
    eq_q: float = 1.0
    limiter_ceiling_db: float = -1.0


@dataclass
class FullActionSet:
    stems: Dict[str, StemParams]
    master: MasterParams


# ---------- bounds for samplers (Optuna / random) ----------

STEM_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "gain_db":           (-12.0,    6.0),
    "pan":               (-0.5,     0.5),
    "eq_low_freq":       ( 60.0,  400.0),
    "eq_low_gain":       ( -6.0,    6.0),
    "eq_low_q":          (  0.3,    2.0),
    "eq_mid_freq":       (400.0, 4000.0),
    "eq_mid_gain":       ( -6.0,    6.0),
    "eq_mid_q":          (  0.3,    2.0),
    "eq_hi_freq":        (4000.0, 16000.0),
    "eq_hi_gain":        ( -6.0,    6.0),
    "eq_hi_q":           (  0.3,    2.0),
    "comp_threshold_db": (-30.0,   -3.0),
    "comp_ratio":        (  1.0,    4.0),
    "comp_attack_ms":    (  3.0,   50.0),
    "comp_release_ms":   ( 50.0,  500.0),
}

MASTER_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "eq_freq":            (60.0, 16000.0),
    "eq_gain":            (-3.0,     3.0),
    "eq_q":               ( 0.3,     2.0),
    "limiter_ceiling_db": (-3.0,    -0.5),
}


# ---------- conversion: ActionSet → MixState ----------

def apply_action_set(state: MixState, actions: FullActionSet) -> MixState:
    s = state
    for trk, p in actions.stems.items():
        if trk not in s.stems:
            continue
        if p.eq_low_gain != 0.0:
            s = s.add_static(trk, StaticEQ(freq=p.eq_low_freq, gain_db=p.eq_low_gain, q=p.eq_low_q),
                             summary=f"BO eq_low[{trk}]")
        if p.eq_mid_gain != 0.0:
            s = s.add_static(trk, StaticEQ(freq=p.eq_mid_freq, gain_db=p.eq_mid_gain, q=p.eq_mid_q),
                             summary=f"BO eq_mid[{trk}]")
        if p.eq_hi_gain != 0.0:
            s = s.add_static(trk, StaticEQ(freq=p.eq_hi_freq, gain_db=p.eq_hi_gain, q=p.eq_hi_q),
                             summary=f"BO eq_hi[{trk}]")
        if p.comp_ratio > 1.05:
            s = s.add_static(trk, StaticCompressor(threshold_db=p.comp_threshold_db,
                                                   ratio=p.comp_ratio,
                                                   attack_ms=p.comp_attack_ms,
                                                   release_ms=p.comp_release_ms),
                             summary=f"BO comp[{trk}]")
        if abs(p.gain_db) > 0.01:
            s = s.add_static(trk, StaticGain(gain_db=p.gain_db), summary=f"BO gain[{trk}]")
        if abs(p.pan) > 0.01:
            s = s.add_static(trk, StaticPan(pan=p.pan), summary=f"BO pan[{trk}]")
    m = actions.master
    if abs(m.eq_gain) > 0.01:
        s = s.add_master(MasterEQ(freq=m.eq_freq, gain_db=m.eq_gain, q=m.eq_q),
                         summary="BO master_eq")
    s = s.add_master(MasterLimiter(ceiling_db=m.limiter_ceiling_db, release_ms=100.0),
                     summary="BO master_lim")
    return s


def sample_random(rng: np.random.Generator, track_names: List[str]) -> FullActionSet:
    """Uniform random sample from the action space."""
    stems = {}
    for trk in track_names:
        p = StemParams()
        for k, (lo, hi) in STEM_PARAM_BOUNDS.items():
            setattr(p, k, float(rng.uniform(lo, hi)))
        stems[trk] = p
    master = MasterParams()
    for k, (lo, hi) in MASTER_PARAM_BOUNDS.items():
        setattr(master, k, float(rng.uniform(lo, hi)))
    return FullActionSet(stems=stems, master=master)


def neutral_action_set(track_names: List[str]) -> FullActionSet:
    """All-zero / neutral defaults (initial rough mix without processing)."""
    stems = {trk: StemParams() for trk in track_names}
    return FullActionSet(stems=stems, master=MasterParams())
