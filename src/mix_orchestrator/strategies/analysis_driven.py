"""P2: analysis-driven mixing.

Rather than "listen to the finished mix, then adjust", we **analyse the dry
stems first and decide the mixing policy**, apply it per track, and only then
sum (internal notes).

Procedure:
  1. per-stem analysis: get each stem's RMS level / spectral centroid / role.
  2. genre identification: classify the dry sum with GenreClassifier
     (MAEST/Discogs 129 class).
  3. genre-conditioned per-role level policy → apply a gain (plus a simple
     EQ/pan) to each track.
  4. mix (sum).

This is a knowledge-based policy (mixing convention), not an optimisation that
maximises the reward directly (which avoids metric-gaming). Fine-tuning is
layered on top as a separate pattern (P1).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..dsp.mix_state import MixState
from ..dsp.renderer import render
from ..tools import action_tools as A


# ---- stem name → role inference (follows the naming of the MEGAMI dry examples) ----
_ROLE_KEYWORDS = {
    "vocal": ["vocal", "vox", "lead", "voice", "sing"],
    "drums": ["drum", "kick", "snare", "hat", "perc", "tom", "cymbal"],
    "bass":  ["bass", "sub"],
    "guitar": ["guitar", "gtr"],
    "keys":  ["piano", "key", "synth", "organ", "rhodes", "epiano"],
    "strings": ["string", "violin", "cello", "viola"],
    "brass": ["brass", "horn", "trumpet", "sax", "trombone"],
}


def _role_of(stem_name: str) -> str:
    s = stem_name.lower()
    for role, kws in _ROLE_KEYWORDS.items():
        if any(k in s for k in kws):
            return role
    return "other"


def role_of_audio(audio: np.ndarray, sr: int) -> str:
    """Coarse audio-based role inference (fallback for stems whose name carries
    no information).

    Decides between vocal/drums/bass/other from the spectral centroid
    (low = bass), the low-band ratio, and percussivity (zero-crossing +
    flatness). Not perfect, but it rescues the name=other cases.
    """
    import librosa
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float32)
    if np.sqrt(np.mean(mono ** 2) + 1e-12) < 1e-4:
        return "other"
    try:
        cen = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
        # low-band energy ratio (<200Hz)
        spec = np.abs(np.fft.rfft(mono[:min(len(mono), sr * 10)].astype(np.float64)))
        freqs = np.fft.rfftfreq(min(len(mono), sr * 10), 1.0 / sr)
        low_ratio = float((spec[freqs < 200] ** 2).sum() / ((spec ** 2).sum() + 1e-12))
        # percussivity: variance of the onset strength (percussion has many transients)
        onset = librosa.onset.onset_strength(y=mono, sr=sr)
        perc = float(np.std(onset) / (np.mean(onset) + 1e-9)) if onset.size else 0.0
        flat = float(np.mean(librosa.feature.spectral_flatness(y=mono)))
    except Exception:                                          # noqa: BLE001
        return "other"
    # decision (order: bass → drums → vocal → other)
    if cen < 350 and low_ratio > 0.5:
        return "bass"
    if perc > 1.2 and flat > 0.02:
        return "drums"
    if 800 <= cen <= 4000 and low_ratio < 0.4 and flat < 0.05:
        return "vocal"
    return "other"


def role_of_combined(name: str, audio: np.ndarray, sr: int) -> str:
    """Name first; if name=other, fill in with the audio-based estimate."""
    r = _role_of(name)
    if r == "other":
        return role_of_audio(audio, sr)
    return r


def analyze_stems(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict]:
    """Return each stem's RMS(dBFS) / spectral centroid / role."""
    import librosa
    out = {}
    for name, x in stems.items():
        mono = x.mean(axis=0) if x.ndim == 2 else x
        rms = float(np.sqrt(np.mean(mono.astype(np.float64)**2) + 1e-12))
        rms_db = 20.0 * np.log10(rms + 1e-12)
        try:
            cen = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
        except Exception:                                  # noqa: BLE001
            cen = 0.0
        out[name] = {"role": _role_of(name), "rms_db": rms_db,
                     "centroid_hz": cen, "active": rms_db > -50.0}
    return out


# ---- genre (coarse category) → per-role relative level policy (dB) ----
# A coarse reflection of mixing convention. Each value is "how many dB to lift or
# drop that role relative to base".
_GENRE_POLICY: Dict[str, Dict[str, float]] = {
    # vocals up front, drums/bass solid
    "pop":      {"vocal": +3.0, "drums": +1.0, "bass": +1.0, "keys": -1.0, "guitar": -1.0, "other": -1.0},
    "rock":     {"vocal": +2.0, "drums": +2.0, "bass": +1.0, "guitar": +1.0, "keys": -1.5, "other": -1.0},
    "electronic": {"vocal": +1.0, "drums": +2.0, "bass": +3.0, "keys": 0.0, "guitar": -1.0, "other": -0.5},
    "hiphop":   {"vocal": +3.0, "drums": +2.0, "bass": +3.0, "keys": -1.0, "guitar": -1.0, "other": -1.0},
    "jazz":     {"vocal": +1.0, "drums": 0.0, "bass": +1.0, "keys": +0.5, "brass": +1.0, "other": 0.0},
    "country":  {"vocal": +3.0, "drums": +0.5, "bass": +1.0, "guitar": +1.0, "keys": -0.5, "other": -0.5},
    "soul":     {"vocal": +3.0, "drums": +1.0, "bass": +2.0, "keys": 0.0, "brass": +1.0, "other": -0.5},
    "default":  {"vocal": +2.0, "drums": +1.0, "bass": +1.0, "keys": -0.5, "guitar": 0.0, "other": -0.5},
}


def _coarse_genre(label: str) -> str:
    """Map a MAEST/Discogs label (e.g. 'Rock---Indie Rock') to a coarse category."""
    s = (label or "").lower()
    if any(k in s for k in ["hip hop", "hip-hop", "rap", "trap"]):
        return "hiphop"
    if any(k in s for k in ["electronic", "techno", "house", "edm", "dance", "disco"]):
        return "electronic"
    if any(k in s for k in ["jazz"]):
        return "jazz"
    if any(k in s for k in ["funk", "soul", "r&b", "rnb"]):
        return "soul"
    if any(k in s for k in ["country", "folk"]):
        return "country"
    if any(k in s for k in ["pop"]):
        return "pop"
    if any(k in s for k in ["rock", "metal", "punk", "grunge"]):
        return "rock"
    return "default"


async def identify_genre(mix: np.ndarray, sr: int,
                         genre_ear=None) -> Tuple[str, str, float]:
    """Identify the genre from the dry sum. Returns (coarse, raw_label, confidence)."""
    if genre_ear is None:
        from ..ears.tier5_task import GenreClassifierEar
        genre_ear = GenreClassifierEar(use_proxy_if_missing=False)
    res = await genre_ear.evaluate(mix, sr)
    raw = ""
    conf = 0.0
    if isinstance(res.raw, dict):
        raw = res.raw.get("top_label", "") or res.raw.get("label", "")
    if isinstance(res.score, dict):
        conf = float(res.score.get("genre_confidence", 0.0))
    return _coarse_genre(raw), raw, conf


async def build_analysis_driven_mix(
        stems: Dict[str, np.ndarray], sr: int,
        genre_ear=None,
        forced_genre: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """P2: build the base mix in an analysis-driven way. Returns (mix_audio (C,L), meta)."""
    main = {k: v for k, v in stems.items() if k != "mixture"}
    analysis = analyze_stems(main, sr)

    # genre identification (uses the dry sum)
    dry_sum = render(MixState.initial_from_stems(main, sr))
    if forced_genre:
        coarse, raw, conf = forced_genre, forced_genre, 1.0
    else:
        coarse, raw, conf = await identify_genre(dry_sum, sr, genre_ear)
    policy = _GENRE_POLICY.get(coarse, _GENRE_POLICY["default"])

    # Adjust the gain of each stem according to the role policy.
    # Active stems only: apply the per-role target dB relative to base.
    state = MixState.initial_from_stems(main, sr)
    applied = {}
    for name, info in analysis.items():
        if not info["active"]:
            continue
        g = policy.get(info["role"], policy.get("other", 0.0))
        if abs(g) > 1e-6:
            state = A.apply_static_gain(state, {"track": name, "gain_db": g})
        applied[name] = {"role": info["role"], "gain_db": g}

    mix = render(state)
    meta = {"method": "analysis_driven_P2", "genre_coarse": coarse,
            "genre_raw": raw, "genre_conf": conf, "applied": applied}
    return mix, meta
