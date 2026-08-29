"""Render a MixState into a stereo waveform via pedalboard.

Pipeline per stem:
    raw → static chain → transient chain → (section overlays w/ crossfade)
Then:
    sum (with pan/width applied as part of static stage) → master chain

Audio convention everywhere: (channels, samples), float32, [-1, 1].
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from pedalboard import (
        Pedalboard, Compressor, Gain, Limiter,
        HighShelfFilter, LowShelfFilter, PeakFilter,
        Reverb as PBReverb, Delay as PBDelay,
        Distortion as PBDistortion, HighpassFilter as PBHighpassFilter,
    )
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

from .mix_state import MixState

# ---------------------------------------------------------------------------
# Envelope-follower kernel (attack/release one-pole).
#
# The de-esser / dynamic-EQ / multiband / sidechain / transient-shaper all need
# a per-sample attack/release envelope follower:
#
#     prev = a*prev + (1-a)*v      with  a = a_att if v > prev else a_rel
#
# This is a *branchy* nonlinear recursion (attack != release), so it cannot be
# vectorised losslessly as an LTI filter. A naive Python `for` loop over every
# sample makes the de-esser ~7x slower than the C-backed pedalboard FX
# (~11.4 s vs ~1-2 s on a full song). We JIT the *exact same* recursion with
# numba so the numerics are preserved (float64 accumulation) while running at
# native speed. If numba is unavailable we fall back to the original pure
# Python loop, so behaviour is identical and nothing breaks.
# ---------------------------------------------------------------------------
try:
    from numba import njit  # type: ignore

    @njit(cache=True, fastmath=False)
    def _env_follow_kernel(absx, a_att, a_rel):  # pragma: no cover - jitted
        """Attack/release one-pole follower over a (C, N) float64 array.

        absx: |signal| as float64, shape (C, N). Returns the envelope, same shape.
        """
        c, n = absx.shape
        out = np.empty((c, n), dtype=np.float64)
        for ch in range(c):
            prev = 0.0
            for i in range(n):
                v = absx[ch, i]
                a = a_att if v > prev else a_rel
                prev = a * prev + (1.0 - a) * v
                out[ch, i] = prev
        return out

    _HAS_NUMBA = True
except Exception:  # numba not installed / failed to import
    _HAS_NUMBA = False

    def _env_follow_kernel(absx, a_att, a_rel):
        """Pure-numpy/Python fallback (identical recursion, slower)."""
        c, n = absx.shape
        out = np.empty((c, n), dtype=np.float64)
        for ch in range(c):
            prev = 0.0
            row = absx[ch]
            o = out[ch]
            for i in range(n):
                v = row[i]
                a = a_att if v > prev else a_rel
                prev = a * prev + (1.0 - a) * v
                o[i] = prev
        return out


def _env_follow(absx: np.ndarray, a_att: float, a_rel: float) -> np.ndarray:
    """Attack/release envelope follower. ``absx`` is |signal|, shape (C, N).

    Returns the per-sample envelope as float64, shape (C, N). Single, shared
    implementation so the de-esser, dynamic EQ, multiband comp, sidechain and
    transient shaper all use the same (now fast) recursion.
    """
    absx = np.ascontiguousarray(absx, dtype=np.float64)
    return _env_follow_kernel(absx, float(a_att), float(a_rel))


from .effects import (
    Effect, StaticEffect, SectionEffect, MasterEffect, TransientEffect,
    StaticGain, StaticPan, StaticWidth, StaticEQ, StaticCompressor,
    SectionGain, SectionEQ, SectionPan, SectionWidth, SectionCompressor,
    MasterEQ, MasterCompressor, MasterLimiter,
    GainAutomation, PanAutomation, WidthAutomation, TransientShaper,
    DynamicEQ, MultibandComp, SidechainComp,
    Reverb, Delay, Saturation, DeEsser,
)


def render(state: MixState, cache: Optional["RenderCache"] = None) -> np.ndarray:
    """Return stereo master, shape (2, N) float32.

    By default this re-applies every chain from the raw stems (the historical
    behaviour, unchanged). Pass a :class:`RenderCache` — or set the environment
    variable ``MIXORCH_RENDER_CACHE=1`` — to enable *differential* rendering,
    which reuses the processed buffers of the previous state and only applies
    the effects that were appended. The two paths are bit-identical; see
    :class:`RenderCache` for the argument and ``tests/test_render_cache.py``
    for the verification.
    """
    if cache is None:
        cache = _global_cache_if_enabled()
    if cache is None:
        return _render_full(state)
    return _render_cached(state, cache)


def _render_full(state: MixState) -> np.ndarray:
    """Non-incremental render: rebuild every chain from the raw stems."""
    if not HAS_PEDALBOARD:
        raise RuntimeError("pedalboard is not installed. `pip install pedalboard`.")

    sr = state.sample_rate
    # Step 1: process each stem through its static+transient chain
    # `raw_stems` is the unprocessed dict — passed for sidechain key inputs.
    raw_stems = {k: _to_stereo(v.astype(np.float32)) for k, v in state.stems.items()}
    processed: Dict[str, np.ndarray] = {}
    for trk, stem in state.stems.items():
        x = _to_stereo(stem.astype(np.float32))
        x = _apply_chain(x, state.static_processors.get(trk, ()), sr,
                         trk_name=trk, all_stems=raw_stems)
        x = _apply_chain(x, state.transient_processors.get(trk, ()), sr,
                         trk_name=trk, all_stems=raw_stems)
        # Section processors: render fully-processed version, then crossfade in
        section_chain = state.section_processors.get(trk, ())
        if section_chain:
            x = _apply_section_chain(x, section_chain, sr)
        processed[trk] = x

    # Step 2: sum to bus
    max_len = max(p.shape[1] for p in processed.values())
    bus = np.zeros((2, max_len), dtype=np.float32)
    for x in processed.values():
        n = x.shape[1]
        bus[:, :n] += x

    # Step 3: master chain
    bus = _apply_chain(bus, state.master_chain, sr, trk_name="master")
    return bus


# ---------- Chain dispatch ----------

def _apply_chain(audio: np.ndarray, chain, sr: int, trk_name: str = "",
                 all_stems: Dict[str, np.ndarray] = None) -> np.ndarray:
    """Apply a sequence of effects in order. Mixes pedalboard + numpy ops."""
    out = audio
    for eff in chain:
        out = _apply_one(out, eff, sr, trk_name=trk_name, all_stems=all_stems)
    return out


def _apply_one(audio: np.ndarray, eff: Effect, sr: int, trk_name: str = "",
               all_stems: Dict[str, np.ndarray] = None) -> np.ndarray:
    if isinstance(eff, StaticGain):
        return _pb_run([Gain(gain_db=eff.gain_db)], audio, sr)
    if isinstance(eff, StaticPan):
        return _apply_pan(audio, eff.pan)
    if isinstance(eff, StaticWidth):
        return _apply_width(audio, eff.width)
    if isinstance(eff, StaticEQ):
        return _pb_run([_peak_filter(eff.freq, eff.gain_db, eff.q)], audio, sr)
    if isinstance(eff, StaticCompressor):
        return _pb_run([Compressor(
            threshold_db=eff.threshold_db, ratio=eff.ratio,
            attack_ms=eff.attack_ms, release_ms=eff.release_ms,
        )], audio, sr)
    if isinstance(eff, MasterEQ):
        return _pb_run([_peak_filter(eff.freq, eff.gain_db, eff.q)], audio, sr)
    if isinstance(eff, MasterCompressor):
        return _pb_run([Compressor(
            threshold_db=eff.threshold_db, ratio=eff.ratio,
            attack_ms=eff.attack_ms, release_ms=eff.release_ms,
        )], audio, sr)
    if isinstance(eff, MasterLimiter):
        return _pb_run([Limiter(threshold_db=eff.ceiling_db, release_ms=eff.release_ms)], audio, sr)
    if isinstance(eff, GainAutomation):
        return _apply_gain_automation(audio, eff.breakpoints, sr)
    if isinstance(eff, PanAutomation):
        return _apply_pan_automation(audio, eff.breakpoints, sr)
    if isinstance(eff, WidthAutomation):
        return _apply_width_automation(audio, eff.breakpoints, sr)
    if isinstance(eff, TransientShaper):
        return _apply_transient_shaper(audio, eff.attack_gain_db, eff.sustain_gain_db, sr)
    if isinstance(eff, DynamicEQ):
        return _apply_dynamic_eq(audio, eff, sr)
    if isinstance(eff, MultibandComp):
        return _apply_multiband_comp(audio, eff, sr)
    if isinstance(eff, SidechainComp):
        if all_stems is None or eff.sidechain_track not in all_stems:
            return audio
        return _apply_sidechain_comp(audio, all_stems[eff.sidechain_track], eff, sr)
    if isinstance(eff, Reverb):
        return _apply_reverb(audio, eff, sr)
    if isinstance(eff, Delay):
        return _pb_run([PBDelay(delay_seconds=eff.delay_seconds,
                                feedback=eff.feedback, mix=eff.mix)], audio, sr)
    if isinstance(eff, Saturation):
        return _pb_run([PBDistortion(drive_db=eff.drive_db)], audio, sr)
    if isinstance(eff, DeEsser):
        return _apply_deesser(audio, eff, sr)
    if isinstance(eff, (SectionGain, SectionEQ, SectionPan, SectionWidth, SectionCompressor)):
        # Section effects are NOT applied here — _apply_section_chain handles them
        return audio
    raise NotImplementedError(f"effect {type(eff).__name__} not handled (track={trk_name})")


# ---------- Section processing ----------

def _apply_section_chain(audio: np.ndarray, chain, sr: int) -> np.ndarray:
    """For each section effect, render a full-length wet version and
    crossfade-mix into the relevant time window."""
    out = audio.copy()
    for eff in chain:
        out = _apply_section_one(out, eff, sr)
    return out


def _apply_section_one(audio: np.ndarray, eff, sr: int) -> np.ndarray:
    """Apply ONE section effect. Split out of `_apply_section_chain` so the
    incremental renderer can resume a chain from a cached prefix; the loop
    above is the only other caller and its semantics are unchanged.

    Never mutates `audio` (`_crossfade_segment` copies its `dry` argument),
    which is what makes it safe to feed a cached buffer straight in.
    """
    if not isinstance(eff, SectionEffect):
        return audio
    start_n = int(eff.start_sec * sr)
    end_n   = int(eff.end_sec   * sr)
    end_n   = min(end_n, audio.shape[1])
    if end_n <= start_n:
        return audio
    wet = _apply_one(audio, _as_static(eff), sr)
    xfade = int(eff.crossfade_ms * sr / 1000)
    return _crossfade_segment(audio, wet, start_n, end_n, xfade)


def _as_static(eff: SectionEffect) -> Effect:
    """Convert a SectionEffect into the equivalent static-style effect
    so we can reuse the same apply logic."""
    if isinstance(eff, SectionGain):
        return StaticGain(gain_db=eff.gain_db)
    if isinstance(eff, SectionEQ):
        return StaticEQ(freq=eff.freq, gain_db=eff.gain_db, q=eff.q)
    if isinstance(eff, SectionPan):
        return StaticPan(pan=eff.pan)
    if isinstance(eff, SectionWidth):
        return StaticWidth(width=eff.width)
    if isinstance(eff, SectionCompressor):
        return StaticCompressor(threshold_db=eff.threshold_db, ratio=eff.ratio,
                                attack_ms=eff.attack_ms, release_ms=eff.release_ms, knee_db=eff.knee_db)
    raise NotImplementedError(f"section effect {type(eff).__name__} not handled")


def _crossfade_segment(dry: np.ndarray, wet: np.ndarray,
                       start_n: int, end_n: int, xfade: int) -> np.ndarray:
    out = dry.copy()
    seg_len = end_n - start_n
    if seg_len <= 0:
        return out
    xfade = min(xfade, seg_len // 2)
    if xfade <= 0:
        out[:, start_n:end_n] = wet[:, start_n:end_n]
        return out
    # Fade-in at start
    fi = np.linspace(0, 1, xfade, dtype=np.float32)
    out[:, start_n:start_n+xfade] = (
        dry[:, start_n:start_n+xfade] * (1 - fi) + wet[:, start_n:start_n+xfade] * fi
    )
    # Full wet middle
    out[:, start_n+xfade:end_n-xfade] = wet[:, start_n+xfade:end_n-xfade]
    # Fade-out at end
    fo = np.linspace(1, 0, xfade, dtype=np.float32)
    out[:, end_n-xfade:end_n] = (
        dry[:, end_n-xfade:end_n] * (1 - fo) + wet[:, end_n-xfade:end_n] * fo
    )
    return out


# ---------- pedalboard helpers ----------

def _pb_run(plugins, audio: np.ndarray, sr: int) -> np.ndarray:
    # pedalboard expects (N, C) float32
    pb = Pedalboard(plugins)
    arr = audio.T.astype(np.float32)
    out = pb(arr, sr)
    return out.T.astype(np.float32)


def _peak_filter(freq: float, gain_db: float, q: float):
    if freq < 200:
        return LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain_db, q=q)
    if freq > 6000:
        return HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain_db, q=q)
    return PeakFilter(cutoff_frequency_hz=freq, gain_db=gain_db, q=q)


# ---------- mono/stereo helpers ----------

def _to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    if x.shape[0] == 2:
        return x
    return x[:2]


def _apply_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    # Equal-power pan
    theta = (pan + 1.0) * np.pi / 4.0   # 0..π/2
    gl = np.cos(theta)
    gr = np.sin(theta)
    if audio.shape[0] == 1:
        mono = audio[0]
        return np.stack([mono * gl, mono * gr]).astype(np.float32)
    # Stereo: collapse to mono first, then re-pan
    mono = audio.mean(axis=0)
    return np.stack([mono * gl, mono * gr]).astype(np.float32)


def _apply_width(audio: np.ndarray, width: float) -> np.ndarray:
    """M/S width: width=0 mono, 1 unchanged, 2 extra-wide."""
    if audio.shape[0] != 2:
        return audio
    width = float(np.clip(width, 0.0, 2.0))
    m = 0.5 * (audio[0] + audio[1])
    s = 0.5 * (audio[0] - audio[1])
    s = s * width
    l = m + s
    r = m - s
    return np.stack([l, r]).astype(np.float32)


def _apply_gain_automation(audio: np.ndarray, breakpoints, sr: int) -> np.ndarray:
    if not breakpoints:
        return audio
    n = audio.shape[1]
    times = np.array([t for t, _ in breakpoints], dtype=np.float32)
    gains = np.array([g for _, g in breakpoints], dtype=np.float32)
    sample_idx = np.arange(n, dtype=np.float32) / sr
    g_db = np.interp(sample_idx, times, gains)
    g_lin = 10.0 ** (g_db / 20.0)
    return (audio * g_lin[None, :]).astype(np.float32)


def _apply_pan_automation(audio: np.ndarray, breakpoints, sr: int) -> np.ndarray:
    """Continuous pan automation. pan(t) is linearly interpolated per sample
    (np.interp), then fed through an equal-power pan. C0-continuous over every
    sample → no zipper noise / clicks. Vectorised (no per-sample loop)."""
    if not breakpoints:
        return audio
    n = audio.shape[1]
    times = np.array([t for t, _ in breakpoints], dtype=np.float32)
    pans = np.clip(np.array([p for _, p in breakpoints], dtype=np.float32), -1.0, 1.0)
    pan_t = np.interp(np.arange(n, dtype=np.float32) / sr, times, pans)
    theta = (pan_t + 1.0) * np.pi / 4.0          # 0..π/2, continuous
    gl = np.cos(theta); gr = np.sin(theta)
    mono = audio[0] if audio.shape[0] == 1 else audio.mean(axis=0)
    return np.stack([mono * gl, mono * gr]).astype(np.float32)


def _apply_width_automation(audio: np.ndarray, breakpoints, sr: int) -> np.ndarray:
    """Continuous width automation. M/S decomposition: M is left unchanged, S is
    scaled by width(t). Per-sample linear interpolation keeps it continuous."""
    if not breakpoints or audio.shape[0] != 2:
        return audio
    n = audio.shape[1]
    times = np.array([t for t, _ in breakpoints], dtype=np.float32)
    widths = np.clip(np.array([w for _, w in breakpoints], dtype=np.float32), 0.0, 2.0)
    w_t = np.interp(np.arange(n, dtype=np.float32) / sr, times, widths)
    m = 0.5 * (audio[0] + audio[1])
    s = 0.5 * (audio[0] - audio[1]) * w_t
    return np.stack([m + s, m - s]).astype(np.float32)


def _apply_transient_shaper(audio: np.ndarray, attack_db: float, sustain_db: float, sr: int) -> np.ndarray:
    """Quick attack/sustain shaper using envelope-follower difference.

    NOT a research-grade implementation — placeholder so the action is
    callable. Replace with a published algorithm in Week 3.
    """
    if attack_db == 0.0 and sustain_db == 0.0:
        return audio
    # Envelope follower (per channel)
    fast_tau = 0.005   # 5ms
    slow_tau = 0.05    # 50ms
    a_fast = np.exp(-1.0 / (fast_tau * sr))
    a_slow = np.exp(-1.0 / (slow_tau * sr))
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        x = audio[ch]
        env_fast = _one_pole(np.abs(x), a_fast)
        env_slow = _one_pole(np.abs(x), a_slow)
        # Positive diff = transient onset
        transient_mask = np.clip(env_fast - env_slow, 0, None)
        sustain_mask = np.clip(env_slow - env_fast, 0, None)
        if env_slow.max() > 0:
            transient_mask /= (env_slow.max() + 1e-9)
            sustain_mask  /= (env_slow.max() + 1e-9)
        g_db = transient_mask * attack_db + sustain_mask * sustain_db
        g = 10.0 ** (g_db / 20.0)
        out[ch] = x * g
    return out.astype(np.float32)


def _one_pole(x: np.ndarray, a: float) -> np.ndarray:
    """Fixed-coefficient one-pole lowpass ``y = a*y + (1-a)*x``.

    Uses the shared envelope kernel with attack == release == a (no branching),
    so it is the same recursion as before but native-speed. Returns the same
    dtype as the input ``x``.
    """
    y = _env_follow(np.atleast_2d(x), a, a)[0]
    return y.astype(x.dtype, copy=False)


# ---------- Dynamic / multiband / sidechain ----------


def _bandpass(audio: np.ndarray, f_lo: float, f_hi: float, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    f_lo = max(20.0, min(f_lo, sr / 2 - 100))
    f_hi = max(f_lo + 10, min(f_hi, sr / 2 - 50))
    sos = butter(2, [f_lo, f_hi], btype="band", fs=sr, output="sos")
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = sosfiltfilt(sos, audio[ch])
    return out.astype(np.float32)


def _envelope_db(audio: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Per-channel attack/release envelope in dB. Returns shape (C, N).

    Same recursion as before (``prev = a*prev + (1-a)*|x|`` with attack/release
    ballistics) but evaluated by the shared, numba-accelerated kernel instead of
    a per-sample Python loop. This is the hot path for the de-esser.
    """
    a_att = float(np.exp(-1.0 / (attack_ms * 1e-3 * sr + 1e-9)))
    a_rel = float(np.exp(-1.0 / (release_ms * 1e-3 * sr + 1e-9)))
    absx = np.abs(np.atleast_2d(audio))
    env = _env_follow(absx, a_att, a_rel)
    env_db = 20.0 * np.log10(env + 1e-9)
    return env_db


def _apply_dynamic_eq(audio: np.ndarray, eff: "DynamicEQ", sr: int) -> np.ndarray:
    # Band-isolate, follow its envelope, modulate gain
    band = _bandpass(audio, eff.freq / np.sqrt(2), eff.freq * np.sqrt(2), sr)
    env_db = _envelope_db(band, sr, eff.attack_ms, eff.release_ms)
    # how much we exceed threshold (in dB), clipped to [0, +∞)
    over = np.clip(env_db - eff.threshold_db, 0, None)
    # Smooth transition: tanh ramps the gain in over 6dB above threshold
    ramp = np.tanh(over / 6.0)
    # Target gain in dB (positive boost or negative cut)
    g_db = eff.gain_db * ramp
    g_lin = 10.0 ** (g_db / 20.0)
    # Apply gain to the band only, then sum (band-passed gain shaping)
    side = band * (g_lin - 1.0)
    return (audio + side).astype(np.float32)


def _apply_multiband_comp(audio: np.ndarray, eff: "MultibandComp", sr: int) -> np.ndarray:
    low_lo, low_hi = 20.0, eff.crossover_lo
    mid_lo, mid_hi = eff.crossover_lo, eff.crossover_hi
    hi_lo,  hi_hi  = eff.crossover_hi, sr / 2 - 100
    bands = [
        (_bandpass(audio, low_lo, low_hi, sr),
         eff.low_threshold_db, eff.low_ratio, eff.low_attack_ms, eff.low_release_ms),
        (_bandpass(audio, mid_lo, mid_hi, sr),
         eff.mid_threshold_db, eff.mid_ratio, eff.mid_attack_ms, eff.mid_release_ms),
        (_bandpass(audio, hi_lo,  hi_hi,  sr),
         eff.high_threshold_db, eff.high_ratio, eff.high_attack_ms, eff.high_release_ms),
    ]
    summed = np.zeros_like(audio)
    for band, thr, ratio, atk, rel in bands:
        env_db = _envelope_db(band, sr, atk, rel)
        over = np.clip(env_db - thr, 0, None)
        red_db = over * (1.0 - 1.0 / max(ratio, 1.001))
        g_lin = 10.0 ** (-red_db / 20.0)
        summed += band * g_lin
    return summed.astype(np.float32)


def _apply_sidechain_comp(audio: np.ndarray, sidechain: np.ndarray,
                          eff: "SidechainComp", sr: int) -> np.ndarray:
    # Use full-band envelope of sidechain (mean abs across channels)
    side_mono = np.mean(np.abs(sidechain), axis=0)[None, :]
    env_db = _envelope_db(side_mono, sr, eff.attack_ms, eff.release_ms)[0]
    # broadcast to match audio length
    n = min(len(env_db), audio.shape[1])
    over = np.clip(env_db[:n] - eff.threshold_db, 0, None)
    red_db = over * (1.0 - 1.0 / max(eff.ratio, 1.001))
    g_lin = 10.0 ** (-red_db / 20.0)
    out = audio.copy()
    out[:, :n] *= g_lin[None, :]
    return out.astype(np.float32)


# ---------- Time-based / harmonic insert FX ----------


def _apply_reverb(audio: np.ndarray, eff: "Reverb", sr: int) -> np.ndarray:
    """Algorithmic reverb with optional pre-reverb highpass.

    We run pedalboard.Reverb in 100%-wet mode (wet=1, dry=0) to obtain a
    pure wet signal, then mix dry+wet ourselves using eff.dry/eff.wet. This
    makes the wet/dry balance explicit and lets us highpass ONLY the wet
    path (so the dry low end is preserved while the reverb tail stays clean).
    """
    plugins = []
    if eff.highpass_hz and eff.highpass_hz > 0:
        plugins.append(PBHighpassFilter(cutoff_frequency_hz=eff.highpass_hz))
    plugins.append(PBReverb(room_size=eff.room_size, damping=eff.damping,
                            wet_level=1.0, dry_level=0.0, width=eff.width))
    wet = _pb_run(plugins, audio, sr)
    n = min(wet.shape[1], audio.shape[1])
    out = np.zeros((audio.shape[0], max(wet.shape[1], audio.shape[1])), dtype=np.float32)
    out[:, :audio.shape[1]] += eff.dry_level * audio
    out[:, :wet.shape[1]] += eff.wet_level * wet
    return out.astype(np.float32)


def _apply_deesser(audio: np.ndarray, eff: "DeEsser", sr: int) -> np.ndarray:
    """Sibilance de-esser via band-limited downward gain reduction.

    1. Isolate the sibilance band [center/√2, center·√2] with a bandpass.
    2. Follow that band's envelope (fast attack / fast release).
    3. Where the band envelope exceeds `threshold_db`, compute a reduction
       red_db = over * (1 - 1/ratio), capped at `range_db` (anti-cheat: the
       band can never be ducked by more than range_db, preserving "s"/"t"
       consonant intelligibility instead of nulling the high end).
    4. Subtract the band scaled by (1 - g_lin) from the full signal, so only
       the sibilance band is attenuated; everything else passes through.
    """
    f_lo = eff.center_hz / np.sqrt(2)
    f_hi = eff.center_hz * np.sqrt(2)
    band = _bandpass(audio, f_lo, f_hi, sr)
    env_db = _envelope_db(band, sr, eff.attack_ms, eff.release_ms)
    over = np.clip(env_db - eff.threshold_db, 0, None)
    red_db = over * (1.0 - 1.0 / max(eff.ratio, 1.001))
    # Anti-cheat: clamp the per-sample reduction depth to range_db.
    red_db = np.clip(red_db, 0.0, eff.range_db)
    g_lin = 10.0 ** (-red_db / 20.0)
    # Remove the attenuated portion of the band only: out = full - band*(1-g)
    return (audio - band * (1.0 - g_lin)).astype(np.float32)


# ===========================================================================
# Differential rendering  (OPT-IN — default behaviour is unchanged)
# ===========================================================================
#
# WHY
# ---
# `_render_full` re-applies every chain from the raw stems on every call, so
# one search step costs O(depth) and a K-step trajectory costs O(K^2). Measured
# on a 220 s song: 0.421 s at depth 0 rising ~0.45 s per accepted effect
# (23.2 s at depth 50). Differential rendering makes one step O(1) in depth.
#
# WHY IT IS BIT-EXACT
# -------------------
# 1. Every mutator on MixState (`add_static` / `add_transient` / `add_section`
#    / `add_master`) *appends* to the end of a tuple: `old + (effect,)`. The
#    prefix is therefore never rewritten.
# 2. `_apply_chain` is a strict left fold, so for any chain C and effect e
#        _apply_chain(x, C + (e,)) == _apply_one(_apply_chain(x, C), e)
#    holds exactly — no re-association of float ops, the same operations run
#    on the same bits in the same order. `_apply_section_chain` is the same
#    fold over `_apply_section_one`.
# 3. Nothing ever writes *through* a cached buffer. Note the precise claim:
#    it is NOT true that every `_apply_one` branch allocates a new array — six
#    branches are degenerate pass-throughs that return their input object
#    unchanged (`GainAutomation`/`PanAutomation`/`WidthAutomation` with empty
#    breakpoints, `TransientShaper(0, 0)`, `SidechainComp` naming a missing
#    key, and `_apply_section_one` with `end_n <= start_n`). What holds is the
#    weaker, sufficient invariant: every branch that *modifies* audio writes
#    into a freshly allocated array (`_pb_run` copies via `.astype`, the numpy
#    branches build new arrays, `_crossfade_segment` / `_apply_sidechain_comp`
#    / `_apply_transient_shaper` `.copy()` or `zeros_like` first), the bus is
#    only ever summed into a fresh `np.zeros`, and `render` hands back
#    `out.copy()`. So a cached buffer can be *read* by many keys but is never
#    the destination of a write.  `_stage` additionally refuses to store a
#    pass-through result (`y is x`), so a buffer is reachable under exactly one
#    key and `store_nbytes` is exact rather than an over-estimate.
#    *If a future change adds an in-place fast path to `_apply_one`, this
#    invariant breaks and the cache silently returns corrupted audio.*
#    `_pb_run` also builds a fresh `Pedalboard` per call, so no plugin state
#    carries over between steps.
# 4. Sidechain keys read `raw_stems`, i.e. the *unprocessed* stems, so there is
#    no cross-stem dependency between processed buffers.
# 5. The bus sum is NOT updated differentially (`bus - old + new` would change
#    the summation order and hence the rounding). It is recomputed by summing
#    the per-stem buffers in `state.stems` iteration order — byte for byte the
#    order `_render_full` uses. It is cheap and depth-independent.
#
# WHAT IS CACHED
# --------------
# Per track, one buffer per *chain prefix* at each of the three stem stages
# (static → transient → section). Keys are the chains themselves (frozen
# dataclasses of floats/strings/tuples → hashable by value), chained with the
# upstream key so that an entry can only be reused when everything upstream of
# it is identical. Keeping *prefixes* (not just the newest state) is what lets
# a rejected candidate fall back to its parent without a full rebuild.
#
# The bus and the master chain are deliberately NOT kept per prefix:
#   * The bus lives in its own 2-slot store, not the main LRU. A bus is only
#     reusable when the entire set of stem buffers repeats (a master-only
#     action), so one per state would pile up ~78 MB of dead weight per step
#     and evict the prefixes that matter.
#   * The master stage stores only the LAST buffer of its fold. Every stem edit
#     produces a new bus and hence a whole new master prefix family, whose
#     intermediates can never be reused — the next step has a different bus
#     again. Only the final buffer has a future (a master-append resumes from
#     it). Measured on a 220 s song with a 14-deep master chain: 15 buffers of
#     garbage per step vs 1, and under a realistic budget that was the
#     difference between 15 and 57 effect applications per step.
#
# Working set in the search loop is therefore small and depth-independent:
# one final prefix per stem (probed on every render, so the LRU always keeps
# them), one master final, two buses. ~10 buffers, i.e. ~800 MB on a 220 s
# 4-stem song. The default 8 GiB budget has a wide margin.
#
# LIMITS  (read this before enabling it anywhere)
# ---------------------------------------------
# * The cache binds to ONE stems dict; it is dropped and rebuilt if `render`
#   is called with different stems or a different sample rate.
# * >>> Consequence: interleaving renders of *different* stem sets on one
#   cache destroys it on every call. This is not hypothetical. Both
#   `strategies/knowledge_base_mix._render_single_stem` and
#   `dsp/stem_lufs_cache` render a SOLO MixState holding a single stem. If
#   those calls share a cache with the full-mix render, every step pays a full
#   rebuild — measured 49x *slower* than no cache at all, while staying
#   bit-identical, i.e. it fails silently. That is exactly why
#   `MIXORCH_RENDER_CACHE=1` (one implicit cache for everything on the thread)
#   is a debugging switch only. **Production code should pass an explicit
#   `cache=` per rendering domain** (one for the full mix, one per solo stem);
#   see `experiments/agreement_loop_all.py --render-cache`. `RenderCache`
#   prints a one-shot warning when it detects this thrashing pattern.
# * It assumes stem arrays are not mutated in place after the first render
#   (nothing in this codebase does; MixState treats them as immutable).
# * A `RenderCache` instance is not thread-safe. The implicit env-var cache is
#   thread-local, so threads never share one.

_ENV_ENABLE = "MIXORCH_RENDER_CACHE"
_ENV_BUDGET_MB = "MIXORCH_RENDER_CACHE_MB"
_DEFAULT_BUDGET_MB = 8192.0
_TRUTHY = {"1", "true", "yes", "on"}

_thread_local = threading.local()


def _stems_unchanged(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    """True if `b` denotes the same stems as `a` (same keys, same order, same
    array objects). MixState._evolve keeps the stems dict by reference, so the
    fast `a is b` path is the normal case."""
    if a is b:
        return True
    if len(a) != len(b):
        return False
    ka = list(a.keys())
    if ka != list(b.keys()):
        return False
    return all(a[k] is b[k] for k in ka)


#: A cache that rebinds this often, relative to its render count, is being
#: shared between rendering domains (full mix vs. solo stem) and is a net loss.
_THRASH_MIN_REBINDS = 8
_THRASH_RATIO = 0.5


class RenderCache:
    """Reusable per-*rendering-domain* buffer cache for :func:`render`.

    Usage::

        cache = RenderCache()
        audio = render(state, cache=cache)        # same bits as render(state)

    One instance per (song, stem set). Reusing it across songs is safe — it
    notices the stems changed and resets — but pointless. Reusing it across
    *stem sets* (e.g. the full mix and the solo-stem renders that
    `stem_lufs_cache` performs) is also safe but actively harmful: every call
    resets the cache, so give those their own instance. `n_rebind` counts the
    resets and a one-shot warning fires once the pattern is unmistakable.
    """

    def __init__(self, max_bytes: Optional[int] = None):
        if max_bytes is None:
            try:
                mb = float(os.environ.get(_ENV_BUDGET_MB, _DEFAULT_BUDGET_MB))
            except ValueError:
                mb = _DEFAULT_BUDGET_MB
            max_bytes = int(mb * 1024 * 1024)
        self.max_bytes = int(max_bytes)
        self._store: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self._bytes = 0
        # Summed buses live in their own tiny slot rather than the main LRU: a
        # bus is only reusable when the *entire* set of stem buffers repeats
        # (i.e. a master-only action), so one per state would otherwise pile up
        # ~78 MB of dead weight per step and evict the prefixes that matter.
        self._bus: "OrderedDict[Any, np.ndarray]" = OrderedDict()
        self._bus_max = 2
        self._stems: Optional[Dict[str, np.ndarray]] = None
        self._sample_rate: Optional[int] = None
        self._raw_stems: Optional[Dict[str, np.ndarray]] = None
        # stats
        self.n_render = 0
        self.n_apply = 0        # effects actually evaluated (the real cost)
        self.n_reuse = 0        # effects skipped thanks to a cached prefix
        self.n_evict = 0
        self.n_rebind = 0
        self._warned_thrash = False
        # Where the residual per-step cost goes. An append to a track's static
        # chain invalidates that track's transient+section stages, and any stem
        # change invalidates the whole master chain, so those stages are the
        # only ones that can still grow with depth.
        self.n_apply_by_stage: Dict[str, int] = {
            "static": 0, "transient": 0, "section": 0, "master": 0}

    # ---------- store ----------

    def clear(self) -> None:
        self._store.clear()
        self._bus.clear()
        self._bytes = 0
        self._stems = None
        self._sample_rate = None
        self._raw_stems = None

    @property
    def store_nbytes(self) -> int:
        """Bytes in the LRU prefix store — this is what `max_bytes` caps.

        Exact, not an estimate: `_stage` never stores a buffer that is merely
        an alias of its input, so distinct keys always hold distinct arrays.
        """
        return self._bytes

    @property
    def nbytes(self) -> int:
        """Bytes held by the prefix store plus the bus slot. The bus slot is
        bounded by entry count (`_bus_max`), not by `max_bytes`, so this can
        sit slightly above the budget. Raw stems are excluded: they are needed
        on every render and are never evicted."""
        return self._bytes + sum(a.nbytes for a in self._bus.values())

    def stats(self) -> Dict[str, int]:
        return {"n_render": self.n_render, "n_apply": self.n_apply,
                "n_reuse": self.n_reuse, "n_evict": self.n_evict,
                "n_rebind": self.n_rebind, "n_entries": len(self._store),
                "nbytes": self.nbytes}

    def _bus_get(self, key) -> Optional[np.ndarray]:
        arr = self._bus.get(key)
        if arr is not None:
            self._bus.move_to_end(key)
        return arr

    def _bus_put(self, key, arr: np.ndarray) -> None:
        self._bus[key] = arr
        self._bus.move_to_end(key)
        while len(self._bus) > self._bus_max:
            self._bus.popitem(last=False)

    def _probe(self, key) -> Optional[np.ndarray]:
        arr = self._store.get(key)
        if arr is not None:
            self._store.move_to_end(key)
        return arr

    def _put(self, key, arr: np.ndarray) -> None:
        old = self._store.pop(key, None)
        if old is not None:
            self._bytes -= old.nbytes
        self._store[key] = arr
        self._bytes += arr.nbytes
        while self._bytes > self.max_bytes and len(self._store) > 1:
            _, victim = self._store.popitem(last=False)
            self._bytes -= victim.nbytes
            self.n_evict += 1

    # ---------- binding ----------

    def _warn_if_thrashing(self) -> None:
        """Say so, once, when this cache is being reset more often than used.

        A cache shared between the full mix and solo-stem renders stays
        bit-exact but costs a full rebuild per call — it can only be noticed as
        "the run got slower", which is precisely the kind of regression that
        goes unnoticed for days. Make it say something instead.
        """
        if self._warned_thrash or self.n_rebind < _THRASH_MIN_REBINDS:
            return
        if self.n_rebind <= _THRASH_RATIO * max(self.n_render, 1):
            return
        self._warned_thrash = True
        import sys as _sys
        print(
            f"[RenderCache] WARNING: rebound {self.n_rebind} times in "
            f"{self.n_render} renders. This cache is being shared between "
            f"different stem sets (e.g. the full mix and solo-stem renders); "
            f"every call rebuilds it, so it is SLOWER than no cache. Results "
            f"stay bit-identical. Give each rendering domain its own "
            f"RenderCache, or stop using MIXORCH_RENDER_CACHE=1 (one implicit "
            f"cache per thread) and pass cache= explicitly.",
            file=_sys.stderr, flush=True)

    def _bind(self, state: MixState) -> Dict[str, np.ndarray]:
        if (self._raw_stems is not None
                and self._sample_rate == state.sample_rate
                and _stems_unchanged(self._stems, state.stems)):
            return self._raw_stems
        if self._raw_stems is not None:
            self.n_rebind += 1
            self._warn_if_thrashing()
        self.clear()
        self._stems = state.stems          # strong ref: pins the arrays alive
        self._sample_rate = state.sample_rate
        # Identical expression to `_render_full`'s `raw_stems`; the per-track
        # stage input is the same expression again, so one buffer serves both.
        self._raw_stems = {k: _to_stereo(v.astype(np.float32))
                           for k, v in state.stems.items()}
        return self._raw_stems

    # ---------- staged fold ----------

    def _stage(self, stage: str, trk: str, base: np.ndarray, base_key,
               chain, apply_fn, keep_intermediates: bool = True
               ) -> Tuple[np.ndarray, Any]:
        """Fold `chain` over `base`, resuming from the longest cached prefix.

        Returns (buffer, key). An empty chain is a pass-through: the upstream
        buffer and key are returned as-is so nothing is stored twice.

        `keep_intermediates=False` stores only the *last* buffer of the fold
        instead of one per prefix. Used for the master stage, where every stem
        edit produces a brand-new bus and therefore a brand-new prefix family:
        the intermediates of that family can never be reused (the next step has
        a different bus again), so storing them is pure garbage that evicts the
        stem prefixes which *are* reused. Only the final buffer has a future —
        a master-append reuses it. Measured on a 220 s song with a 14-deep
        master chain: this is 15 buffers/step of garbage vs 1, and under a
        realistic budget it was the difference between 15 and 57 effect
        applications per step.
        """
        if not chain:
            return base, base_key
        prefix = (stage, trk, base_key)
        n = len(chain)
        x = base
        start = 0
        for i in range(n, 0, -1):
            hit = self._probe((prefix, chain[:i]))
            if hit is not None:
                x, start = hit, i
                break
        self.n_reuse += start
        for j in range(start, n):
            y = apply_fn(x, chain[j])
            self.n_apply += 1
            self.n_apply_by_stage[stage] += 1
            # Degenerate effects (empty automation breakpoints, a zero-width
            # section, a sidechain whose key stem is absent, ...) return their
            # input object. Storing that would make two keys alias one buffer,
            # double-counting `_bytes` and muddying the "one key, one array"
            # invariant the correctness argument rests on. Re-running a
            # pass-through on the next render costs nothing, so just skip it.
            if y is not x and (keep_intermediates or j == n - 1):
                self._put((prefix, chain[:j + 1]), y)
            x = y
        return x, (prefix, chain)


def _global_cache_if_enabled() -> Optional[RenderCache]:
    """Thread-local implicit cache, enabled by ``MIXORCH_RENDER_CACHE=1``.

    DEBUGGING SWITCH ONLY. One cache serves every `render` on the thread,
    including the solo-stem renders in `knowledge_base_mix` and
    `stem_lufs_cache`; those bind a different stem set and reset it on every
    call, which is bit-identical but ~49x slower than not caching at all.
    Production code passes `cache=` explicitly, one instance per stem set.
    """
    if os.environ.get(_ENV_ENABLE, "").strip().lower() not in _TRUTHY:
        return None
    cache = getattr(_thread_local, "render_cache", None)
    if cache is None:
        cache = RenderCache()
        _thread_local.render_cache = cache
    return cache


def _render_cached(state: MixState, cache: RenderCache) -> np.ndarray:
    """Differential render. Bit-identical to `_render_full(state)`."""
    if not HAS_PEDALBOARD:
        raise RuntimeError("pedalboard is not installed. `pip install pedalboard`.")

    sr = state.sample_rate
    raw_stems = cache._bind(state)
    cache.n_render += 1

    # Step 1: per-stem chains, resumed from the deepest cached prefix.
    processed: List[np.ndarray] = []
    stem_keys: List[Any] = []
    for trk in state.stems:
        x = raw_stems[trk]
        key: Any = ("raw", trk)
        x, key = cache._stage(
            "static", trk, x, key, state.static_processors.get(trk, ()),
            lambda a, e, _t=trk: _apply_one(a, e, sr, trk_name=_t,
                                            all_stems=raw_stems))
        x, key = cache._stage(
            "transient", trk, x, key, state.transient_processors.get(trk, ()),
            lambda a, e, _t=trk: _apply_one(a, e, sr, trk_name=_t,
                                            all_stems=raw_stems))
        x, key = cache._stage(
            "section", trk, x, key, state.section_processors.get(trk, ()),
            lambda a, e: _apply_section_one(a, e, sr))
        processed.append(x)
        stem_keys.append(key)

    # Step 2: sum to bus — recomputed, never patched, so the float summation
    # order matches `_render_full` exactly.
    bus_key = ("bus", tuple(stem_keys))
    bus = cache._bus_get(bus_key)
    if bus is None:
        max_len = max(p.shape[1] for p in processed)
        bus = np.zeros((2, max_len), dtype=np.float32)
        for x in processed:
            n = x.shape[1]
            bus[:, :n] += x
        cache._bus_put(bus_key, bus)

    # Step 3: master chain. Only the final buffer is kept — see `_stage`.
    out, _ = cache._stage(
        "master", "master", bus, bus_key, state.master_chain,
        lambda a, e: _apply_one(a, e, sr, trk_name="master"),
        keep_intermediates=False)
    # `_render_full` always hands back a freshly allocated array; copy so a
    # caller mutating the result cannot corrupt the cache.
    return out.copy()
