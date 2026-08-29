"""Loudness normalization utilities applied before evaluation.

When comparing mix candidates, unmatched loudness or peak levels create an
artifact where a candidate that is simply louder / has a higher peak is favored
or penalized by the quality metrics (in the  dry probe, MEGAMI took a
peak penalty because it normalizes to 0 dBFS; internal notes).

This util aligns all candidates to a common integrated LUFS and scales down only
when the true-peak exceeds the ceiling, preventing clipping. That way the ear
evaluation measures differences in audio quality and mix content rather than
differences in loudness/peak.
"""
from __future__ import annotations

import numpy as np


def _measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Return the integrated LUFS. `audio` is (C, N) or (N,)."""
    import pyloudnorm as pyln
    x = audio.T if (audio.ndim == 2) else audio   # pyln: (N,) or (N, C)
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(x))


def _true_peak_db(audio: np.ndarray) -> float:
    """Approximate true-peak (sample peak) in dBFS."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20.0 * np.log10(peak + 1e-12)


def equal_lufs_mix(stems: dict, sr: int,
                   stem_lufs: float = -23.0,
                   silence_floor_lufs: float = -70.0) -> np.ndarray:
    """Simple baseline that aligns every stem to the same integrated LUFS, then sums.

    The most naive "just match the loudness" mixing method. It applies no EQ / pan /
    comp / reverb whatsoever; it only applies a gain that brings each stem's
    integrated LUFS to ``stem_lufs`` and sums the result. Used as a naive baseline
    (lower bound) to show whether the proposed method / MEGAMI / P2 / P6 do anything
    beyond matching loudness.

    Difference from the dry baseline (plain sum): dry adds the stems at their recorded
    levels, so originally loud stems dominate. equal-LUFS brings every stem to equal
    loudness, giving a "flat" mix that does not depend on recording-level differences.

    Args:
        stems: {name: ndarray(C, N) or (N,)} of dry stems (do not include the mixture).
        sr: sample rate.
        stem_lufs: integrated LUFS every stem is aligned to (default -23, a level high
                   enough that each stem can be measured individually). The master
                   loudness of the sum is expected to be aligned to a common LUFS by
                   the caller via normalize_for_eval.
        silence_floor_lufs: stems whose measured LUFS is at or below this value
                   (= effectively silent) are passed through without gain (so we do not
                   blow up silence with a huge gain).

    Returns:
        (C, N) float32 master mix (raw; loudness normalization is up to the caller).
    """
    if not stems:
        raise ValueError("equal_lufs_mix requires at least 1 stem.")
    arrs = []
    max_n = 0
    max_c = 1
    for v in stems.values():
        x = np.asarray(v, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        arrs.append(x)
        max_n = max(max_n, x.shape[-1])
        max_c = max(max_c, x.shape[0])
    acc = np.zeros((max_c, max_n), dtype=np.float32)
    for x in arrs:
        try:
            loud = _measure_lufs(x, sr)
        except Exception:                                   # noqa: BLE001
            loud = float("nan")
        if np.isfinite(loud) and loud > silence_floor_lufs:
            gain = 10.0 ** ((stem_lufs - loud) / 20.0)
            x = x * gain
        # Channel broadcast (mono stem -> max_c channels) and length alignment.
        if x.shape[0] < max_c:
            x = np.repeat(x, max_c, axis=0)
        if x.shape[-1] < max_n:
            x = np.pad(x, ((0, 0), (0, max_n - x.shape[-1])))
        acc[:, :] += x[:max_c, :max_n]
    return acc.astype(np.float32)


def common_gain_to_lufs(stems: dict, sr: int,
                        target_lufs: float = -14.0,
                        silence_floor_lufs: float = -70.0) -> "tuple":
    """Apply a **common scalar gain** to all stems so that their plain sum (the raw mix) hits target_lufs.

    A util for normalizing the absolute level in the randomized-stem recovery
    experiment, after each stem has been disrupted by a per-stem random gain. Because
    the gain is a common scalar, the **relative balance between stems (= the disruption
    ratio g_i/g_j) is completely unchanged**; only the absolute level (master loudness)
    is normalized.

    Purpose: prevent models like MEGAMI, which decide silence from the absolute dBFS of
    the input stems, from dropping a stem that the disruption made absolutely quiet as
    silent and then crashing on an empty tensor. The output is re-normalized downstream
    by normalize_for_eval(-14), so this common normalization is neutral with respect to
    the relative-balance-disruption evaluation (it does not change what the results mean).

    Args:
        stems: {name: ndarray(C, N) or (N,)} of (already disrupted) stems.
        sr: sample rate.
        target_lufs: integrated LUFS the sum is aligned to (default -14, same as the
                   evaluation target).
        silence_floor_lufs: if the LUFS of the sum is at or below this value
                   (= effectively silent), use gain=1.0 (divergence guard).

    Returns:
        (normalized_stems: dict[name -> float32 with the original shape], common_gain: float).
    """
    if not stems:
        raise ValueError("common_gain_to_lufs requires at least 1 stem.")
    arrs = []
    max_n = 0
    max_c = 1
    for v in stems.values():
        x = np.asarray(v, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        arrs.append(x)
        max_n = max(max_n, x.shape[-1])
        max_c = max(max_c, x.shape[0])
    acc = np.zeros((max_c, max_n), dtype=np.float32)
    for x in arrs:
        xb = np.repeat(x, max_c, axis=0) if x.shape[0] < max_c else x
        if xb.shape[-1] < max_n:
            xb = np.pad(xb, ((0, 0), (0, max_n - xb.shape[-1])))
        acc[:, :] += xb[:max_c, :max_n]
    try:
        loud = _measure_lufs(acc, sr)
    except Exception:                                       # noqa: BLE001
        loud = float("nan")
    if np.isfinite(loud) and loud > silence_floor_lufs:
        c = float(10.0 ** ((target_lufs - loud) / 20.0))
    else:
        c = 1.0
    out = {k: (np.asarray(v, dtype=np.float32) * c).astype(np.float32)
           for k, v in stems.items()}
    return out, c


def normalize_for_eval(audio: np.ndarray, sr: int,
                       target_lufs: float = -14.0,
                       peak_ceiling_db: float = -1.0) -> np.ndarray:
    """Normalize to a common loudness and, if needed, hold the peak below the ceiling.

    Args:
        audio: (C, N) float32.
        sr: sample rate.
        target_lufs: integrated LUFS to align to (default -14).
        peak_ceiling_db: scale down if the true-peak exceeds this after normalization.

    Returns:
        Normalized audio (C, N) float32. The original shape is preserved.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    try:
        loud = _measure_lufs(x, sr)
    except Exception:                                   # noqa: BLE001
        loud = float("nan")
    if np.isfinite(loud) and loud > -70.0:
        gain = 10.0 ** ((target_lufs - loud) / 20.0)
        x = x * gain
    # peak protection
    pk = _true_peak_db(x)
    if pk > peak_ceiling_db:
        x = x * 10.0 ** ((peak_ceiling_db - pk) / 20.0)
    return x.astype(np.float32)
