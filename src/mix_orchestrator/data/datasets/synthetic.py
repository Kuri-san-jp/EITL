"""Synthetic multitrack stems for smoke testing.

Generates 3 deterministically-seeded stems (drums, bass, vocal) over
`duration_sec` seconds at 44.1 kHz. NOT music — just enough for the
pipeline to run with realistic spectral content.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def synthetic_stems(duration_sec: float = 6.0,
                    sr: int = 44100,
                    seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    t = np.arange(n) / sr

    # -------- drums: noise bursts on a 4-on-the-floor pattern ----
    bpm = 110
    beat_period = 60.0 / bpm
    drum = np.zeros(n, dtype=np.float32)
    for k in range(int(duration_sec / beat_period) + 1):
        i = int(k * beat_period * sr)
        if i >= n:
            break
        burst_len = int(0.05 * sr)
        env = np.exp(-np.arange(burst_len) / (0.01 * sr))
        burst = rng.standard_normal(burst_len).astype(np.float32) * env
        end = min(i + burst_len, n)
        drum[i:end] += burst[: end - i] * 0.25

    # -------- bass: sub-frequency square-ish ----
    bass_freq = 80.0
    bass_env = 0.5 * (1 + np.sin(2 * np.pi * t / beat_period / 2))   # rolling pulse
    bass = np.sin(2 * np.pi * bass_freq * t).astype(np.float32) * bass_env * 0.15

    # -------- vocal: sweep around 220-440 Hz with vibrato ----
    vib = 5.0 * np.sin(2 * np.pi * 5.0 * t)        # ±5 Hz vibrato @ 5Hz
    f_inst = 220 + 220 * 0.5 * (1 + np.sin(2 * np.pi * 0.2 * t)) + vib
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    vocal = np.sin(phase).astype(np.float32) * 0.12
    # gentle amplitude modulation
    vocal *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)).astype(np.float32)

    # Each as mono (1, N) so renderer makes them stereo
    return {
        "drums": drum[None, :],
        "bass":  bass[None, :],
        "vocal": vocal[None, :],
    }
