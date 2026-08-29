"""Integrated LUFS (ITU-R BS.1770) via pyloudnorm."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False


class LoudnessDetector(Ear):
    name = "lufs"
    tier = 4
    cost_per_call_usd = 0.0

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference=None, window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        audio = _slice(audio, sr, window)
        if not HAS_PYLOUDNORM:
            # Fallback: simple RMS in dB
            rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
            score = 20 * np.log10(rms + 1e-12)
            return EarResult(name=self.name, score={"integrated_lufs": float(score)},
                             elapsed_sec=time.time()-t0,
                             warnings=["pyloudnorm not installed, returned dBFS RMS"])
        meter = pyln.Meter(sr)
        # pyloudnorm expects (N, C)
        arr = audio.T if audio.ndim == 2 else audio
        if arr.ndim == 1:
            arr = arr[:, None]
        try:
            integrated = float(meter.integrated_loudness(arr))
        except ValueError:
            integrated = float("-inf")
        return EarResult(
            name=self.name,
            score={"integrated_lufs": integrated},
            elapsed_sec=time.time() - t0,
        )


def _slice(audio: np.ndarray, sr: int, window) -> np.ndarray:
    if window is None:
        return audio
    s, e = window
    s_n = int(max(0, s) * sr)
    e_n = int(e * sr)
    if audio.ndim == 1:
        return audio[s_n:e_n]
    return audio[:, s_n:e_n]
