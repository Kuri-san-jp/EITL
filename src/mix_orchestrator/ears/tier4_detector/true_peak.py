"""True-peak estimation via 4x oversampling."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult


class TruePeakDetector(Ear):
    name = "true_peak"
    tier = 4
    cost_per_call_usd = 0.0

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference=None, window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        audio = _slice(audio, sr, window)
        if audio.size == 0:
            return EarResult(name=self.name, score={"true_peak_db": -np.inf},
                             elapsed_sec=time.time()-t0)
        # 4x upsample for inter-sample peak
        up = _polyphase_upsample(audio, factor=4)
        peak = float(np.max(np.abs(up)))
        peak_db = 20.0 * np.log10(peak + 1e-12)
        sample_peak = float(np.max(np.abs(audio)))
        return EarResult(
            name=self.name,
            score={"true_peak_db": peak_db, "sample_peak_db": 20*np.log10(sample_peak + 1e-12)},
            elapsed_sec=time.time() - t0,
        )


def _polyphase_upsample(x: np.ndarray, factor: int = 4) -> np.ndarray:
    """Cheap zero-stuff + lowpass for approximate true-peak. Fine for screening."""
    from scipy.signal import resample_poly
    if x.ndim == 1:
        return resample_poly(x, factor, 1)
    out = []
    for ch in range(x.shape[0]):
        out.append(resample_poly(x[ch], factor, 1))
    return np.stack(out)


def _slice(audio, sr, window):
    if window is None:
        return audio
    s, e = window
    s_n = int(max(0, s) * sr); e_n = int(e * sr)
    return audio[:, s_n:e_n] if audio.ndim == 2 else audio[s_n:e_n]
