"""Stereo correlation + width descriptors."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult


def _stereo_quality(corr: float) -> float:
    """Map stereo correlation to a mixing quality score in [0,1] (target band).

    Neither high correlation (close to mono, corr->1) nor excessive
    decorrelation / out-of-phase content (corr<0) is desirable. This is a
    plateau shape that treats moderate width (corr ~0.3-0.9) as best.
    : the old reward rewarded corr linearly and so mis-scored
    "closer to mono = better", hence the switch to a target band (internal notes).
    """
    c = float(corr)
    if 0.3 <= c <= 0.9:
        return 1.0
    if c > 0.9:                              # too mono
        return max(0.5, 1.0 - (c - 0.9) * 5.0)      # corr=1.0 → 0.5
    if c >= 0.0:                             # 0..0.3 narrow
        return 0.5 + (c / 0.3) * 0.5                 # corr=0 → 0.5, 0.3 → 1.0
    return max(0.0, 0.5 + c)                 # out of phase: corr=-0.5 → 0.0


class StereoCorrelationDetector(Ear):
    name = "stereo"
    tier = 4
    cost_per_call_usd = 0.0

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference=None, window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        audio = _slice(audio, sr, window)
        if audio.ndim != 2 or audio.shape[0] != 2:
            return EarResult(name=self.name,
                             score={"stereo_correlation": 1.0, "ms_ratio_db": 0.0,
                                    "stereo_quality": _stereo_quality(1.0)},
                             elapsed_sec=time.time()-t0,
                             warnings=["audio is not stereo"])
        l, r = audio[0], audio[1]
        if np.std(l) < 1e-7 or np.std(r) < 1e-7:
            corr = 1.0
        else:
            corr = float(np.corrcoef(l, r)[0, 1])
            if not np.isfinite(corr):
                corr = 1.0
        m = 0.5 * (l + r)
        s = 0.5 * (l - r)
        rms_m = float(np.sqrt(np.mean(m**2) + 1e-12))
        rms_s = float(np.sqrt(np.mean(s**2) + 1e-12))
        ms_ratio_db = 20.0 * np.log10((rms_s + 1e-12) / (rms_m + 1e-12))
        return EarResult(
            name=self.name,
            score={"stereo_correlation": corr, "ms_ratio_db": ms_ratio_db,
                   "stereo_quality": _stereo_quality(corr)},
            elapsed_sec=time.time() - t0,
        )


def _slice(audio, sr, window):
    if window is None:
        return audio
    s, e = window
    return audio[:, int(s*sr):int(e*sr)] if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]
