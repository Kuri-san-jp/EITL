"""Heuristic proxy for any not-yet-installed MOS predictor.

Returns a value in roughly [1.0, 5.0] that correlates weakly with audio
quality (RMS-based). NEVER use this for reported results — its only
purpose is to keep the pipeline runnable without the real model weights.
"""
from __future__ import annotations

import numpy as np


def proxy_mos_score(audio: np.ndarray, sr: int) -> float:
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    if mono.size == 0:
        return 1.0
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
    crest = float(np.max(np.abs(mono)) / (rms + 1e-9))
    # rms in [-30..-12] dBFS maps to MOS 1..5
    rms_db = 20.0 * np.log10(rms + 1e-12)
    rms_score = float(np.clip((rms_db + 30) / 18.0, 0.0, 1.0))
    # crest 6-15 considered good
    crest_score = float(np.clip(1.0 - abs(crest - 10) / 10.0, 0.0, 1.0))
    return 1.0 + 4.0 * (0.6 * rms_score + 0.4 * crest_score)
