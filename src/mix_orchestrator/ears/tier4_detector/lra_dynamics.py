"""Loudness Range (LRA) via pyloudnorm short-term windows."""
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


class LRADetector(Ear):
    name = "lra"
    tier = 4
    cost_per_call_usd = 0.0
    supports_temporal = True

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference=None, window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        if window is not None:
            s, e = window
            audio = audio[:, int(s*sr):int(e*sr)] if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]
        if not HAS_PYLOUDNORM or audio.size == 0:
            return EarResult(name=self.name, score={"lra": 0.0},
                             elapsed_sec=time.time()-t0,
                             warnings=["pyloudnorm unavailable"])
        # Approximate LRA: 3s short-term LUFS, take p95 - p10 (BS.1770-4 spec uses 95-10)
        arr = audio.T if audio.ndim == 2 else audio
        if arr.ndim == 1:
            arr = arr[:, None]
        meter = pyln.Meter(sr)
        win = 3 * sr
        hop = sr  # 1s hop
        scores = []
        n = arr.shape[0]
        for start in range(0, max(1, n - win), hop):
            seg = arr[start:start+win]
            if len(seg) < sr:  # too short
                continue
            try:
                lu = meter.integrated_loudness(seg)
            except ValueError:
                continue
            if np.isfinite(lu):
                scores.append(lu)
        if len(scores) < 2:
            return EarResult(name=self.name, score={"lra": 0.0},
                             elapsed_sec=time.time()-t0)
        s_arr = np.array(scores)
        lra = float(np.percentile(s_arr, 95) - np.percentile(s_arr, 10))
        return EarResult(
            name=self.name,
            score={"lra": lra, "short_term_lufs_mean": float(s_arr.mean())},
            frame_scores=s_arr,
            elapsed_sec=time.time() - t0,
        )
