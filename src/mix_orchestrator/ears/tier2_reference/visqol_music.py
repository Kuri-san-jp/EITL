"""ViSQOL music-mode reference-based ear.

Requires Google's `visqol` CLI/python binding. Falls back to a PESQ-style
SNR proxy if visqol is unavailable.
"""
from __future__ import annotations

import os
import time
import tempfile
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class ViSQOLMusicEar(Ear):
    name = "visqol_music"
    tier = 2
    cost_per_call_usd = 0.0
    requires_reference = True

    def __init__(self, use_proxy_if_missing: bool = False):
        self.use_proxy = use_proxy_if_missing

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference: Optional[np.ndarray] = None,
                       window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        if reference is None:
            return EarResult(name=self.name, score=0.0,
                             elapsed_sec=time.time()-t0,
                             warnings=["no reference; returning 0"])
        try:
            score = self._real(audio, reference, sr)
            return EarResult(name=self.name, score={"moslqo": float(score)},
                             elapsed_sec=time.time()-t0)
        except Exception as ex:                                            # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, reference, t0, warning=f"visqol failed: {ex}")

    def _real(self, deg: np.ndarray, ref: np.ndarray, sr: int) -> float:
        try:
            from visqol import visqol_lib_py
            from visqol.pb2 import visqol_config_pb2, similarity_result_pb2
        except ImportError as ex:
            raise ImportError("`pip install visqol` (Google)") from ex
        cfg = visqol_config_pb2.VisqolConfig()
        cfg.audio.sample_rate = 48000
        cfg.options.use_speech_scoring = False
        cfg.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__),
            "model", "libsvm_nu_svr_model.txt")
        api = visqol_lib_py.VisqolApi()
        api.Create(cfg)
        # Resample to 48k mono
        deg_m = deg.mean(axis=0) if deg.ndim == 2 else deg
        ref_m = ref.mean(axis=0) if ref.ndim == 2 else ref
        if sr != 48000:
            import librosa
            deg_m = librosa.resample(deg_m.astype(np.float32), orig_sr=sr, target_sr=48000)
            ref_m = librosa.resample(ref_m.astype(np.float32), orig_sr=sr, target_sr=48000)
        result = api.Measure(ref_m.astype(np.float64), deg_m.astype(np.float64))
        return float(result.moslqo)

    def _proxy(self, audio, reference, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        n = min(audio.shape[-1], reference.shape[-1])
        a = audio[..., :n] if audio.ndim == 2 else audio[:n]
        r = reference[..., :n] if reference.ndim == 2 else reference[:n]
        if a.ndim == 2:
            a = a.mean(axis=0)
        if r.ndim == 2:
            r = r.mean(axis=0)
        num = float(np.sum(r ** 2))
        den = float(np.sum((r - a) ** 2)) + 1e-9
        sdr = 10.0 * np.log10((num + 1e-9) / den)
        # Map SDR to a pseudo MOSLQO: clamp to [1, 5]
        moslqo = float(np.clip(1 + (sdr + 5) / 10, 1, 5))
        return EarResult(name=self.name, score={"moslqo": moslqo},
                         elapsed_sec=time.time()-t0, warnings=[warning])
