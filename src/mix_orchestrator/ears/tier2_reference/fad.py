"""Frechet Audio Distance.

Real implementation uses `fadtk` (Google) or `frechet_audio_distance`.
Without it, falls back to a Gaussian-stats spectral distance proxy.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class FADEar(Ear):
    name = "fad"
    tier = 2
    cost_per_call_usd = 0.0
    requires_reference = True
    uses_gpu_model = True

    def __init__(self, model_id: str = "vggish",
                 device: str = "cuda",
                 use_proxy_if_missing: bool = False):
        self.model_id = model_id
        self.device = device
        self.use_proxy = use_proxy_if_missing
        self._model = None

    def release_model(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                          # noqa: BLE001
            pass

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference: Optional[np.ndarray] = None,
                       window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        if reference is None:
            return EarResult(name=self.name, score=0.0,
                             elapsed_sec=time.time()-t0, warnings=["no reference"])
        try:
            d = self._real(audio, reference, sr)
            return EarResult(name=self.name, score={"fad": float(d)},
                             elapsed_sec=time.time()-t0)
        except Exception as ex:                                            # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, reference, t0, warning=f"fad failed: {ex}")

    def _real(self, audio, ref, sr) -> float:
        try:
            from frechet_audio_distance import FrechetAudioDistance
        except ImportError as ex:
            raise ImportError("`pip install frechet-audio-distance`") from ex
        import tempfile, soundfile as sf, os
        if self._model is None:
            self._model = FrechetAudioDistance(model_name=self.model_id,
                                               use_pca=False, use_activation=False, verbose=False)
        # FAD expects directories; write degraded + reference to temp dirs
        with tempfile.TemporaryDirectory() as deg_dir, tempfile.TemporaryDirectory() as ref_dir:
            for name, x, dest in [("deg.wav", audio, deg_dir), ("ref.wav", ref, ref_dir)]:
                arr = x.T if x.ndim == 2 else x
                sf.write(os.path.join(dest, name), arr, sr)
            return float(self._model.score(ref_dir, deg_dir))

    def _proxy(self, audio, ref, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        # Fallback computation: KL divergence between the log-spectrum distributions
        from numpy.fft import rfft
        def logspec(x):
            x = x.mean(axis=0) if x.ndim == 2 else x
            x = x[:min(len(x), 44100 * 5)]
            spec = np.abs(rfft(x)) + 1e-12
            return np.log(spec / spec.sum())
        la = logspec(audio); lr = logspec(ref)
        n = min(len(la), len(lr))
        kl = float(np.sum(np.exp(la[:n]) * (la[:n] - lr[:n])))
        return EarResult(name=self.name, score={"fad": kl},
                         elapsed_sec=time.time()-t0, warnings=[warning])
