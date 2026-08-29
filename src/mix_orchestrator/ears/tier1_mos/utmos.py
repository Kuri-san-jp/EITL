"""UTMOS wrapper (Saeki et al., 2022)."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed
from ._proxy_mos import proxy_mos_score


_UTMOS_MODEL = None       # cached at module level


class UTMOSEar(Ear):
    name = "utmos"
    tier = 1
    cost_per_call_usd = 0.0
    uses_gpu_model = True

    def __init__(self, use_proxy_if_missing: bool = False):
        self.use_proxy = use_proxy_if_missing

    def release_model(self) -> None:
        global _UTMOS_MODEL
        _UTMOS_MODEL = None
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
        if window is not None:
            s, e = window
            audio = audio[:, int(s*sr):int(e*sr)] if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]
        try:
            score = self._real(audio, sr)
            warnings = []
        except Exception as ex:                                    # noqa: BLE001
            if not self.use_proxy:
                raise
            assert_proxy_allowed(self.name)
            score = proxy_mos_score(audio, sr)
            warnings = [f"UTMOS fallback proxy: {ex}"]
        return EarResult(name=self.name, score=float(score),
                         elapsed_sec=time.time()-t0, warnings=warnings)

    def _real(self, audio, sr) -> float:
        """UTMOS via torch.hub.

        `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")` fetches
        the model and its weights. Expects 16 kHz mono.
        """
        import torch
        global _UTMOS_MODEL
        if _UTMOS_MODEL is None:
            _UTMOS_MODEL = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                trust_repo=True)
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if sr != 16000:
            import librosa
            mono = librosa.resample(mono.astype(np.float32),
                                    orig_sr=sr, target_sr=16000)
        x = torch.from_numpy(np.asarray(mono, dtype=np.float32))[None, :]
        with torch.no_grad():
            score = _UTMOS_MODEL(x, 16000)
        return float(score.item())
