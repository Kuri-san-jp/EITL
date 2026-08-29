"""CDPAM perceptual audio distance ear."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class CDPAMEar(Ear):
    name = "cdpam"
    tier = 2
    cost_per_call_usd = 0.0
    requires_reference = True
    uses_gpu_model = True

    def __init__(self, device: str = "cuda",
                 use_proxy_if_missing: bool = False):
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
            return EarResult(name=self.name, score=0.0, elapsed_sec=time.time()-t0,
                             warnings=["no reference"])
        try:
            return EarResult(name=self.name, score={"cdpam_distance": float(self._real(audio, reference, sr))},
                             elapsed_sec=time.time()-t0)
        except Exception as ex:                                            # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, reference, t0, warning=f"cdpam failed: {ex}")

    def _real(self, audio, ref, sr) -> float:
        try:
            import cdpam
        except ImportError as ex:
            raise ImportError("`pip install cdpam`") from ex
        import torch
        if self._model is None:
            # The weights bundled with CDPAM use an old pickle format, which
            # raises UnpicklingError under torch 2.6's default weights_only=True.
            # Monkey-patch torch.load for the duration of initialisation only,
            # so cdpam.CDPAM's internal torch.load call gets weights_only=False.
            original_torch_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            torch.load = _patched_torch_load
            try:
                self._model = cdpam.CDPAM(dev=self.device)
            finally:
                torch.load = original_torch_load
        a = audio.mean(axis=0) if audio.ndim == 2 else audio
        r = ref.mean(axis=0) if ref.ndim == 2 else ref
        if sr != 22050:
            import librosa
            a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=22050)
            r = librosa.resample(r.astype(np.float32), orig_sr=sr, target_sr=22050)
        # CDPAM.forward calls audio.unsqueeze(1) internally, so the input has
        # to be a 2D (B, N) tensor (B=1 for a single call).
        # The pretrained weights also assume 16-bit integers, so scale to int16.
        a_t = (torch.from_numpy(a).float() * 32768.0).unsqueeze(0).to(self.device)
        r_t = (torch.from_numpy(r).float() * 32768.0).unsqueeze(0).to(self.device)
        return float(self._model.forward(a_t, r_t).item())

    def _proxy(self, audio, ref, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        n = min(audio.shape[-1], ref.shape[-1])
        a = audio[..., :n].mean(axis=0) if audio.ndim == 2 else audio[:n]
        r = ref[..., :n].mean(axis=0) if ref.ndim == 2 else ref[:n]
        rms_diff = float(np.sqrt(np.mean((a - r) ** 2) + 1e-12))
        return EarResult(name=self.name, score={"cdpam_distance": rms_diff},
                         elapsed_sec=time.time()-t0, warnings=[warning])
