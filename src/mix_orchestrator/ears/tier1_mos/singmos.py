"""SingMOS wrapper (singing-specific MOS predictor)."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed
from ._proxy_mos import proxy_mos_score


_SINGMOS = None        # module-level cache


class SingMOSEar(Ear):
    name = "singmos"
    tier = 1
    cost_per_call_usd = 0.0
    uses_gpu_model = True

    def __init__(self, use_proxy_if_missing: bool = False):
        self.use_proxy = use_proxy_if_missing

    def release_model(self) -> None:
        global _SINGMOS
        _SINGMOS = None
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
            warnings = [f"SingMOS fallback proxy: {ex}"]
        return EarResult(name=self.name, score=float(score),
                         elapsed_sec=time.time()-t0, warnings=warnings)

    def _real(self, audio, sr) -> float:
        """SingMOS (singing-specific MOS predictor, Tang et al. 2024).

        The implementation goes through the hubconf of ``South-Twilight/SingMOS``.
        The repo is cloned into ``external/SingMOS`` and added to PYTHONPATH
        (``_common._setup_hf_cache_env``). The weights are downloaded from the
        GitHub releases into the torch.hub cache (``TORCH_HOME/checkpoints``).
        """
        import torch
        global _SINGMOS
        if _SINGMOS is None:
            try:
                _SINGMOS = torch.hub.load(
                    "South-Twilight/SingMOS",
                    "singmos_v1",
                    pretrained=True,
                    trust_repo=True,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _SINGMOS = _SINGMOS.to(device).eval()
            except Exception as ex:                                # noqa: BLE001
                raise ImportError(
                    "Cannot fetch SingMOS from torch.hub: "
                    f"{type(ex).__name__}: {ex}. "
                    "s3prl is required (`pip install s3prl`); otherwise "
                    "check that external/SingMOS has been git cloned "
                    "(docs/install_real_ears.md §3.4)"
                ) from ex

        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if sr != 16000:
            import librosa
            mono = librosa.resample(mono.astype(np.float32),
                                    orig_sr=sr, target_sr=16000)
        x = torch.from_numpy(np.asarray(mono, dtype=np.float32))[None, :]
        device = next(_SINGMOS.parameters()).device
        x = x.to(device)
        # MOS_Predictor.forward(audio, audio_length) signature
        audio_length = torch.tensor([x.shape[1]], dtype=torch.long).to(device)
        with torch.no_grad():
            score = _SINGMOS(x, audio_length)
        return float(score.squeeze().item())
