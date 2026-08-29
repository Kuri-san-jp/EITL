"""SCOREQ wrapper."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed
from ._proxy_mos import proxy_mos_score


_SCOREQ = None     # module-level cache


class SCOREQEar(Ear):
    name = "scoreq"
    tier = 1
    cost_per_call_usd = 0.0
    uses_gpu_model = True

    def __init__(self, use_proxy_if_missing: bool = False):
        self.use_proxy = use_proxy_if_missing

    def release_model(self) -> None:
        global _SCOREQ
        _SCOREQ = None
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
            # SCOREQ is on a [0,1] range, so normalize proxy_mos's [1..5]
            score = (proxy_mos_score(audio, sr) - 1.0) / 4.0
            warnings = [f"SCOREQ fallback proxy: {ex}"]
        return EarResult(name=self.name, score=float(score),
                         elapsed_sec=time.time()-t0, warnings=warnings)

    def _real(self, audio, sr) -> float:
        """`pip install scoreq` (Mittag 2024).

        ``Scoreq.predict(test_path)`` expects a wav file path, so write the audio
        to a temporary wav first and pass that.
        """
        try:
            import scoreq
        except ImportError as ex:
            raise ImportError(
                "cannot import scoreq. Run `pip install scoreq` through slurm "
                "(docs/install_real_ears.md §3.4)"
            ) from ex
        global _SCOREQ
        if _SCOREQ is None:
            # Initialize with the default data_domain='natural'.
            # mode='nr' = no-reference. On the first call the weights are
            # downloaded from Zenodo into ~/.cache/scoreq (HOME=/mnt/hf_cache
            # points that inside the project).
            _SCOREQ = scoreq.Scoreq(data_domain="natural", mode="nr")
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if sr != 16000:
            import librosa
            mono = librosa.resample(mono.astype(np.float32),
                                    orig_sr=sr, target_sr=16000)
        import tempfile, os, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            sf.write(wav_path, mono.astype(np.float32), 16000, subtype="FLOAT")
            result = _SCOREQ.predict(test_path=wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        # predict may return a dict, a tuple, or a float
        if isinstance(result, dict):
            return float(result.get("score", result.get("mos", 0.0)))
        if isinstance(result, (tuple, list)) and result:
            return float(result[0])
        return float(result)
