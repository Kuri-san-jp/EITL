"""CLAP (LAION) embedding cosine similarity ear.

Useful as a style/genre-faithfulness proxy: similarity between the mix's
CLAP embedding and a reference's embedding (e.g., the original rough
mix, or a target style audio).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class CLAPSimilarityEar(Ear):
    name = "clap"
    tier = 2
    cost_per_call_usd = 0.0
    requires_reference = True
    uses_gpu_model = True

    def __init__(self, device: str = "cuda",
                 hf_model_id: str = "laion/clap-htsat-unfused",
                 use_proxy_if_missing: bool = False):
        self.device = device
        self.hf_model_id = hf_model_id
        self.use_proxy = use_proxy_if_missing
        self._model = None
        self._processor = None

    def release_model(self) -> None:
        self._model = None
        self._processor = None
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
            sim = self._real(audio, reference, sr)
            return EarResult(name=self.name, score={"clap_cosine": float(sim)},
                             elapsed_sec=time.time()-t0)
        except Exception as ex:                                            # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, reference, t0, warning=f"clap failed: {ex}")

    def _real(self, audio, ref, sr) -> float:
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as ex:
            raise ImportError(
                "transformers / ClapModel cannot be imported. "
                "Run `pip install 'transformers>=4.46,<4.50'` via slurm "
                "(docs/install_real_ears.md §2)"
            ) from ex
        import torch
        if self._model is None:
            self._processor = ClapProcessor.from_pretrained(self.hf_model_id)
            self._model = ClapModel.from_pretrained(
                self.hf_model_id).to(self.device)
            self._model.eval()
        # CLAP expects 48 kHz mono
        a = audio.mean(axis=0) if audio.ndim == 2 else audio
        r = ref.mean(axis=0) if ref.ndim == 2 else ref
        if sr != 48000:
            import librosa
            a = librosa.resample(a.astype(np.float32),
                                 orig_sr=sr, target_sr=48000)
            r = librosa.resample(r.astype(np.float32),
                                 orig_sr=sr, target_sr=48000)
        inputs = self._processor(audios=[a, r], sampling_rate=48000,
                                 return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self._model.get_audio_features(**inputs)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
        return float((emb[0] @ emb[1]).item())

    def _proxy(self, audio, ref, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        # Proxy: spectral-centroid + crest similarity
        from numpy.fft import rfft, rfftfreq
        def feats(x, sr):
            x = x.mean(axis=0) if x.ndim == 2 else x
            x = x[:min(len(x), sr * 5)]
            spec = np.abs(rfft(x)) + 1e-12
            freqs = rfftfreq(len(x), 1/sr)
            return float((spec * freqs).sum() / spec.sum())
        f_a = feats(audio, 44100); f_r = feats(ref, 44100)
        sim = float(1 - min(abs(f_a - f_r) / 8000, 1.0))
        return EarResult(name=self.name, score={"clap_cosine": sim},
                         elapsed_sec=time.time()-t0, warnings=[warning])
