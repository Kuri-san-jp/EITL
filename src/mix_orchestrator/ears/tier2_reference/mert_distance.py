"""MERT embedding distance ear (music-aware embedding from m-a-p/MERT)."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class MERTDistanceEar(Ear):
    name = "mert"
    tier = 2
    cost_per_call_usd = 0.0
    requires_reference = True
    uses_gpu_model = True

    def __init__(self, device: str = "cuda",
                 hf_model_id: str = "m-a-p/MERT-v1-330M",
                 use_proxy_if_missing: bool = False):
        self.device = device
        self.hf_model_id = hf_model_id
        self.use_proxy = use_proxy_if_missing
        self._model = None
        self._fe = None

    def release_model(self) -> None:
        self._model = None
        self._fe = None
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
            return EarResult(name=self.name, score={"mert_distance": float(d)},
                             elapsed_sec=time.time()-t0)
        except Exception as ex:                                            # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, reference, t0, warning=f"mert failed: {ex}")

    def _real(self, audio, ref, sr) -> float:
        try:
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except ImportError as ex:
            raise ImportError(
                "Cannot import transformers. "
                "Run `pip install 'transformers>=4.46,<4.50'` via slurm "
                "(docs/install_real_ears.md §2)"
            ) from ex
        import torch
        if self._model is None:
            self._fe = Wav2Vec2FeatureExtractor.from_pretrained(
                self.hf_model_id, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.hf_model_id, trust_remote_code=True).to(self.device)
            self._model.eval()
        target_sr = self._fe.sampling_rate
        a = audio.mean(axis=0) if audio.ndim == 2 else audio
        r = ref.mean(axis=0) if ref.ndim == 2 else ref
        if sr != target_sr:
            import librosa
            a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            r = librosa.resample(r.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        def embed(x):
            inputs = self._fe(x, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs, output_hidden_states=True)
            # average hidden states across layers and time
            h = torch.stack(out.hidden_states).mean(0).mean(1)
            return h
        e_a = embed(a); e_r = embed(r)
        # Cosine distance
        cos = float((e_a @ e_r.T).item() /
                    ((e_a.norm() * e_r.norm()).item() + 1e-9))
        return 1.0 - cos        # distance, smaller is better

    def _proxy(self, audio, ref, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        # Fallback computation: (1 - corr) of the mel-spectrogram correlation
        from scipy.signal import spectrogram
        def melstats(x):
            x = x.mean(axis=0) if x.ndim == 2 else x
            f, _, sxx = spectrogram(x, fs=44100, nperseg=2048)
            return np.log(sxx.mean(axis=1) + 1e-9)
        a = melstats(audio); r = melstats(ref)
        n = min(len(a), len(r))
        corr = float(np.corrcoef(a[:n], r[:n])[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        return EarResult(name=self.name, score={"mert_distance": 1.0 - corr},
                         elapsed_sec=time.time()-t0, warnings=[warning])
