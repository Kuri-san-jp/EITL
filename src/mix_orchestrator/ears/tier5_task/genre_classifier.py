"""Genre classifier as a task-grounded ear.

Reports confidence in the dominant genre. The agent can use this to
detect when a mix shifts perceived genre (typically a bad sign for
faithfulness).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class GenreClassifierEar(Ear):
    name = "genre"
    tier = 5
    cost_per_call_usd = 0.0
    uses_gpu_model = True

    def __init__(self,
                 hf_model_id: str = "mtg-upf/discogs-maest-30s-pw-129e",
                 device: str = "cuda",
                 use_proxy_if_missing: bool = False):
        self.hf_model_id = hf_model_id
        self.device = device
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
        if window is not None:
            s, e = window
            audio = audio[:, int(s*sr):int(e*sr)] if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]
        try:
            return self._real(audio, sr, t0)
        except Exception as ex:                                            # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, sr, t0, warning=f"genre model failed: {ex}")

    def _real(self, audio, sr, t0) -> EarResult:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        import torch
        if self._model is None:
            self._fe = AutoFeatureExtractor.from_pretrained(self.hf_model_id, trust_remote_code=True)
            self._model = AutoModelForAudioClassification.from_pretrained(
                self.hf_model_id, trust_remote_code=True).to(self.device)
            self._model.eval()
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        # Most genre models expect 16k
        if sr != 16000:
            import librosa
            mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=16000)
        inputs = self._fe(mono, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        # label_map may be in the model config
        id2label = getattr(self._model.config, "id2label", {})
        top_label = id2label.get(top_idx, str(top_idx))
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        return EarResult(
            name=self.name,
            score={"genre_confidence": top_conf, "entropy": entropy},
            raw={"top_label": top_label, "all_probs_top5": probs.argsort()[-5:][::-1].tolist()},
            elapsed_sec=time.time()-t0,
        )

    def _proxy(self, audio, sr, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        # Substitute computation: confidence ~= 1 - spectral flatness
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        from numpy.fft import rfft
        spec = np.abs(rfft(mono[:min(len(mono), sr * 5)])) + 1e-12
        geo = np.exp(np.mean(np.log(spec)))
        arith = np.mean(spec)
        sfm = float(geo / arith)         # 0 = peaky (tonal), 1 = noisy
        conf = float(np.clip(1.0 - sfm, 0.1, 0.95))
        return EarResult(name=self.name,
                         score={"genre_confidence": conf, "entropy": -float(np.log(conf+1e-9))},
                         raw={"top_label": "unknown"},
                         elapsed_sec=time.time()-t0,
                         warnings=[warning])
