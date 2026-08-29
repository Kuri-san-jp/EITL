"""Vocal intelligibility via Whisper WER (or CER).

If a reference transcript is provided, compare Whisper's transcription of
the mix's vocal stem against it (WER). When no reference, we use
average no-speech-probability inversely as a proxy for intelligibility.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class VocalIntelligibilityEar(Ear):
    name = "vocal_intelligibility"
    tier = 5
    cost_per_call_usd = 0.0      # local Whisper
    uses_gpu_model = True

    def __init__(self,
                 model_size: str = "small",
                 reference_text: Optional[str] = None,
                 device: str = "cuda",
                 use_proxy_if_missing: bool = False):
        self.model_size = model_size
        self.reference_text = reference_text
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
        if window is not None:
            s, e = window
            audio = audio[:, int(s*sr):int(e*sr)] if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]
        try:
            return self._real(audio, sr, t0)
        except Exception as ex:                                            # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, sr, t0, warning=f"whisper failed: {ex}")

    def _real(self, audio, sr, t0) -> EarResult:
        try:
            import whisper                  # openai-whisper
        except ImportError:
            try:
                from faster_whisper import WhisperModel as _FW
            except ImportError as ex:
                raise ImportError("`pip install openai-whisper` (or faster-whisper)") from ex
            return self._real_faster(audio, sr, t0, _FW)
        # openai-whisper path
        if self._model is None:
            import os as _os
            # If WHISPER_CACHE_DIR is set, pass it explicitly as download_root
            # (it also works via XDG_CACHE_HOME, but this pins down the load path)
            kw = {"device": self.device}
            wcd = _os.environ.get("WHISPER_CACHE_DIR")
            if wcd:
                kw["download_root"] = wcd
            self._model = whisper.load_model(self.model_size, **kw)
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if sr != 16000:
            import librosa
            mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=16000)
        result = self._model.transcribe(mono.astype(np.float32), fp16=False)
        text = result["text"].strip()
        no_speech = float(np.mean([s.get("no_speech_prob", 0.0)
                                   for s in result.get("segments", [])])
                          if result.get("segments") else 0.5)
        score: dict = {"intelligibility": 1.0 - no_speech,
                       "transcript_chars": len(text)}
        if self.reference_text:
            wer = _compute_wer(self.reference_text, text)
            score["wer"] = wer
            score["1_minus_wer"] = 1.0 - wer
        return EarResult(name=self.name, score=score,
                         raw={"transcript": text}, elapsed_sec=time.time()-t0)

    def _real_faster(self, audio, sr, t0, FW) -> EarResult:
        if self._model is None:
            self._model = FW(self.model_size, device="cuda" if self.device != "cpu" else "cpu",
                             compute_type="float16" if self.device != "cpu" else "int8")
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if sr != 16000:
            import librosa
            mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=16000)
        segments, info = self._model.transcribe(mono.astype(np.float32))
        text = " ".join(seg.text for seg in segments).strip()
        score = {"intelligibility": 1.0 - (info.language_probability if info else 0.5),
                 "transcript_chars": len(text)}
        if self.reference_text:
            wer = _compute_wer(self.reference_text, text)
            score["wer"] = wer
            score["1_minus_wer"] = 1.0 - wer
        return EarResult(name=self.name, score=score,
                         raw={"transcript": text}, elapsed_sec=time.time()-t0)

    def _proxy(self, audio, sr, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        # Proxy: vocal-band energy ratio (~300Hz - 3kHz vs total)
        from numpy.fft import rfft, rfftfreq
        spec = np.abs(rfft(mono[:min(len(mono), sr * 5)])) ** 2
        freqs = rfftfreq(min(len(mono), sr * 5), 1/sr)
        vocal_band = (freqs >= 300) & (freqs < 3000)
        ratio = float(spec[vocal_band].sum() / (spec.sum() + 1e-12))
        return EarResult(name=self.name,
                         score={"intelligibility": float(np.clip(ratio * 3, 0, 1))},
                         elapsed_sec=time.time()-t0, warnings=[warning])


def _compute_wer(ref: str, hyp: str) -> float:
    """Levenshtein word error rate."""
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    # DP table
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.int32)
    for i in range(len(r)+1):
        d[i, 0] = i
    for j in range(len(h)+1):
        d[0, j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)
    return float(d[len(r), len(h)] / len(r))
