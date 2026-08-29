"""Qwen2-Audio-7B-Instruct as a Tier 3 audio-LLM judge.

Two usage modes:
  • Single-audio:  qualitative scoring on a 1-10 scale
  • A/B compare:   "which mix is better and why" — preference + reason

GPU required for the real model (~14 GB VRAM at fp16). When not loadable,
falls back to a deterministic proxy so the eval pipeline still runs.
"""
from __future__ import annotations

import io
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed

try:
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    HAS_QWEN2_AUDIO = True
except Exception:                                                  # noqa: BLE001
    HAS_QWEN2_AUDIO = False


_MODEL = None
_PROCESSOR = None
_DEVICE: str = "cpu"


def _lazy_load(model_id: str, device: str):
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return
    _PROCESSOR = AutoProcessor.from_pretrained(model_id)
    _MODEL = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    _MODEL.eval()
    _DEVICE = device


class Qwen2AudioJudge(Ear):
    name = "qwen2_audio_judge"
    tier = 3
    cost_per_call_usd = 0.0
    requires_reference = False           # works as single-audio AND A/B
    uses_gpu_model = True

    def release_model(self) -> None:
        global _MODEL, _PROCESSOR
        _MODEL = None
        _PROCESSOR = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                          # noqa: BLE001
            pass

    def __init__(self, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
                 device: str = "cuda",
                 max_new_tokens: int = 200,
                 use_proxy_if_missing: bool = False):
        self.model_id = model_id
        self.device = device if (HAS_QWEN2_AUDIO and _cuda_ok(device)) else "cpu"
        self.max_new_tokens = max_new_tokens
        self.use_proxy = use_proxy_if_missing

    async def evaluate(self, audio: np.ndarray, sr: int,
                       reference: Optional[np.ndarray] = None,
                       window: Optional[Tuple[float, float]] = None) -> EarResult:
        t0 = time.time()
        if not HAS_QWEN2_AUDIO:
            if not self.use_proxy:
                raise RuntimeError("transformers/Qwen2-Audio unavailable")
            return self._proxy(audio, reference, t0, warning="transformers not installed")
        try:
            _lazy_load(self.model_id, self.device)
            if reference is None:
                out = self._call_single(audio, sr)
            else:
                out = self._call_compare(audio, reference, sr)
            return EarResult(name=self.name, score=out["score"],
                             confidence=out.get("confidence"),
                             raw=out, elapsed_sec=time.time()-t0)
        except Exception as ex:                                    # noqa: BLE001
            if not self.use_proxy:
                raise
            return self._proxy(audio, reference, t0, warning=f"Qwen2-Audio failed: {ex}")

    # ----------------- prompts -----------------

    _SINGLE_PROMPT = (
        "You are an expert audio engineer. Listen to this music mix and rate it on a scale of 1 to 10 "
        "based on overall production quality. Briefly explain key strengths and weaknesses. "
        "End with: SCORE: <number>"
    )
    _COMPARE_PROMPT = (
        "You are an expert audio engineer. Compare these two mixes (A and B) and decide which is better, "
        "focusing on balance, clarity, dynamics, and stereo image. "
        "End with one of: PREFERS: A | PREFERS: B | PREFERS: TIE"
    )

    def _call_single(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        # Qwen2-Audio expects 16 kHz
        a16 = _resample(mono.astype(np.float32), sr, 16000)
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "<inline>"},
            {"type": "text", "text": self._SINGLE_PROMPT},
        ]}]
        text = _PROCESSOR.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = _PROCESSOR(text=text, audios=[a16], sampling_rate=16000,
                            return_tensors="pt", padding=True)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            gen = _MODEL.generate(**inputs, max_new_tokens=self.max_new_tokens)
        out_text = _PROCESSOR.batch_decode(gen, skip_special_tokens=True)[0]
        score = _parse_score(out_text)
        return {"score": float(score), "confidence": 0.7, "text": out_text}

    def _call_compare(self, a: np.ndarray, b: np.ndarray, sr: int) -> Dict[str, Any]:
        a_m = a.mean(axis=0) if a.ndim == 2 else a
        b_m = b.mean(axis=0) if b.ndim == 2 else b
        a16 = _resample(a_m.astype(np.float32), sr, 16000)
        b16 = _resample(b_m.astype(np.float32), sr, 16000)
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "<A>"},
            {"type": "audio", "audio_url": "<B>"},
            {"type": "text", "text": self._COMPARE_PROMPT},
        ]}]
        text = _PROCESSOR.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = _PROCESSOR(text=text, audios=[a16, b16], sampling_rate=16000,
                            return_tensors="pt", padding=True)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            gen = _MODEL.generate(**inputs, max_new_tokens=self.max_new_tokens)
        out_text = _PROCESSOR.batch_decode(gen, skip_special_tokens=True)[0]
        pref = _parse_preference(out_text)
        # Encode as numeric: A=1.0, B=-1.0, tie=0.0
        score = {"prefers_A": 1.0, "prefers_B": -1.0, "tie": 0.0}.get(pref, 0.0)
        return {"score": float(score), "confidence": 0.8, "preference": pref, "text": out_text}

    def _proxy(self, audio, reference, t0, warning: str) -> EarResult:
        assert_proxy_allowed(self.name)
        from ..tier1_mos._proxy_mos import proxy_mos_score
        a_score = proxy_mos_score(audio, 44100)
        if reference is None:
            return EarResult(name=self.name, score=float(a_score),
                             confidence=0.3, elapsed_sec=time.time()-t0,
                             warnings=[warning])
        b_score = proxy_mos_score(reference, 44100)
        # Positive ⇒ prefers A
        sgn = float(np.sign(a_score - b_score))
        return EarResult(name=self.name,
                         score={"score": sgn, "delta": a_score - b_score},
                         confidence=0.3, elapsed_sec=time.time()-t0,
                         warnings=[warning])


def _cuda_ok(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                                              # noqa: BLE001
        return False


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        import librosa
        return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out)
    except Exception:                                              # noqa: BLE001
        # naive decimation fallback
        ratio = sr_out / sr_in
        n = int(len(x) * ratio)
        idx = (np.arange(n) / ratio).astype(int)
        idx = np.clip(idx, 0, len(x) - 1)
        return x[idx]


def _parse_score(text: str) -> float:
    """Extract the mix score from the Qwen2-Audio output (robust to several phrasings).

    Besides "SCORE: 7" the model sometimes answers "I would rate it 7/10",
    "a 7 out of 10", "score of 7" and so on. We try the patterns in order and
    fall back to NaN, which tells the caller "could not parse" (and keeps it
    distinguishable from the artifact of a default 5.0).
    """
    import re
    pats = [
        r"SCORE\s*:?\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s*out of\s*10",
        r"rate (?:it|this)?\s*(?:a|an)?\s*(\d+(?:\.\d+)?)",
        r"score of\s*(\d+(?:\.\d+)?)",
    ]
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 0.0 <= v <= 10.0:
                return v
    return float("nan")     # could not parse (the caller detects it)


def _parse_preference(text: str) -> str:
    import re
    m = re.search(r"PREFERS\s*:\s*(A|B|TIE)", text, re.IGNORECASE)
    return f"prefers_{m.group(1).upper()}".lower().replace("prefers_tie", "tie") if m else "tie"
