"""Re-separation SDR ear.

Run Demucs on the mix, compare the recovered stems against the original
stems. Higher SDR ⇒ mix preserves separability ⇒ mix didn't "smear"
sources together. Used as a faithfulness-to-stems proxy.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


class ReseparationSDREar(Ear):
    name = "reseparation_sdr"
    tier = 5
    cost_per_call_usd = 0.0
    requires_reference = True        # original stems = reference
    uses_gpu_model = True

    def __init__(self, demucs_model: str = "htdemucs",
                 device: str = "cuda",
                 use_proxy_if_missing: bool = False):
        self.demucs_model = demucs_model
        self.device = device
        self.use_proxy = use_proxy_if_missing
        self._model = None
        # `reference` is set out-of-band as a dict[name -> stem] via attach_stems
        self.original_stems: Optional[Dict[str, np.ndarray]] = None

    def release_model(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                          # noqa: BLE001
            pass

    def attach_stems(self, stems: Dict[str, np.ndarray]) -> None:
        """Set the original stems to compare against (orchestrator wires this)."""
        self.original_stems = stems

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
            return self._proxy(audio, sr, t0, warning=f"demucs failed: {ex}")

    def _real(self, audio, sr, t0) -> EarResult:
        if self.original_stems is None:
            raise RuntimeError("call attach_stems() before evaluating")
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        import torch
        if self._model is None:
            self._model = get_model(self.demucs_model).to(self.device)
            self._model.eval()
        # Demucs expects (batch, channels, samples) at 44.1k
        x = torch.from_numpy(audio.astype(np.float32))[None, ...].to(self.device)
        with torch.no_grad():
            sources = apply_model(self._model, x, split=True, overlap=0.1)
        sources = sources[0].cpu().numpy()  # (n_sources, C, N)
        source_names = self._model.sources  # ["drums", "bass", "other", "vocals"]

        # Compute SDR per matched original stem (best-name match)
        sdr_per = {}
        for est, name in zip(sources, source_names):
            ref = self._match_original(name)
            if ref is None:
                continue
            sdr_per[name] = _sdr(ref, est)
        if not sdr_per:
            return EarResult(name=self.name, score={"sdr_mean": 0.0},
                             elapsed_sec=time.time()-t0,
                             warnings=["no matching original stems"])
        return EarResult(
            name=self.name,
            score={"sdr_mean": float(np.mean(list(sdr_per.values()))),
                   **{f"sdr_{k}": float(v) for k, v in sdr_per.items()}},
            elapsed_sec=time.time()-t0,
        )

    def _match_original(self, demucs_name: str) -> Optional[np.ndarray]:
        if self.original_stems is None:
            return None
        for orig_name, stem in self.original_stems.items():
            o = orig_name.lower()
            if demucs_name == "vocals" and ("vocal" in o or "voc" in o):
                return _ensure_stereo(stem)
            if demucs_name == "drums" and any(k in o for k in ("drum", "kick", "snare")):
                return _ensure_stereo(stem)
            if demucs_name == "bass" and "bass" in o:
                return _ensure_stereo(stem)
        return None

    def _proxy(self, audio, sr, t0, warning) -> EarResult:
        assert_proxy_allowed(self.name)
        # Fallback computation: correlation between the mix and the sum of the original stems
        if self.original_stems is None:
            return EarResult(name=self.name, score={"sdr_mean": 0.0},
                             elapsed_sec=time.time()-t0,
                             warnings=[warning, "no stems"])
        ref_sum = None
        for s in self.original_stems.values():
            s2 = _ensure_stereo(s)
            if ref_sum is None:
                ref_sum = s2.copy()
            else:
                n = min(ref_sum.shape[-1], s2.shape[-1])
                ref_sum = ref_sum[:, :n] + s2[:, :n]
        sdr = _sdr(ref_sum, audio[:, :ref_sum.shape[-1]])
        return EarResult(name=self.name, score={"sdr_mean": float(sdr)},
                         elapsed_sec=time.time()-t0, warnings=[warning])


def _sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-9) -> float:
    """SDR in dB between two (C, N) arrays."""
    n = min(ref.shape[-1], est.shape[-1])
    ref = ref[..., :n]
    est = est[..., :n]
    num = float(np.sum(ref ** 2))
    den = float(np.sum((ref - est) ** 2))
    return 10.0 * np.log10((num + eps) / (den + eps))


def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x])
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    return x[:2]
