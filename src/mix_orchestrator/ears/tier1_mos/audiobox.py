"""Audiobox-Aesthetics 4-axis ear (PQ / PC / CE / CU).

A no-reference acoustic aesthetics model released by Meta in 2025. For one
audio input it returns mean opinion scores on 4 axes (Production Quality /
Production Complexity / Content Enjoyment / Content Usefulness).

Implementation notes:
  - Uses the `audiobox-aesthetics` (pip) package.
  - The predictor is initialised once per module and reused for every
    subsequent evaluation.
  - The input audio is written to a temporary wav and passed to the
    package's forward().
  - Supports both the old API (`pred.forward([{"path": ...}])`) and the new
    API (`pred(["..."])` etc.).
  - The proxy is kept for unit-test compatibility but is disabled by default
    (it is never used in experiments).
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import Ear, EarResult
from .._proxy_guard import assert_proxy_allowed


# Only one model predictor per process, to limit GPU memory allocation.
_PREDICTOR: Any = None


def _get_predictor() -> Any:
    """Initialise the Audiobox-Aesthetics predictor (once only)."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    # Old API
    try:
        from audiobox_aesthetics.infer import initialize_predictor      # type: ignore
        _PREDICTOR = initialize_predictor()
        return _PREDICTOR
    except ImportError:
        pass
    # New API (after the package was refactored)
    try:
        from audiobox_aesthetics.predictor import AudioBoxAesthetics    # type: ignore
        _PREDICTOR = AudioBoxAesthetics.from_pretrained(
            "facebook/audiobox-aesthetics")
        return _PREDICTOR
    except ImportError as ex:
        raise ImportError(
            "The audiobox-aesthetics package is not installed. "
            "Run `pip install audiobox-aesthetics` through slurm."
        ) from ex


def _call_predictor(predictor: Any, wav_paths: List[str]) -> List[Dict[str, float]]:
    """Absorb both the old and new APIs and return a list of 4-axis dicts."""
    # Old API: forward([{"path": ...}, ...])
    if hasattr(predictor, "forward"):
        try:
            return predictor.forward([{"path": p} for p in wav_paths])
        except TypeError:
            pass
    # New API: __call__(paths) or predict(paths)
    if callable(predictor):
        try:
            return predictor(wav_paths)
        except TypeError:
            pass
    if hasattr(predictor, "predict"):
        return predictor.predict(wav_paths)
    raise RuntimeError(
        "Cannot tell how to call the Audiobox-Aesthetics predictor "
        f"(type={type(predictor).__name__})"
    )


class AudioboxEar(Ear):
    name = "audiobox"
    tier = 1
    cost_per_call_usd = 0.0
    supports_temporal = False
    uses_gpu_model = True

    def release_model(self) -> None:
        """Release the global predictor. It is re-initialised on the next evaluate."""
        global _PREDICTOR
        _PREDICTOR = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                          # noqa: BLE001
            pass

    # Canonical keys for the 4 axes. If Audiobox emits different keys
    # (e.g. lowercase) they are normalised to the uppercase form.
    AXES = ("PQ", "PC", "CE", "CU")

    def __init__(self, weights: str = "facebook/audiobox-aesthetics",
                 use_proxy_if_missing: bool = False):
        self.weights = weights
        self.use_proxy_if_missing = use_proxy_if_missing

    async def evaluate(self,
                       audio: np.ndarray,
                       sr: int,
                       reference: Optional[np.ndarray] = None,
                       window: Optional[Tuple[float, float]] = None,
                       ) -> EarResult:
        t0 = time.time()
        if window is not None:
            s, e = window
            audio = audio[:, int(s*sr):int(e*sr)] \
                if audio.ndim == 2 else audio[int(s*sr):int(e*sr)]

        try:
            scores = self._run_real(audio, sr)
            return EarResult(name=self.name, score=scores,
                             elapsed_sec=time.time() - t0)
        except Exception as ex:                                              # noqa: BLE001
            if not self.use_proxy_if_missing:
                raise
            return self._proxy(audio, sr, t0,
                               warning=f"audiobox real failed: {ex}")

    # ------------------------------------------------------------------
    # Real evaluation
    # ------------------------------------------------------------------

    def _run_real(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        import soundfile as sf
        predictor = _get_predictor()

        # Audiobox expects a wav file, so write a temporary one.
        # 16-bit PCM, 44.1 kHz, stereo (Audiobox can resample other rates, but
        # we pass the audio through unchanged to be safe).
        arr = audio.T if audio.ndim == 2 else audio
        # Range check (avoid clipping)
        arr = np.clip(arr, -1.0, 1.0).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            sf.write(wav_path, arr, sr, subtype="FLOAT")
            result = _call_predictor(predictor, [wav_path])
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        # The output is expected to be a List[Dict] (batch of 1). Absorb other shapes.
        if isinstance(result, dict):
            d = result
        elif isinstance(result, list) and result:
            d = result[0]
        else:
            raise RuntimeError(
                f"Unknown Audiobox output format: type={type(result).__name__}"
            )

        # Absorb differences in key case / spelling
        normalised: Dict[str, float] = {}
        upper_map = {k.upper(): k for k in d.keys()}
        for axis in self.AXES:
            if axis in d:
                normalised[axis] = float(d[axis])
            elif axis in upper_map:
                normalised[axis] = float(d[upper_map[axis]])
            else:
                raise KeyError(
                    f"Audiobox output is missing the required key {axis!r}. "
                    f"Keys received: {list(d.keys())}"
                )
        return normalised

    # ------------------------------------------------------------------
    # Surrogate computation for unit-test compatibility (never used in real experiments)
    # ------------------------------------------------------------------

    def _proxy(self, audio: np.ndarray, sr: int, t0: float,
               warning: str) -> EarResult:
        """For dev machines where the package is not installed, under unit tests.
        NOTE: build_ears for experiments sets use_proxy_if_missing=False, so this
        is never reached there.
        NOTE: with AUTOMIX_FORBID_PROXY=1 this hard-stops with an exception.
        """
        assert_proxy_allowed(self.name)
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
        from numpy.fft import rfft, rfftfreq
        spec = np.abs(rfft(mono[: min(len(mono), sr * 5)]))
        freqs = rfftfreq(min(len(mono), sr * 5), 1 / sr)
        centroid = float((spec * freqs).sum() / (spec.sum() + 1e-12))
        crest = float(np.max(np.abs(mono)) / (rms + 1e-9))

        def squash(x: float, lo: float, hi: float) -> float:
            return 5.0 + 2.0 * (np.tanh((x - lo) / (hi - lo)) * 0.5 + 0.5)

        return EarResult(
            name=self.name,
            score={
                "PQ": squash(rms, 0.05, 0.3),
                "PC": squash(centroid, 1500, 4000),
                "CE": squash(1.0 / (crest + 1e-3), 0.1, 0.4),
                "CU": squash(rms * centroid / 1000, 0.1, 1.0),
            },
            elapsed_sec=time.time() - t0,
            warnings=[warning],
        )
