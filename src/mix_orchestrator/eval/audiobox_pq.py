"""Thin scorer that returns **only the PQ (Production Quality) axis** of Audiobox-Aesthetics.

Wraps the existing `AudioboxEar` (which returns the 4 axes PQ/PC/CE/CU) and
pulls out just the PQ float needed by the min agreement reward. AudioboxEar's
predictor is a single per-module instance, so scoring repeatedly here does not
trigger a reload.

Strict policy (memory: feedback_no_proxy_no_experiment): use_proxy_if_missing=False.
If audiobox-aesthetics cannot be imported / loaded, stop with an exception.
"""
from __future__ import annotations

import asyncio
from typing import Optional, Tuple

import numpy as np

from ..ears.tier1_mos.audiobox import AudioboxEar


class AudioboxPQScorer:
    """Returns Audiobox-PQ via `score(audio, sr) -> float`."""

    def __init__(self, ear: Optional[AudioboxEar] = None,
                 weights: str = "facebook/audiobox-aesthetics"):
        self.ear = ear if ear is not None else AudioboxEar(
            weights=weights, use_proxy_if_missing=False)

    def score(self, audio: np.ndarray, sr: int,
              window: Optional[Tuple[float, float]] = None) -> float:
        """Return the Production Quality (PQ) axis."""
        result = asyncio.run(self.ear.evaluate(audio, sr, window=window))
        score = result.score
        if isinstance(score, dict):
            if "PQ" not in score:
                raise KeyError(
                    f"Audiobox output has no PQ: keys={list(score.keys())}")
            return float(score["PQ"])
        raise RuntimeError(
            f"AudioboxEar did not return a dict: type={type(score).__name__}")

    def release_model(self) -> None:
        self.ear.release_model()
