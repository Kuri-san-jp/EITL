"""Ear interface — every perceptual / numerical evaluator implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
from pydantic import BaseModel, Field


class EarResult(BaseModel):
    """Output of one ear call."""
    model_config = {"arbitrary_types_allowed": True}

    name: str
    score: Union[float, Dict[str, float]]
    confidence: Optional[float] = None
    frame_scores: Optional[Any] = None      # np.ndarray when supports_temporal
    elapsed_sec: float = 0.0
    cost_usd: float = 0.0
    raw: Optional[Dict[str, Any]] = None
    warnings: list[str] = Field(default_factory=list)


class Ear(ABC):
    """Abstract ear. Subclasses must set the class attrs and implement evaluate()."""
    name: str = "unnamed"
    tier: int = 4
    cost_per_call_usd: float = 0.0
    requires_reference: bool = False
    supports_temporal: bool = False
    # Only ears that actually use the GPU override this to True.
    # MultiResolutionEvaluator reads it to decide whether to call
    # release_model() after evaluation (policy: never hold several ears
    # on the GPU at the same time).
    uses_gpu_model: bool = False

    @abstractmethod
    async def evaluate(
        self,
        audio: np.ndarray,
        sr: int,
        reference: Optional[np.ndarray] = None,
        window: Optional[Tuple[float, float]] = None,
    ) -> EarResult:
        """Score `audio` (channels, samples). `window` = (start_sec, end_sec)."""
        raise NotImplementedError

    def release_model(self) -> None:
        """Release the loaded GPU model (default: no-op).

        Ears that use the GPU should reset references such as self._model /
        self._fe / self._processor to None and call torch.cuda.empty_cache()
        to give the VRAM back. This is called from
        `MultiResolutionEvaluator._run_all` after each ear's evaluate().
        """
        pass

    # convenience for sync callers
    def evaluate_sync(self, audio, sr, **kw) -> EarResult:
        import asyncio
        return asyncio.run(self.evaluate(audio, sr, **kw))
