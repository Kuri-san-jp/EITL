"""Common base class for plugging external SOTA methods into the E3 head-to-head.

Our orchestrator is closed-loop iterative, whereas external methods are
basically one-shot (stems -> mix), so only the evaluation phase is shared.

Evaluation runs through ``MultiResolutionEvaluator`` + the 18-ear ensemble,
lining up the 7 systems under the same reward / FAD / MOS.

Implementation guide (per subclass):
  - load the weight files in ``__init__`` (in-project ``external/<name>/``)
  - do one-shot inference in ``mix(stems, sr, reference=None)``
  - the returned audio is ``np.float32`` of shape (C, N), with sr matching the input
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ExternalBaseline(ABC):
    """Base class for external methods that return stems -> mix in one shot."""

    name: str = "unnamed"
    paper: str = ""
    requires_reference: bool = False

    @abstractmethod
    def mix(self,
            stems: Dict[str, np.ndarray],
            sr: int,
            reference: Optional[np.ndarray] = None,
            ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run one-shot inference.

        Args:
            stems: MUSDB18-HQ format, i.e. ``{"drums": (C, N), "bass": (C, N),
                     "vocals": (C, N), "other": (C, N)}``.
                   Some subclasses accept an arbitrary number of stems (MEGAMI etc.).
            sr: input sample rate (44100 assumed). A subclass resamples
                internally if it needs to.
            reference: optional reference wav (used by MEGAMI and similar).

        Returns:
            (mix_audio, meta)
              - ``mix_audio``: ``np.float32`` of shape ``(C=2, N)``, sr matching the input
              - ``meta``: ``{"elapsed_sec": float, "model_name": str, ...}``
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
