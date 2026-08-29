"""The multi-temporal-resolution evaluator.

Returns a structured dict (see Section 4.2 of spec) that the LLM
introspects directly. All ears at all resolutions are scheduled in
parallel via asyncio.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ears.base import Ear, EarResult
from ..ears.aggregation import aggregate, flatten
from .windowing import section_windows, short_term_windows


@dataclass
class MultiResolutionScores:
    """Output of MultiResolutionEvaluator.evaluate()."""
    global_scores: Dict[str, float]                              = field(default_factory=dict)
    section_scores: Dict[str, Dict[str, float]]                  = field(default_factory=dict)
    short_term_scores: Dict[str, List[float]]                    = field(default_factory=dict)
    momentary_scores: Dict[str, float]                           = field(default_factory=dict)
    meta: Dict[str, Any]                                         = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global":     self.global_scores,
            "section":    self.section_scores,
            "short_term": self.short_term_scores,
            "momentary":  self.momentary_scores,
            "meta":       self.meta,
        }

    def aggregate_reward(self) -> float:
        return aggregate(self.global_scores)


class MultiResolutionEvaluator:
    def __init__(self, ears: List[Ear],
                 short_term_win_sec: float = 3.0,
                 short_term_hop_sec: float = 1.0):
        self.ears = ears
        self.short_term_win = short_term_win_sec
        self.short_term_hop = short_term_hop_sec

    async def evaluate(self,
                       audio: np.ndarray,
                       sr: int,
                       section_boundaries: Optional[List[Tuple[float, str]]] = None,
                       resolutions: Optional[List[str]] = None,
                       reference: Optional[np.ndarray] = None,
                       ) -> MultiResolutionScores:
        t0 = time.time()
        resolutions = resolutions or ["global", "section", "short_term", "momentary"]
        duration = audio.shape[-1] / sr

        scores = MultiResolutionScores()

        # ---- Global ----
        if "global" in resolutions:
            global_results = await self._run_all(audio, sr, reference, window=None)
            scores.global_scores = flatten(global_results)

        # ---- Sections ----
        if "section" in resolutions and section_boundaries:
            sec_wins = section_windows(section_boundaries, duration)
            # Run all sections in parallel
            sec_tasks = [self._run_all(audio, sr, reference, w) for (w, _) in sec_wins]
            sec_results = await asyncio.gather(*sec_tasks)
            for (_, label), res in zip(sec_wins, sec_results):
                scores.section_scores[label] = flatten(res)

        # ---- Short-term (3s/1s windows) — cheap detectors only ----
        if "short_term" in resolutions:
            wins = short_term_windows(duration, self.short_term_win, self.short_term_hop)
            cheap_ears = [e for e in self.ears if e.tier >= 4 and e.cost_per_call_usd == 0]
            if wins and cheap_ears:
                lufs_traj = []
                for w in wins:
                    res = await asyncio.gather(*[e.evaluate(audio, sr, window=w) for e in cheap_ears])
                    for r in res:
                        if r.name == "lufs" and isinstance(r.score, dict):
                            lufs_traj.append(r.score.get("integrated_lufs", float("nan")))
                scores.short_term_scores["lufs_trajectory"] = lufs_traj
                if lufs_traj:
                    arr = np.array([x for x in lufs_traj if np.isfinite(x)])
                    if len(arr) > 0:
                        scores.short_term_scores["lufs_std"] = [float(arr.std())]

        # ---- Momentary (true peak + onset density) ----
        if "momentary" in resolutions:
            # Already captured in global if true_peak is in ears; re-expose for clarity
            tp = next((e for e in self.ears if e.name == "true_peak"), None)
            if tp is not None:
                r = await tp.evaluate(audio, sr)
                if isinstance(r.score, dict):
                    scores.momentary_scores.update(r.score)

        scores.meta = {
            "elapsed_sec": time.time() - t0,
            "duration_sec": duration,
            "n_ears": len(self.ears),
            "resolutions": resolutions,
        }
        return scores

    async def _run_all(self, audio, sr, reference, window) -> Dict[str, EarResult]:
        """Evaluate the ears one at a time, sequentially.

        Why sequential awaits instead of parallel (asyncio.gather):
          - Loading the GPU-model ears (CLAP / MERT / UTMOS / DNSMOS / Demucs /
            MAEST / Whisper / Qwen2-Audio, etc.) onto the GPU at the same time
            runs out of VRAM, or consumes 2 or more GPUs.
          - Once an ear finishes evaluating we call `release_model()` to give
            the VRAM back, ready for the next ear's initialisation (only for
            ears with uses_gpu_model=True).
          - The tier 4 detectors (LUFS/TP/Stereo/LRA) are CPU computations, so
            running them sequentially costs negligible time.
        """
        ears = [e for e in self.ears
                if not e.requires_reference or reference is not None]
        out: Dict[str, EarResult] = {}
        for ear in ears:
            try:
                r = await ear.evaluate(audio, sr, reference=reference,
                                       window=window)
            except Exception as ex:                                # noqa: BLE001
                # Log + skip; do not fail the whole eval
                out[ear.name] = EarResult(
                    name=ear.name, score=float("nan"),
                    warnings=[f"ear failed: {type(ex).__name__}: {ex!r}"])
            else:
                out[ear.name] = r
                # Anti-slacking: record in ear_real_status.json that the real
                # ear succeeded. Ears that fell back to a proxy carry
                # "proxy"/"fallback" in their warnings, so skip those.
                warns = " ".join(r.warnings).lower()
                if "proxy" not in warns and "fallback" not in warns:
                    try:
                        from ..ears.status_tracker import mark_real_success
                        mark_real_success(ear.name)
                    except Exception:                              # noqa: BLE001
                        pass
            finally:
                # If the ear holds a GPU model, unload it after use.
                # "load per ear -> evaluate -> release immediately" keeps us in
                # a state where multiple GPU models are never resident at once.
                if getattr(ear, "uses_gpu_model", False):
                    try:
                        ear.release_model()
                    except Exception:                              # noqa: BLE001
                        pass
        return out
