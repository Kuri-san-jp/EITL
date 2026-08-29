"""Common interface for every baseline.

All baselines MUST:
  • use the same MultiResolutionEvaluator + ear ensemble as the agent
  • use the same iteration / trial budget for sample-efficiency comparison
  • return a `BaselineResult` containing the per-iter reward trajectory
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..dsp.mix_state import MixState
from ..resolution.multi_res_evaluator import MultiResolutionEvaluator, MultiResolutionScores


@dataclass
class TrialRecord:
    iteration: int
    reward: float
    flat_scores: Dict[str, float]
    state_id: str
    action_summary: str
    elapsed_sec: float = 0.0


@dataclass
class BaselineResult:
    name: str
    final_state: MixState
    final_scores: MultiResolutionScores
    initial_scores: MultiResolutionScores
    trials: List[TrialRecord] = field(default_factory=list)
    best_iteration: int = 0
    best_reward: float = 0.0
    total_cost_usd: float = 0.0
    elapsed_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def reward_curve(self) -> List[float]:
        """Per-iter best-so-far reward (monotone non-decreasing)."""
        best = -np.inf
        curve = []
        for t in self.trials:
            best = max(best, t.reward)
            curve.append(float(best))
        return curve


class BaselineRunner(ABC):
    """Abstract baseline. Subclasses implement `run`."""
    name: str = "base"

    def __init__(self, evaluator: MultiResolutionEvaluator,
                 max_iterations: int,
                 seed: int = 42):
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.seed = seed

    @abstractmethod
    async def run(self,
                  stems: Dict[str, np.ndarray],
                  sr: int) -> BaselineResult:
        raise NotImplementedError
