"""Accept / rollback / stop decisions.

These rules are documented in `configs/agent/prompts/decision_rules.txt`
and embedded in the system prompt. The CODE version is the source of
truth and is enforced after every candidate render.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ..ears.aggregation import aggregate, divergence
from ..dsp.safety import SafetyLimits


class DecisionKind(str, Enum):
    ACCEPT = "accept"
    WEAK = "weak"
    INVESTIGATE = "investigate"
    ROLLBACK = "rollback"


@dataclass
class Decision:
    kind: DecisionKind
    reason: str = ""
    delta_reward: float = 0.0

    @property
    def is_accept(self) -> bool: return self.kind is DecisionKind.ACCEPT
    @property
    def is_weak(self) -> bool: return self.kind is DecisionKind.WEAK
    @property
    def is_rollback(self) -> bool: return self.kind is DecisionKind.ROLLBACK
    @property
    def is_investigate(self) -> bool: return self.kind is DecisionKind.INVESTIGATE


@dataclass
class DecisionPolicy:
    target_lufs: float = -14.0
    lufs_tolerance: float = 6.0
    true_peak_max_db: float = -1.0
    stereo_corr_min: float = -0.3
    min_delta: float = 0.02
    epsilon: float = 0.005
    patience: int = 3
    max_iterations: int = 20

    def safety_limits(self) -> SafetyLimits:
        return SafetyLimits(true_peak_db_max=self.true_peak_max_db,
                            lufs_target=self.target_lufs,
                            lufs_tolerance=self.lufs_tolerance,
                            stereo_corr_min=self.stereo_corr_min)

    def evaluate_candidate(self,
                           prev_flat: Dict[str, float],
                           cand_flat: Dict[str, float]) -> Decision:
        # ---- hard gates ----
        tp = cand_flat.get("true_peak.true_peak_db")
        if tp is not None and tp > self.true_peak_max_db:
            return Decision(DecisionKind.ROLLBACK,
                            reason=f"true peak {tp:.2f}dB > {self.true_peak_max_db}dB")
        lufs = cand_flat.get("lufs.integrated_lufs")
        if lufs is not None and abs(lufs - self.target_lufs) > self.lufs_tolerance:
            return Decision(DecisionKind.ROLLBACK,
                            reason=f"LUFS {lufs:.1f} outside target ±{self.lufs_tolerance}")
        corr = cand_flat.get("stereo.stereo_correlation")
        if corr is not None and corr < self.stereo_corr_min:
            return Decision(DecisionKind.ROLLBACK,
                            reason=f"stereo corr {corr:.2f} < {self.stereo_corr_min}")

        # ---- reward delta ----
        prev_r = aggregate(prev_flat)
        cand_r = aggregate(cand_flat)
        delta = cand_r - prev_r

        # Per-key regression check on Audiobox 4-axis (most informative)
        for k in ("audiobox.PQ", "audiobox.PC", "audiobox.CE", "audiobox.CU"):
            if k in prev_flat and k in cand_flat:
                d = cand_flat[k] - prev_flat[k]
                if d < -self.epsilon and (cand_flat[k] - prev_flat[k]) < -3 * self.epsilon:
                    return Decision(DecisionKind.INVESTIGATE,
                                    reason=f"{k} regressed by {d:.3f}",
                                    delta_reward=delta)

        if delta >= self.min_delta:
            return Decision(DecisionKind.ACCEPT, reason="reward improved", delta_reward=delta)
        if delta >= -self.epsilon:
            return Decision(DecisionKind.WEAK, reason="no meaningful change", delta_reward=delta)
        return Decision(DecisionKind.ROLLBACK,
                        reason=f"reward decreased by {-delta:.3f}",
                        delta_reward=delta)

    def should_stop(self, decisions: List[str], iteration: int) -> bool:
        if iteration >= self.max_iterations:
            return True
        if len(decisions) >= self.patience:
            recent = decisions[-self.patience:]
            if all(d in ("rollback", "weak") for d in recent):
                return True
        return False
