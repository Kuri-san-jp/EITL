"""Agent memory: action history, rejected actions, checkpoints, best mix."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActionRecord:
    iteration: int
    scope: str
    summary: str
    tool_calls: List[Dict[str, Any]]
    prev_scores_flat: Dict[str, float]
    post_scores_flat: Dict[str, float]
    delta_reward: float
    decision: str      # "accept" | "rollback" | "investigate" | "weak"
    reason: str = ""


class AgentMemory:
    def __init__(self):
        self.action_history: List[ActionRecord] = []
        self.rejected_actions: List[ActionRecord] = []
        self.checkpoints: Dict[str, "Any"] = {}     # label -> MixState
        self.scores_by_state: Dict[str, Any] = {}   # state_id -> MultiResolutionScores
        self.best_state_id: Optional[str] = None
        self.best_reward: float = -math.inf
        self.action_fingerprints: set = set()
        self.decision_history: List[str] = []

    # ---------- checkpoints ----------

    def save_checkpoint(self, label: str, state, scores) -> None:
        self.checkpoints[label] = state
        self.scores_by_state[state.state_id] = scores

    def update_best(self, state_id: str, state, scores, reward: float) -> None:
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_state_id = state_id
            self.checkpoints["best"] = state
            self.scores_by_state[state_id] = scores

    # ---------- actions ----------

    def record_action(self, rec: ActionRecord) -> None:
        if rec.decision in ("rollback", "weak"):
            self.rejected_actions.append(rec)
        else:
            self.action_history.append(rec)
        self.decision_history.append(rec.decision)
        self.action_fingerprints.add(self._fingerprint(rec))

    def _fingerprint(self, rec: ActionRecord) -> str:
        payload = json.dumps([rec.tool_calls,
                              sorted(rec.prev_scores_flat.items())],
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def has_tried_similar(self, tool_calls: List[Dict[str, Any]],
                          current_scores: Dict[str, float]) -> bool:
        payload = json.dumps([tool_calls, sorted(current_scores.items())],
                             sort_keys=True, default=str)
        fp = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return fp in self.action_fingerprints

    # ---------- prompt-facing summaries ----------

    def history_summary(self, n: int = 8) -> List[Dict[str, Any]]:
        recent = self.action_history[-n:]
        return [{"iteration": r.iteration, "scope": r.scope,
                 "decision": r.decision,
                 "delta_reward": r.delta_reward, "summary": r.summary}
                for r in recent]

    def rejected_summary(self, n: int = 8) -> List[Dict[str, Any]]:
        recent = self.rejected_actions[-n:]
        return [{"summary": r.summary, "reason": r.reason} for r in recent]
