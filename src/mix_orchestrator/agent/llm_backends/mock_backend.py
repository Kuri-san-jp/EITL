"""Deterministic scripted backend for smoke tests (no API key)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import LLMBackend, LLMResponse


class MockBackend(LLMBackend):
    """Emits a small scripted sequence of actions.

    Iteration 1 → static_gain on the first track
    Iteration 2 → static_eq high-shelf cut on the first track
    Iteration 3 → master_limiter
    Iteration 4+→ stop=True
    """
    name = "mock"
    model = "mock-deterministic-v1"

    def __init__(self, tracks: List[str]):
        self.tracks = tracks
        self.turn = 0

    async def complete(self, system: str, user: str,
                       tool_specs: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        self.turn += 1
        t0 = self.tracks[0] if self.tracks else "track"

        if self.turn == 1:
            call = {"name": "apply_static_gain",
                    "arguments": {"track": t0, "gain_db": -2.0}}
            parsed = {
                "diagnosis": f"Mock: trim {t0} by 2 dB to check the loop.",
                "target_resolution": "global",
                "target_scope": t0,
                "planned_actions": [f"static_gain {t0} -2 dB"],
                "tool_calls": [call],
                "acceptance_criteria": {"min_delta_at_target": 0.0,
                                        "no_regression_elsewhere": True,
                                        "lufs_within": 3.0,
                                        "no_clipping": True},
                "stop": False,
            }
            return LLMResponse(parsed_json=parsed, tool_calls=[call],
                               raw_text="mock turn 1", cost_usd=0.0)

        if self.turn == 2:
            call = {"name": "apply_static_eq",
                    "arguments": {"track": t0, "freq": 8000.0, "gain_db": -1.5, "q": 0.7}}
            parsed = {
                "diagnosis": "Mock: gentle high-shelf cut to test EQ path.",
                "target_resolution": "global",
                "target_scope": t0,
                "planned_actions": ["high-shelf -1.5dB @ 8kHz"],
                "tool_calls": [call],
                "acceptance_criteria": {"min_delta_at_target": 0.0,
                                        "no_regression_elsewhere": True,
                                        "lufs_within": 3.0,
                                        "no_clipping": True},
                "stop": False,
            }
            return LLMResponse(parsed_json=parsed, tool_calls=[call],
                               raw_text="mock turn 2", cost_usd=0.0)

        if self.turn == 3:
            call = {"name": "apply_master_limiter",
                    "arguments": {"ceiling_db": -1.0, "release_ms": 100.0}}
            parsed = {
                "diagnosis": "Mock: brick-wall the master.",
                "target_resolution": "momentary",
                "target_scope": "master",
                "planned_actions": ["master limiter ceiling -1.0 dB"],
                "tool_calls": [call],
                "acceptance_criteria": {"min_delta_at_target": 0.0,
                                        "no_regression_elsewhere": True,
                                        "lufs_within": 3.0,
                                        "no_clipping": True},
                "stop": False,
            }
            return LLMResponse(parsed_json=parsed, tool_calls=[call],
                               raw_text="mock turn 3", cost_usd=0.0)

        parsed = {
            "diagnosis": "Mock: converged after 3 actions.",
            "target_resolution": "global",
            "target_scope": "n/a",
            "planned_actions": [],
            "tool_calls": [],
            "acceptance_criteria": {},
            "stop": True,
        }
        return LLMResponse(parsed_json=parsed, tool_calls=[],
                           raw_text="mock stop", cost_usd=0.0)
