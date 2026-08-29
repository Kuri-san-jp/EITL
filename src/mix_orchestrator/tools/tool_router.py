"""Dispatch ToolCall → action handler / perception call."""
from __future__ import annotations

import logging as _log
from typing import Any, Dict, List, Optional, Set

from ..dsp.mix_state import MixState
from .action_tools import ACTION_HANDLERS
from .schemas import ToolCall, TOOL_CATALOG


# Every defined scope in TOOL_CATALOG (used as default = "no restriction").
ALL_SCOPES: Set[str] = {info["scope"] for info in TOOL_CATALOG.values()}

# Tokens the LLM hallucinates instead of a real stem name.
_PSEUDO_TRACKS: Set[str] = {
    "all", "master", "mix", "song", "whole", "everything",
    "all_stems", "all_tracks", "global", "bus", "main",
}

# When a pseudo-track is paired with a static-scope tool, prefer the
# matching master-scope tool. The 'track' arg is stripped on redirect.
_STATIC_TO_MASTER: Dict[str, str] = {
    "apply_static_eq":         "apply_master_eq",
    "apply_static_compressor": "apply_master_compressor",
}

_logger = _log.getLogger(__name__)


# Table of action scopes consistent with each target_resolution.
# ToolCalls whose scope does not match the target_resolution the LLM declared are rejected.
# perception / state are allowed at any resolution (querying information and managing
# state are harmless).
RESOLUTION_TO_ALLOWED_SCOPES: Dict[str, Set[str]] = {
    "global":     {"static", "master", "perception", "state"},
    "section":    {"section", "perception", "state"},
    "short_term": {"dynamic", "automation", "perception", "state"},
    "momentary":  {"transient", "master", "perception", "state"},
}


class ScopeViolation(RuntimeError):
    """Raised when an action's scope is outside the configured allow-list."""


class ResolutionMismatch(RuntimeError):
    """Raised when an action's scope does not match the declared
    target_resolution. Enforces the resolution-scope alignment that the
    novelty metric SRA only measured (now made a hard guarantee)."""


class ToolRouter:
    """Routes structured tool calls to the right handler.

    Action tools mutate state. Perception/state tools are delegated to
    the orchestrator (which holds the evaluator and memory).

    Set `allowed_scopes` to restrict which action scopes the agent may
    invoke (used for S0..S4 ablations). When `None`, all scopes are
    permitted (the default for normal runs).
    """

    def __init__(self,
                 evaluator=None,
                 memory=None,
                 allowed_scopes: Optional[Set[str]] = None,
                 enforce_resolution_match: bool = True):
        self.evaluator = evaluator
        self.memory = memory
        self.allowed_scopes: Set[str] = set(allowed_scopes) if allowed_scopes else ALL_SCOPES
        # Whether to reject a mismatch between target_resolution and action scope
        # at run time. Enabled by default in Phase 1 (hard guarantee of
        # resolution-scope alignment).
        self.enforce_resolution_match = enforce_resolution_match

    # ---------- introspection ----------

    def is_action(self, name: str) -> bool:
        return name in ACTION_HANDLERS

    def is_allowed(self, name: str) -> bool:
        info = TOOL_CATALOG.get(name)
        if info is None:
            return False
        return info["scope"] in self.allowed_scopes

    def allowed_tool_names(self) -> List[str]:
        return [n for n, info in TOOL_CATALOG.items() if info["scope"] in self.allowed_scopes]

    # ---------- execution ----------

    def apply_action(self, state: MixState, call: ToolCall,
                     target_resolution: Optional[str] = None) -> MixState:
        if call.name not in ACTION_HANDLERS:
            raise KeyError(f"unknown action tool {call.name!r}")

        # ---- LLM-hallucination fixups on the `track` argument ----
        call = self._fixup_track_arg(state, call)

        if not self.is_allowed(call.name):
            scope = TOOL_CATALOG[call.name]["scope"]
            raise ScopeViolation(
                f"{call.name!r} (scope={scope!r}) is outside the allow-list "
                f"{sorted(self.allowed_scopes)}"
            )

        # ---- consistency check between target_resolution and scope ----
        if (self.enforce_resolution_match and target_resolution
                and target_resolution in RESOLUTION_TO_ALLOWED_SCOPES):
            scope = TOOL_CATALOG[call.name]["scope"]
            allowed = RESOLUTION_TO_ALLOWED_SCOPES[target_resolution]
            if scope not in allowed:
                raise ResolutionMismatch(
                    f"{call.name!r} (scope={scope!r}) does not match "
                    f"target_resolution={target_resolution!r} "
                    f"(allowed scopes for this resolution: {sorted(allowed)})"
                )

        # Validate args via Pydantic (raises if invalid)
        ArgsCls = TOOL_CATALOG[call.name]["args"]
        ArgsCls(**call.arguments)
        return ACTION_HANDLERS[call.name](state, call.arguments)

    # ---------- LLM hallucination fixups ----------

    def _fixup_track_arg(self, state: MixState, call: ToolCall) -> ToolCall:
        """Repair pseudo or unknown track names from LLM hallucinations.

        Rules:
          (a) `track in {'all','master','mix',...}` paired with a static tool
              that has a master counterpart → redirect to master tool, drop `track`.
          (b) `track in {'all',...}` but no master counterpart → fall back to
              the first stem name (with a warning log).
          (c) `track` is not a known stem at all → fall back to first stem.
        """
        if "track" not in call.arguments:
            return call
        raw = str(call.arguments["track"])
        t = raw.strip().lower()
        stems = list(state.stems.keys())

        if t in _PSEUDO_TRACKS:
            target = _STATIC_TO_MASTER.get(call.name)
            if target is not None:
                new_args = {k: v for k, v in call.arguments.items() if k != "track"}
                _logger.warning(
                    "hallucination fixup: %s(track=%r) → %s(...)",
                    call.name, raw, target,
                )
                return ToolCall(name=target, arguments=new_args)
            # No master counterpart → first stem fallback
            if stems:
                _logger.warning(
                    "hallucination fixup: %s(track=%r) → track=%r (first stem)",
                    call.name, raw, stems[0],
                )
                return ToolCall(
                    name=call.name,
                    arguments={**call.arguments, "track": stems[0]},
                )

        if raw not in state.stems and stems:
            # Sometimes the LLM emits a near-name like "vocal" vs "vocals" —
            # try case-insensitive substring match before defaulting.
            match = next(
                (s for s in stems if t == s.lower() or t in s.lower() or s.lower() in t),
                None,
            )
            chosen = match or stems[0]
            _logger.warning(
                "hallucination fixup: %s(track=%r) not in stems → track=%r",
                call.name, raw, chosen,
            )
            return ToolCall(
                name=call.name,
                arguments={**call.arguments, "track": chosen},
            )
        return call

    def apply_actions(self, state: MixState, calls: List[ToolCall]) -> MixState:
        for c in calls:
            if self.is_action(c.name):
                state = self.apply_action(state, c)
        return state
