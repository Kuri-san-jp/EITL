"""Three new metrics introduced by this work (β-plan novelty boosters).

  - SRA (Scope-Resolution Alignment): of all action turns, the fraction in
    which the agent's declared target_resolution matched the temporal
    scope of the tool(s) it actually invoked. High SRA ⇒ the agent
    matches problem scope to action scope (the β claim made operational).

  - DPS (Diagnoses Per Section): how many of the agent's diagnoses were
    section-localised. High DPS ⇒ multi-resolution observation is being
    used to localise issues, not just trigger global changes.

  - REC (Recovery from Rejection): of all rollback events, the fraction
    that were followed within K turns by an *accepted* action with a
    *different* scope. High REC ⇒ memory + rollback enables strategy
    revision (not stuck repeating).

All three are computed from one experiment run root (containing
`turn_NNN/response.json` files) plus the agent's RunResult / memory.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..tools.schemas import TOOL_CATALOG


# Which scopes "belong to" which target_resolution (β-plan operational mapping).
RESOLUTION_TO_SCOPES: Dict[str, Set[str]] = {
    "global":     {"static", "master"},
    "section":    {"section"},
    "short_term": {"dynamic", "automation"},
    "momentary":  {"transient", "master"},  # master limiter / fast limiter
}


@dataclass
class NoveltyMetrics:
    sra: float          # in [0, 1]
    dps: int            # absolute count
    dps_rate: float     # dps / n_turns
    rec: float          # in [0, 1]; NaN if no rejections
    n_turns: int
    n_rollbacks: int
    aligned_turns: int
    section_targeted_turns: int


# ---------- Per-turn parser ----------


def _scope_of(tool_name: str) -> Optional[str]:
    info = TOOL_CATALOG.get(tool_name)
    return info["scope"] if info else None


def _iter_turn_records(run_root: Path) -> List[Dict[str, Any]]:
    out = []
    for response_json in sorted(run_root.rglob("turn_*/response.json")):
        try:
            data = json.loads(response_json.read_text("utf-8"))
        except Exception:                                                # noqa: BLE001
            continue
        parsed = data.get("parsed_json") or {}
        tool_calls = data.get("tool_calls") or parsed.get("tool_calls") or []
        target_res = parsed.get("target_resolution")
        target_scope = parsed.get("target_scope")
        out.append({
            "path":            str(response_json),
            "target_resolution": target_res,
            "target_scope":      target_scope,
            "tool_calls":        tool_calls,
        })
    return out


# ---------- SRA ----------


def compute_sra(turns: List[Dict[str, Any]]) -> Dict[str, float]:
    """Scope-Resolution Alignment. Skips turns with no actionable tools."""
    aligned = 0
    counted = 0
    for t in turns:
        tr = t.get("target_resolution")
        if tr not in RESOLUTION_TO_SCOPES:
            continue
        if not t.get("tool_calls"):
            continue
        scopes_used = set()
        for c in t["tool_calls"]:
            sc = _scope_of(c.get("name", ""))
            if sc and sc not in ("perception", "state"):
                scopes_used.add(sc)
        if not scopes_used:
            continue
        counted += 1
        if scopes_used & RESOLUTION_TO_SCOPES[tr]:
            aligned += 1
    return {"aligned": aligned, "counted": counted,
            "sra": aligned / counted if counted else float("nan")}


# ---------- DPS ----------


def compute_dps(turns: List[Dict[str, Any]]) -> Dict[str, float]:
    n_section = 0
    n_turns = 0
    for t in turns:
        if not t.get("tool_calls") and t.get("target_resolution") is None:
            continue
        n_turns += 1
        tr = t.get("target_resolution")
        ts = t.get("target_scope") or ""
        if tr == "section":
            n_section += 1
            continue
        if any(s in ts.lower() for s in ("verse", "chorus", "bridge", "intro", "outro", "section")):
            n_section += 1
    return {"dps": n_section, "n_turns": n_turns,
            "dps_rate": n_section / n_turns if n_turns else 0.0}


# ---------- REC ----------


def compute_rec(action_history: List[Any], rejected_history: List[Any] = None,
                memory: Any = None, window: int = 3) -> Dict[str, float]:
    """Compute REC from a chronological action sequence.

    Inputs (any of):
      * `memory.decision_history` (list[str]) plus `memory.action_history`
      * Both `action_history` and `rejected_history` lists merged here

    For each rollback, look ahead `window` events; count it as a recovery
    if any accept exists with a different scope.
    """
    chronological: List[Any] = []
    if memory is not None:
        # Interleave by iteration if records have .iteration
        all_recs = list(memory.action_history) + list(memory.rejected_actions)
        all_recs.sort(key=lambda r: getattr(r, "iteration", 0))
        chronological = all_recs
    else:
        chronological = (action_history or []) + (rejected_history or [])
        chronological.sort(key=lambda r: getattr(r, "iteration", 0))

    rollbacks = [r for r in chronological if getattr(r, "decision", "") == "rollback"]
    if not rollbacks:
        return {"rec": float("nan"), "n_rollbacks": 0, "n_recovered": 0}

    recovered = 0
    for r in rollbacks:
        idx = chronological.index(r)
        for cand in chronological[idx + 1: idx + 1 + window]:
            if getattr(cand, "decision", "") == "accept":
                if (cand.scope or "") != (r.scope or ""):
                    recovered += 1
                    break
    return {"rec": recovered / len(rollbacks),
            "n_rollbacks": len(rollbacks),
            "n_recovered": recovered}


# ---------- One-shot bundler ----------


def compute_all_from_run(run_root: Path, memory: Any = None) -> NoveltyMetrics:
    turns = _iter_turn_records(run_root)
    sra = compute_sra(turns)
    dps = compute_dps(turns)
    rec = compute_rec([], memory=memory) if memory is not None else \
          {"rec": float("nan"), "n_rollbacks": 0, "n_recovered": 0}
    return NoveltyMetrics(
        sra=sra["sra"],
        dps=dps["dps"],
        dps_rate=dps["dps_rate"],
        rec=rec["rec"],
        n_turns=dps["n_turns"],
        n_rollbacks=rec["n_rollbacks"],
        aligned_turns=sra["aligned"],
        section_targeted_turns=dps["dps"],
    )


def metrics_to_dict(m: NoveltyMetrics) -> Dict[str, Any]:
    return {
        "sra":         m.sra,
        "dps":         m.dps,
        "dps_rate":    m.dps_rate,
        "rec":         m.rec,
        "n_turns":     m.n_turns,
        "n_rollbacks": m.n_rollbacks,
    }
