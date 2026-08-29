"""Prompt builders (Jinja2 templates)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from jinja2 import Template


def load_system_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_iteration_template(path: str) -> Template:
    return Template(Path(path).read_text(encoding="utf-8"))


def build_iteration_prompt(template: Template,
                           iteration: int,
                           max_iterations: int,
                           tracks: Dict[str, Dict[str, float]],
                           sections: List[Tuple[float, str]],
                           current_scores: Dict[str, Any],
                           best_scores: Dict[str, Any],
                           best_state_id: str,
                           best_reward: float,
                           history: List[Dict[str, Any]],
                           rejected: List[Dict[str, Any]],
                           tool_list: str,
                           iters_remaining: int,
                           usd_remaining: float,
                           rule_advice: str = "(no advice)",
                           allowed_scopes: Optional[List[str]] = None,
                           resolutions: Optional[List[str]] = None,
                           ) -> str:
    return template.render(
        iteration=iteration,
        max_iterations=max_iterations,
        tracks=tracks,
        sections=sections,
        current_scores_json=json.dumps(current_scores, indent=2, default=_safe),
        best_scores_json=json.dumps(best_scores, indent=2, default=_safe),
        best_state_id=best_state_id,
        best_reward=best_reward,
        history=history,
        rejected=rejected,
        tool_list=tool_list,
        iters_remaining=iters_remaining,
        usd_remaining=usd_remaining,
        rule_advice=rule_advice,
        allowed_scopes=allowed_scopes or [],
        resolutions=resolutions or [],
    )


def stem_descriptors(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, float]]:
    """Cheap per-stem stats for the iteration prompt."""
    out: Dict[str, Dict[str, float]] = {}
    for name, arr in stems.items():
        a = arr if arr.ndim == 1 else arr.mean(axis=0)
        peak = float(np.max(np.abs(a)) + 1e-12)
        rms = float(np.sqrt(np.mean(a**2) + 1e-12))
        out[name] = {
            "duration_sec": a.shape[-1] / sr,
            "peak_db": float(20 * np.log10(peak)),
            "rms_db":  float(20 * np.log10(rms)),
        }
    return out


def _safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return str(obj)
