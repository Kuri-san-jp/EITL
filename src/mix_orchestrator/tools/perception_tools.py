"""Perception tools the LLM can call (wrappers around the evaluator)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


async def evaluate_multi_resolution(evaluator, audio, sr, section_boundaries, args: dict) -> Dict[str, Any]:
    resolutions = args.get("resolutions", ["global", "section", "short_term", "momentary"])
    scores = await evaluator.evaluate(audio, sr, section_boundaries=section_boundaries, resolutions=resolutions)
    return scores.to_dict()


async def score_ear(evaluator, audio, sr, args: dict) -> Dict[str, Any]:
    ear_name = args["ear"]
    win = args.get("window_sec")
    ear = next((e for e in evaluator.ears if e.name == ear_name), None)
    if ear is None:
        return {"error": f"unknown ear {ear_name!r}"}
    r = await ear.evaluate(audio, sr, window=tuple(win) if win else None)
    return {"score": r.score, "warnings": r.warnings}
