"""One-shot LLM baseline.

The LLM sees the stem analysis once and outputs the FULL action set in a
single JSON. No loop, no evaluator feedback. The exact "I told the LLM
once" comparison against the closed-loop agent (RQ1).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np

from ..agent.llm_backends.base import LLMBackend
from ..agent.prompts import stem_descriptors
from ..dsp.mix_state import MixState
from ..dsp.renderer import render
from ..resolution.multi_res_evaluator import MultiResolutionEvaluator
from ..resolution.section_detector import detect_sections
from ..tools.schemas import anthropic_tool_specs, TOOL_CATALOG
from ..tools.tool_router import ToolRouter
from ..tools.schemas import ToolCall
from .base import BaselineRunner, BaselineResult, TrialRecord


ONE_SHOT_SYSTEM = """You are a mixing engineer doing ONE-SHOT mixing.
You will be shown stem statistics and section boundaries. Output ONE JSON
object containing a `tool_calls` array — a complete mix plan in a single
response. You will NOT see the result or get to revise. Be conservative.

JSON shape:
{
  "diagnosis": "...",
  "tool_calls": [
    {"name":"apply_static_gain","arguments":{"track":"vocal","gain_db":-2.0}},
    {"name":"apply_static_eq","arguments":{"track":"vocal","freq":3000,"gain_db":2.0,"q":1.0}},
    ...,
    {"name":"apply_master_limiter","arguments":{"ceiling_db":-1.0,"release_ms":100}}
  ]
}
"""


class OneShotLLMBaseline(BaselineRunner):
    name = "one_shot_llm"

    def __init__(self, evaluator: MultiResolutionEvaluator,
                 llm: LLMBackend,
                 max_iterations: int = 1,   # always 1
                 seed: int = 42):
        super().__init__(evaluator, max_iterations, seed)
        self.llm = llm
        self.router = ToolRouter(evaluator=evaluator)

    async def run(self, stems: Dict[str, np.ndarray], sr: int) -> BaselineResult:
        t0 = time.time()
        init_state = MixState.initial_from_stems(stems, sr)
        rough = render(init_state)
        boundaries = detect_sections(rough, sr)
        init_state = init_state.with_sections(boundaries)

        initial_scores = await self.evaluator.evaluate(
            rough, sr, section_boundaries=boundaries,
            resolutions=["global", "section", "short_term", "momentary"],
        )

        # Stem summary prompt
        desc = stem_descriptors(stems, sr)
        prompt_lines = ["### Tracks"]
        for n, info in desc.items():
            prompt_lines.append(
                f"- {n}: {info['duration_sec']:.1f}s peak={info['peak_db']:.1f}dB rms={info['rms_db']:.1f}dB")
        prompt_lines.append("\n### Sections")
        for t, lab in boundaries:
            prompt_lines.append(f"- {t:.2f}s {lab}")
        prompt_lines.append("\nProduce ONE JSON object with a `tool_calls` array as the COMPLETE mix plan.")
        user = "\n".join(prompt_lines)

        resp = await self.llm.complete(
            system=ONE_SHOT_SYSTEM, user=user,
            tool_specs=anthropic_tool_specs(),
        )

        # Collect tool calls (prefer native tool_use, fall back to JSON)
        tool_calls = resp.tool_calls or (resp.parsed_json.get("tool_calls", [])
                                         if resp.parsed_json else [])

        state = init_state
        for c in tool_calls:
            try:
                tc = ToolCall(name=c["name"], arguments=c.get("arguments", {}))
                if self.router.is_action(tc.name):
                    state = self.router.apply_action(state, tc)
            except Exception:                                  # noqa: BLE001
                continue

        try:
            cand = render(state)
        except Exception:                                  # noqa: BLE001
            state = init_state
            cand = rough

        scores = await self.evaluator.evaluate(
            cand, sr, section_boundaries=boundaries,
            resolutions=["global", "section", "short_term", "momentary"],
        )
        reward = float(scores.aggregate_reward())

        trials = [TrialRecord(
            iteration=1, reward=reward,
            flat_scores=dict(scores.global_scores),
            state_id=state.state_id,
            action_summary=f"one-shot LLM, {len(tool_calls)} calls",
            elapsed_sec=time.time() - t0,
        )]
        return BaselineResult(
            name=self.name, final_state=state,
            final_scores=scores, initial_scores=initial_scores,
            trials=trials, best_iteration=1, best_reward=reward,
            total_cost_usd=resp.cost_usd, elapsed_sec=time.time() - t0,
            extra={"n_tool_calls_proposed": len(tool_calls),
                   "raw_response": resp.raw_text[:500]},
        )
