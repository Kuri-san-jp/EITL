"""The orchestrator loop — observe → diagnose → act → evaluate → decide."""
from __future__ import annotations

import asyncio
import json
import logging as _log
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..dsp.mix_state import MixState
from ..dsp.renderer import render
from ..dsp.loudness_norm import normalize_for_eval
from ..dsp.safety import (
    MixSafetyError, SafetyLimits,
    check_true_peak, check_lufs, check_stereo_correlation,
)
from ..ears.base import Ear
from ..ears.aggregation import aggregate, flatten as flatten_results
from ..resolution.multi_res_evaluator import MultiResolutionEvaluator, MultiResolutionScores
from ..resolution.section_detector import detect_sections
from ..tools.tool_router import ToolRouter
from ..tools.schemas import ToolCall, anthropic_tool_specs, TOOL_CATALOG
from .decision_policy import DecisionPolicy, Decision, DecisionKind
from .memory import AgentMemory, ActionRecord
from .prompts import build_iteration_prompt, load_system_prompt, load_iteration_template, stem_descriptors
from .rule_advisor import RuleAdvisor
from .llm_backends.base import LLMBackend, LLMResponse


_logger = _log.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    max_iterations: int = 20
    target_lufs: float = -14.0
    lufs_tolerance: float = 6.0
    min_delta: float = 0.02
    epsilon: float = 0.005
    patience: int = 3
    history_window: int = 8
    seed: int = 42
    resolutions: List[str] = field(default_factory=lambda: ["global", "section", "short_term", "momentary"])
    # S0..S4 ablation knob — None means "all scopes". Used by E1 R×S sweep.
    allowed_scopes: Optional[List[str]] = None

    system_prompt_path: str = "configs/agent/prompts/system_prompt.txt"
    iteration_prompt_path: str = "configs/agent/prompts/iteration_prompt.j2"
    output_dir: str = "outputs/runs/default"
    budget_usd: float = 2.0


@dataclass
class RunResult:
    final_state: MixState
    final_scores: MultiResolutionScores
    initial_scores: MultiResolutionScores
    memory: AgentMemory
    n_iterations: int
    total_cost_usd: float
    elapsed_sec: float


class Orchestrator:
    def __init__(self,
                 llm: LLMBackend,
                 ears: List[Ear],
                 config: OrchestratorConfig):
        self.llm = llm
        self.ears = ears
        self.config = config
        self.evaluator = MultiResolutionEvaluator(ears=ears)
        self.policy = DecisionPolicy(
            target_lufs=config.target_lufs,
            lufs_tolerance=config.lufs_tolerance,
            min_delta=config.min_delta,
            epsilon=config.epsilon,
            patience=config.patience,
            max_iterations=config.max_iterations,
        )
        self.router = ToolRouter(
            evaluator=self.evaluator,
            allowed_scopes=set(config.allowed_scopes) if config.allowed_scopes else None,
        )
        self.system_prompt = load_system_prompt(config.system_prompt_path)
        self.iter_template = load_iteration_template(config.iteration_prompt_path)
        # New: advisory layer that proposes concrete ToolCalls from the ear scores
        self.advisor = RuleAdvisor(target_lufs=config.target_lufs)

    # ---------- public ----------

    async def run(self,
                  stems: Dict[str, np.ndarray],
                  sr: int,
                  reference: Optional[np.ndarray] = None) -> RunResult:
        t0 = time.time()
        np.random.seed(self.config.seed)

        # Initial state
        state = MixState.initial_from_stems(stems, sr)
        rough_mix = render(state)
        boundaries = detect_sections(rough_mix, sr)
        state = state.with_sections(boundaries)

        _logger.info("Detected sections: %s", boundaries)

        # Wire ears that need extra context (stems / fallback reference)
        self._wire_ears(stems, rough_mix, reference)
        if reference is None:
            # Use the initial mix as the reference for Tier 2 ears
            reference = rough_mix

        initial_scores = await self.evaluator.evaluate(
            normalize_for_eval(rough_mix, sr),
            sr,
            section_boundaries=boundaries,
            resolutions=self.config.resolutions,
            reference=reference,
        )
        memory = AgentMemory()
        init_reward = initial_scores.aggregate_reward()
        memory.save_checkpoint("initial", state, initial_scores)
        memory.update_best(state.state_id, state, initial_scores, init_reward)

        _logger.info("Initial reward = %.4f", init_reward)

        total_cost = 0.0
        latest_scores = initial_scores
        latest_flat = flatten_global(initial_scores)

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Iteration loop
        for it in range(1, self.config.max_iterations + 1):
            usd_remaining = self.config.budget_usd - total_cost
            if usd_remaining <= 0:
                _logger.warning("Budget exhausted at iter %d", it)
                break

            # Have RuleAdvisor propose ToolCalls from the ear scores
            prev_flat_for_advice = (
                memory.action_history[-1].post_scores_flat
                if memory.action_history else dict(initial_scores.global_scores)
            )
            advice = self.advisor.advise(
                cur_flat=latest_flat,
                prev_flat=prev_flat_for_advice,
                section_scores=latest_scores.section_scores,
                stems=list(stems.keys()),
            )

            user_prompt = build_iteration_prompt(
                template=self.iter_template,
                iteration=it,
                max_iterations=self.config.max_iterations,
                tracks=stem_descriptors(stems, sr),
                sections=boundaries,
                current_scores=latest_scores.to_dict(),
                best_scores=memory.scores_by_state[memory.best_state_id].to_dict(),
                best_state_id=memory.best_state_id or "n/a",
                best_reward=memory.best_reward,
                history=memory.history_summary(self.config.history_window),
                rejected=memory.rejected_summary(self.config.history_window),
                tool_list=_compact_tool_list(),
                iters_remaining=self.config.max_iterations - it + 1,
                usd_remaining=usd_remaining,
                rule_advice=advice.to_prompt_block(),
                allowed_scopes=list(self.config.allowed_scopes)
                    if self.config.allowed_scopes else None,
                resolutions=self.config.resolutions,
            )

            llm_resp: LLMResponse = await self.llm.complete(
                system=self.system_prompt,
                user=user_prompt,
                tool_specs=anthropic_tool_specs(),
            )
            total_cost += llm_resp.cost_usd

            stop = bool(llm_resp.parsed_json.get("stop", False))
            tool_calls = _merge_tool_calls(llm_resp)

            _logger.info("iter=%d  cost=%.4f  stop=%s  n_calls=%d",
                         it, llm_resp.cost_usd, stop, len(tool_calls))

            self._dump_turn(out_dir, it, user_prompt, llm_resp)

            if stop:
                _logger.info("LLM requested stop at iter %d", it)
                break
            if not tool_calls:
                _logger.warning("LLM produced no tool calls at iter %d; skipping", it)
                continue

            # Apply actions (hard check that scope is consistent with target_resolution)
            target_resolution = llm_resp.parsed_json.get("target_resolution") \
                if llm_resp.parsed_json else None
            candidate_state = state
            applied_ok = True
            for c in tool_calls:
                tc = ToolCall(**c)
                if not self.router.is_action(tc.name):
                    continue
                try:
                    candidate_state = self.router.apply_action(
                        candidate_state, tc, target_resolution=target_resolution)
                except Exception as ex:  # noqa: BLE001
                    _logger.error("Action %s failed: %s", tc.name, ex)
                    applied_ok = False
                    break

            if not applied_ok:
                continue

            # Render + safety
            try:
                cand_audio = render(candidate_state)
                self._enforce_safety(cand_audio, sr, latest_flat)
            except MixSafetyError as ex:
                memory.record_action(ActionRecord(
                    iteration=it, scope=_scope_of(tool_calls),
                    summary=_summarize(tool_calls),
                    tool_calls=tool_calls,
                    prev_scores_flat=latest_flat,
                    post_scores_flat={},
                    delta_reward=0.0,
                    decision="rollback", reason=f"safety: {ex}",
                ))
                continue

            cand_scores = await self.evaluator.evaluate(
                normalize_for_eval(cand_audio, sr),
                sr,
                section_boundaries=boundaries,
                resolutions=self.config.resolutions,
                reference=reference,
            )
            cand_flat = flatten_global(cand_scores)

            decision = self.policy.evaluate_candidate(latest_flat, cand_flat)

            memory.record_action(ActionRecord(
                iteration=it, scope=_scope_of(tool_calls),
                summary=_summarize(tool_calls),
                tool_calls=tool_calls,
                prev_scores_flat=latest_flat,
                post_scores_flat=cand_flat,
                delta_reward=decision.delta_reward,
                decision=decision.kind.value,
                reason=decision.reason,
            ))

            if decision.is_accept or decision.is_investigate:
                # For now treat INVESTIGATE as ACCEPT (Tier 3 judge to be added)
                state = candidate_state
                latest_scores = cand_scores
                latest_flat = cand_flat
                reward = cand_scores.aggregate_reward()
                memory.update_best(state.state_id, state, cand_scores, reward)
            elif decision.is_weak:
                # WEAK = "no meaningful change", but we still advance the state so
                # the agent keeps exploring. best is updated by the update_best
                # logic only when the reward improves, so a WEAK step in the
                # worsening direction is not reflected in best (the state advances
                # without a rollback).
                state = candidate_state
                latest_scores = cand_scores
                latest_flat = cand_flat

            if self.policy.should_stop(memory.decision_history, it):
                _logger.info("Stop criterion met at iter %d", it)
                break

        # Final: always return "best" (the state with the highest reward so far).
        # best is initialized with initial at the start of the run (update_best, L125),
        # so if nothing ever improved, best == initial and delta_reward = 0 is
        # reported as the honest result.
        #
        # NOTE: the old v4/v5 fix returned latest_state (possibly a degraded mix
        # advanced through WEAK) as final when there were 0 ACCEPTs, but that was a
        # band-aid to "look like it works" and scientifically wrong behavior, since
        # it reports a degraded mix as the outcome. Reverted to best-only on
        #  by design. See the internal notes for details.
        best_state = memory.checkpoints.get("best", state)
        best_scores = memory.scores_by_state.get(memory.best_state_id,
                                                 initial_scores)

        result = RunResult(
            final_state=best_state,
            final_scores=best_scores,
            initial_scores=initial_scores,
            memory=memory,
            n_iterations=len(memory.decision_history),
            total_cost_usd=total_cost,
            elapsed_sec=time.time() - t0,
        )
        return result

    # ---------- helpers ----------

    def _wire_ears(self, stems: Dict[str, np.ndarray],
                   rough_mix: np.ndarray,
                   reference: Optional[np.ndarray]) -> None:
        """Inject context (original stems, reference fallback) into ears that
        cannot get it purely from `evaluate()` arguments."""
        for ear in self.ears:
            # ReseparationSDREar needs original stems out-of-band
            attach = getattr(ear, "attach_stems", None)
            if callable(attach):
                try:
                    attach(stems)
                except Exception as ex:  # noqa: BLE001
                    _logger.warning("attach_stems failed on %s: %s", ear.name, ex)

    def _enforce_safety(self, audio: np.ndarray, sr: int, prev_flat: Dict[str, float]) -> None:
        limits = self.policy.safety_limits()
        for check in (
            check_true_peak(audio, limits),
            check_stereo_correlation(audio, limits),
        ):
            if check:
                raise MixSafetyError(check)
        # LUFS via current ears (if present) — cheap re-check via pyloudnorm
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            arr = audio.T if audio.ndim == 2 else audio
            if arr.ndim == 1:
                arr = arr[:, None]
            lu = float(meter.integrated_loudness(arr))
            msg = check_lufs(lu, limits)
            if msg:
                raise MixSafetyError(msg)
        except ImportError:
            pass

    def _dump_turn(self, out_dir: Path, iteration: int,
                   prompt: str, response: LLMResponse) -> None:
        d = out_dir / f"turn_{iteration:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompt.txt").write_text(prompt, encoding="utf-8")
        (d / "response.json").write_text(
            json.dumps({
                "parsed_json": response.parsed_json,
                "tool_calls":  response.tool_calls,
                "raw_text":    response.raw_text,
                "cost_usd":    response.cost_usd,
                "tokens_in":   response.tokens_in,
                "tokens_out":  response.tokens_out,
                "meta":        response.meta,
            }, indent=2, default=str),
            encoding="utf-8",
        )


# ---------- module helpers ----------


def flatten_global(scores: MultiResolutionScores) -> Dict[str, float]:
    return dict(scores.global_scores)


def _merge_tool_calls(resp: LLMResponse) -> List[Dict[str, Any]]:
    """Prefer native tool_use; fall back to parsed_json.tool_calls."""
    if resp.tool_calls:
        return resp.tool_calls
    raw = resp.parsed_json.get("tool_calls", []) if resp.parsed_json else []
    out = []
    for c in raw:
        if isinstance(c, dict) and "name" in c:
            out.append({"name": c["name"], "arguments": c.get("arguments", {})})
    return out


def _scope_of(tool_calls: List[Dict[str, Any]]) -> str:
    if not tool_calls:
        return "n/a"
    scopes = set()
    for c in tool_calls:
        info = TOOL_CATALOG.get(c.get("name", ""))
        if info:
            scopes.add(info["scope"])
    return ",".join(sorted(scopes)) or "unknown"


def _summarize(tool_calls: List[Dict[str, Any]]) -> str:
    parts = [c.get("name", "?") + "(" + ",".join(f"{k}={v}" for k, v in (c.get("arguments") or {}).items()) + ")"
             for c in tool_calls]
    return "; ".join(parts)


def _compact_tool_list() -> str:
    lines = []
    for name, info in TOOL_CATALOG.items():
        schema = info["args"].model_json_schema()
        props = schema.get("properties", {})
        keys = ",".join(props.keys())
        lines.append(f"  - {name}({keys})  [{info['scope']}]")
    return "\n".join(lines)
