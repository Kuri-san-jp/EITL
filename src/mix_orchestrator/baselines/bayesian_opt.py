"""Optuna TPE Bayesian optimization baseline (CRITICAL spec baseline).

Each trial samples a `FullActionSet` from the same continuous space the
agent has access to, applies it to a fresh MixState, renders, evaluates,
and returns the aggregate reward. Optuna's TPE sampler then proposes
the next configuration.

The whole point of this baseline is the sample-efficiency curve:
"How many trials does BO need to reach the LLM agent's final reward?"
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import numpy as np

from ..dsp.mix_state import MixState
from ..dsp.renderer import render
from ..dsp.safety import MixSafetyError
from ..ears.aggregation import flatten
from ..resolution.multi_res_evaluator import MultiResolutionEvaluator
from ..resolution.section_detector import detect_sections
from .base import BaselineRunner, BaselineResult, TrialRecord
from ._action_space import (
    STEM_PARAM_BOUNDS, MASTER_PARAM_BOUNDS,
    StemParams, MasterParams, FullActionSet,
    apply_action_set,
)


class OptunaBOBaseline(BaselineRunner):
    name = "bayesian_opt_tpe"

    def __init__(self, evaluator: MultiResolutionEvaluator,
                 max_iterations: int,
                 seed: int = 42,
                 sampler: str = "tpe",
                 startup_trials: int = 4):
        super().__init__(evaluator, max_iterations, seed)
        self.sampler_name = sampler
        self.startup_trials = startup_trials

    async def run(self, stems: Dict[str, np.ndarray], sr: int) -> BaselineResult:
        try:
            import optuna
        except ImportError as ex:
            raise RuntimeError("optuna not installed; `pip install optuna`") from ex
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Initial state & section boundaries (sections unused for static BO,
        # but kept consistent with the agent for fair scoring).
        init_state = MixState.initial_from_stems(stems, sr)
        rough = render(init_state)
        boundaries = detect_sections(rough, sr)
        init_state = init_state.with_sections(boundaries)
        track_names = list(stems)

        initial_scores = await self.evaluator.evaluate(
            rough, sr,
            section_boundaries=boundaries,
            resolutions=["global", "section", "short_term", "momentary"],
        )

        # ---- Build study ----
        if self.sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(
                seed=self.seed, n_startup_trials=self.startup_trials)
        elif self.sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        else:
            raise ValueError(f"unknown sampler {self.sampler_name!r}")
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trials: List[TrialRecord] = []
        best_state: MixState = init_state
        best_scores = initial_scores
        best_reward = initial_scores.aggregate_reward()
        t0 = time.time()

        # Use a top-level event loop manually (Optuna's objective is sync).
        def objective(trial: "optuna.Trial") -> float:
            params = self._sample_params_from_trial(trial, track_names)
            try:
                cand_state = apply_action_set(init_state, params)
                cand_audio = render(cand_state)
            except Exception:                                  # noqa: BLE001
                return -1e6
            # safety: penalize hard
            if np.max(np.abs(cand_audio)) > 1.5:
                return -1e3
            scores = asyncio.run(self.evaluator.evaluate(
                cand_audio, sr,
                section_boundaries=boundaries,
                resolutions=["global"],
            ))
            flat = flatten_global_only(scores)
            reward = scores.aggregate_reward()
            # Record trial
            nonlocal best_state, best_scores, best_reward
            t_iter = TrialRecord(
                iteration=len(trials) + 1,
                reward=float(reward),
                flat_scores=flat,
                state_id=cand_state.state_id,
                action_summary=f"BO trial {len(trials)+1}",
                elapsed_sec=time.time() - t0,
            )
            trials.append(t_iter)
            if reward > best_reward:
                best_reward = float(reward)
                best_state = cand_state
                best_scores = scores
            return float(reward)

        # Run Optuna in a worker thread so the objective can safely call
        # `asyncio.run()` (we're already inside an event loop from the caller).
        await asyncio.to_thread(study.optimize,
                                objective,
                                n_trials=self.max_iterations,
                                show_progress_bar=False)

        # Final full multi-resolution evaluation on the BEST candidate
        # (the per-trial eval was global-only for speed)
        best_audio = render(best_state)
        best_scores_full = await self.evaluator.evaluate(
            best_audio, sr,
            section_boundaries=boundaries,
            resolutions=["global", "section", "short_term", "momentary"],
        )

        best_iter = int(np.argmax([t.reward for t in trials])) + 1 if trials else 0

        return BaselineResult(
            name=self.name,
            final_state=best_state,
            final_scores=best_scores_full,
            initial_scores=initial_scores,
            trials=trials,
            best_iteration=best_iter,
            best_reward=best_reward,
            total_cost_usd=0.0,
            elapsed_sec=time.time() - t0,
            extra={"sampler": self.sampler_name,
                   "n_trials": len(trials)},
        )

    # ---------- sampler ----------

    def _sample_params_from_trial(self, trial, track_names) -> FullActionSet:
        stems_p = {}
        for trk in track_names:
            p = StemParams()
            for k, (lo, hi) in STEM_PARAM_BOUNDS.items():
                v = trial.suggest_float(f"{trk}__{k}", lo, hi)
                setattr(p, k, v)
            stems_p[trk] = p
        master = MasterParams()
        for k, (lo, hi) in MASTER_PARAM_BOUNDS.items():
            setattr(master, k, trial.suggest_float(f"master__{k}", lo, hi))
        return FullActionSet(stems=stems_p, master=master)


def flatten_global_only(scores) -> Dict[str, float]:
    return dict(scores.global_scores)
