"""Uniform random search over the same action space — the BO sanity floor."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ..dsp.mix_state import MixState
from ..dsp.renderer import render
from ..resolution.multi_res_evaluator import MultiResolutionEvaluator
from ..resolution.section_detector import detect_sections
from .base import BaselineRunner, BaselineResult, TrialRecord
from ._action_space import sample_random, apply_action_set


class RandomSearchBaseline(BaselineRunner):
    name = "random_search"

    async def run(self, stems: Dict[str, np.ndarray], sr: int) -> BaselineResult:
        rng = np.random.default_rng(self.seed)
        init_state = MixState.initial_from_stems(stems, sr)
        rough = render(init_state)
        boundaries = detect_sections(rough, sr)
        init_state = init_state.with_sections(boundaries)

        initial_scores = await self.evaluator.evaluate(
            rough, sr, section_boundaries=boundaries,
            resolutions=["global", "section", "short_term", "momentary"],
        )

        trials: List[TrialRecord] = []
        best_state, best_scores = init_state, initial_scores
        best_reward = initial_scores.aggregate_reward()
        t0 = time.time()
        track_names = list(stems)

        for it in range(1, self.max_iterations + 1):
            actions = sample_random(rng, track_names)
            try:
                cand_state = apply_action_set(init_state, actions)
                cand_audio = render(cand_state)
            except Exception:                                  # noqa: BLE001
                continue
            if np.max(np.abs(cand_audio)) > 1.5:
                continue
            scores = await self.evaluator.evaluate(
                cand_audio, sr, section_boundaries=boundaries, resolutions=["global"],
            )
            reward = float(scores.aggregate_reward())
            trials.append(TrialRecord(
                iteration=it, reward=reward,
                flat_scores=dict(scores.global_scores),
                state_id=cand_state.state_id,
                action_summary=f"random trial {it}",
                elapsed_sec=time.time() - t0,
            ))
            if reward > best_reward:
                best_reward = reward
                best_state = cand_state
                best_scores = scores

        best_audio = render(best_state)
        best_scores_full = await self.evaluator.evaluate(
            best_audio, sr, section_boundaries=boundaries,
            resolutions=["global", "section", "short_term", "momentary"],
        )
        best_iter = int(np.argmax([t.reward for t in trials])) + 1 if trials else 0
        return BaselineResult(
            name=self.name, final_state=best_state,
            final_scores=best_scores_full, initial_scores=initial_scores,
            trials=trials, best_iteration=best_iter, best_reward=best_reward,
            elapsed_sec=time.time() - t0,
        )
