"""Classical automix rules (Ward & Reiss / De Man style).

Implemented (Phase-1 minimum viable):
  R1. Equal-LUFS balance: trim each stem to match a target per-stem LUFS.
  R2. Spectral mask reduction: for the loudest band overlap, cut 2dB on
      the less-important stem (priority: vocal > drums > bass > others).
  R3. Frequency-based pan: bass/drums = center, mid-range stems = wider.
  R4. Master limiter at -1 dBTP.

This is intentionally simple. The cite-worthy version would implement
Ward's partial loudness model fully; that's a Phase-1.5 follow-up.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Tuple

import numpy as np

from ..dsp.effects import (StaticEQ, StaticGain, StaticPan, MasterLimiter)
from ..dsp.mix_state import MixState
from ..dsp.renderer import render
from ..resolution.multi_res_evaluator import MultiResolutionEvaluator
from ..resolution.section_detector import detect_sections
from .base import BaselineRunner, BaselineResult, TrialRecord


# Priority for mask-resolution (higher = wins fights)
TRACK_PRIORITY = {
    "vocal": 5, "vocals": 5, "lead": 5,
    "drums": 4, "kick": 4, "snare": 4,
    "bass":  3,
    "guitar":2, "keys": 2, "synth": 2,
}


def _priority(name: str) -> int:
    n = name.lower()
    for k, v in TRACK_PRIORITY.items():
        if k in n:
            return v
    return 1


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def _band_energy_db(audio: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    from numpy.fft import rfft, rfftfreq
    a = audio if audio.ndim == 1 else audio.mean(axis=0)
    spec = np.abs(rfft(a))**2
    freqs = rfftfreq(len(a), 1/sr)
    band = (freqs >= f_lo) & (freqs < f_hi)
    e = spec[band].sum()
    return 10 * np.log10(e + 1e-12)


class RuleBasedBaseline(BaselineRunner):
    name = "rule_based"

    def __init__(self, evaluator: MultiResolutionEvaluator,
                 max_iterations: int = 1,
                 seed: int = 42,
                 target_stem_rms_db: float = -20.0):
        super().__init__(evaluator, max_iterations, seed)
        self.target_stem_rms_db = target_stem_rms_db

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

        # R1: per-stem gain to target RMS
        state = init_state
        rms_per_stem: Dict[str, float] = {}
        for name, s in stems.items():
            mono = s.mean(axis=0) if s.ndim == 2 else s
            r = _rms_db(mono)
            rms_per_stem[name] = r
            delta = self.target_stem_rms_db - r
            delta = float(np.clip(delta, -12, 12))
            if abs(delta) > 0.5:
                state = state.add_static(name, StaticGain(gain_db=delta),
                                         summary=f"rule R1: gain {name} {delta:+.1f}dB")

        # R2: mask reduction in dominant overlap band (200-800 Hz)
        bands = [(80, 300), (300, 1000), (1000, 3000), (3000, 8000)]
        for f_lo, f_hi in bands:
            energies = []
            for n, s in stems.items():
                energies.append((n, _band_energy_db(s.mean(axis=0) if s.ndim == 2 else s,
                                                    sr, f_lo, f_hi)))
            energies.sort(key=lambda x: -x[1])
            if len(energies) < 2:
                continue
            top, runner_up = energies[0], energies[1]
            if top[1] - runner_up[1] < 6:  # genuine competition
                # cut 2dB on the LOWER priority track in this band
                loser = top[0] if _priority(top[0]) < _priority(runner_up[0]) else runner_up[0]
                f_mid = (f_lo * f_hi) ** 0.5
                state = state.add_static(
                    loser, StaticEQ(freq=f_mid, gain_db=-2.0, q=1.0),
                    summary=f"rule R2: mask-cut {loser} @ {f_mid:.0f}Hz",
                )

        # R3: frequency-based pan (only non-bass / non-vocal stems)
        for i, name in enumerate(stems):
            n = name.lower()
            if any(k in n for k in ("vocal", "bass", "kick", "snare")):
                continue
            pan = ((-1) ** i) * 0.25
            state = state.add_static(name, StaticPan(pan=pan),
                                     summary=f"rule R3: pan {name} {pan:+.2f}")

        # R4: master limiter
        state = state.add_master(MasterLimiter(ceiling_db=-1.0, release_ms=100.0),
                                 summary="rule R4: master limiter")

        # Render + evaluate
        cand_audio = render(state)
        scores = await self.evaluator.evaluate(
            cand_audio, sr, section_boundaries=boundaries,
            resolutions=["global", "section", "short_term", "momentary"],
        )
        reward = float(scores.aggregate_reward())

        trials = [TrialRecord(
            iteration=1, reward=reward,
            flat_scores=dict(scores.global_scores),
            state_id=state.state_id, action_summary="rule_based one-shot",
            elapsed_sec=time.time() - t0,
        )]
        return BaselineResult(
            name=self.name, final_state=state,
            final_scores=scores, initial_scores=initial_scores,
            trials=trials, best_iteration=1, best_reward=reward,
            elapsed_sec=time.time() - t0,
        )
