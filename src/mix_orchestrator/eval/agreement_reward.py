"""min agreement reward: R = min(z(Audiobox-PQ), z(SongBench-Mixing)).

Design (internal notes, by design):
  - The optimization reward is the **agreement** of two independent model
    metrics that can score a single mix: Audiobox-**PQ** (Production Quality)
    and SongBench-**Mixing**.
  - Only mixes that both models call good score highly = strong
    anti-reward-hacking.
  - z normalization is **per-song calibration**: for each song we score
    {dry + baseline(P2, P6) + a few initial random candidates} and use the
    mean/std of PQ and SB as that song's z reference.

This module is the pure computation "already-computed scores -> reward value"
(CPU). The actual scoring (the Audiobox / SongBench GPU forward passes) is done
by the caller, which passes in floats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class SongCalibration:
    """z normalization reference for one song (mean/std for PQ and for SB).

    The ``*_floored`` flags mean "the measured std fell below the floor, so the
    floor was used instead" = that metric's calibration pool has weak
    perturbations / too few samples.
    When the SB side is floored, the SB z is compressed and the anti-hack guard
    becomes dull, so the caller (runner) should emit a WARNING and strengthen
    the pool.

    **Measured (added , internal notes)**:
    the floor is not an exceptional fallback; it fires all the time. On the
    the reported runs (outputs/runs/the main random-proposer run, 100 songs x 2 seeds = 200
    seed-records), the floor determines the z scale for
    **PQ 47.5% (95/200) / SB 22.5% (45/200)** of the records. The old claim that
    "measured per-song SB-Mix std ~=0.12 >> floor 0.05, so the floor is a
    no-op" was a value under sb_sensitivity_probe conditions (strong
    perturbations of gain +/-8 dB / EQ +/-9 dB) and does not apply to the
    production runner's calibration pool.
    """
    pq_mean: float
    pq_std: float
    sb_mean: float
    sb_std: float
    n_samples: int = 0
    pq_std_measured: float = float("nan")
    sb_std_measured: float = float("nan")
    pq_floored: bool = False
    sb_floored: bool = False

    # Lower bound so that std=0 (all candidates identical / a single sample)
    # does not blow up the division.
    _STD_FLOOR: float = 1e-6

    def z_pq(self, pq: float) -> float:
        return (pq - self.pq_mean) / max(self.pq_std, self._STD_FLOOR)

    def z_sb(self, sb: float) -> float:
        return (sb - self.sb_mean) / max(self.sb_std, self._STD_FLOOR)

    @property
    def is_sb_guard_reliable(self) -> bool:
        """Whether SB can be trusted as an anti-hack guard.

        SB std raised by the floor = the pool is weak and z_SB is compressed
        = SB rarely attains the min and barely works as a guard. True when there
        are enough samples (``n_samples >= _MIN_RELIABLE_N``) and the floor did
        not fire.
        """
        return (not self.sb_floored) and self.n_samples >= _MIN_RELIABLE_N


# ---------------------------------------------------------------------------
# std floor (lower bound that prevents the noise amplification of z = delta/std)
#
# Rationale (internal notes):
#   An earlier smoke run (agreement_loop_smoke, 1 song, 6-item calib pool)
#   measured
#     PQ std = 0.0537, SB-Mixing std = 0.0202
#   SB-Mixing is the tanh*4.5+5.5 output of the Generator on MuQ hidden[6].
#   With z = delta/std, an std around 0.02 means a 0.05 move in SB already gives
#   |z|>2, and reward = min(z_PQ, z_SB) becomes dominated by SB-side noise
#   (numerical jitter in the Generator, tiny variations from resampling). That
#   destroys the meaning of "agreement".
#
#   The floor is set as "the minimum effective scale this metric should have
#   against perturbations". Floor 0.05 for both PQ and SB (put on the same scale
#   to suppress z explosions caused by a tiny std).
#   Taking the floor larger than the std compresses z, but for min agreement we
#   judged the side effect of "one metric thrashing on noise and unfairly
#   crushing the other" to be more harmful.
#
# ---------------------------------------------------------------------------
# [Correction from  measurements] internal notes
#
#   (1) **The floor is not "almost a no-op".** The old comment said "the
#       measured PQ std ~0.05 is the low end of the typical range, so
#       floor_pq=0.05 is about the same as the measured value = almost a no-op",
#       but that reasoning is backwards. Putting the floor **near the median**
#       of the distribution makes it fire on about half the cases by definition.
#       The measured firing rate on the the reported runs (the main random-proposer run, 200
#       seed-records) is **PQ 47.5% / SB 22.5%**, and the median stored std is
#       PQ 0.0515 / SB 0.0681. It is a routine code path, not "insurance for
#       degenerate cases".
#
#   (2) **"SB-Mixing responds extremely weakly to perturbations" does not hold
#       under production conditions.** Under the production action distribution
#       (agreement_loop_v2._propose_action: gain U(-4,+4) dB / EQ U(-5,+5) dB),
#       measuring the raw std over the initial mix + 28 search candidates = 29
#       mixes gives PQ median 0.0637 / SB median 0.0817 (ratio SB/PQ = 1.28), so
#       **SB is if anything the more sensitive one**. The basis of the old claim
#       (SB std 0.0202 on a 6-item smoke pool for 1 song) is exactly the
#       condition the internal notes themselves declared "an artifact of the
#       +/-4 dB perturbation", and that same condition is what production
#       calibration runs under.
#
#   (3) **The root cause is that the calibration pool's perturbations are weak.**
#       agreement_loop_all.py:508 builds the pool with the same _propose_action
#       as the search, so it does not meet the premise of sb_sensitivity_probe
#       (gain +/-8 dB / EQ +/-9 dB, per-song SB std ~=0.12). Changing the floor
#       value 0.05 would make comparison with past runs impossible, so
#       **do not change the value**. The fix belongs on the pool side (adding
#       strongly-perturbed candidates).
#
#   (4) **On existing runs the measured std at the time the floor fired cannot
#       be recovered.** fit_calibration returns pq_std_measured /
#       sb_std_measured, but the runner's serialization
#       (agreement_loop_all.py:451-457 and :727-732) only writes out
#       pq_mean/pq_std/sb_mean/sb_std/n_samples/*_floored and discards them.
#       Future runs should also store *_std_measured (and, if possible, the raw
#       score series of the calib pool).
#
# Backward compatibility: the existing SongCalibration._STD_FLOOR=1e-6
#   (divide-by-zero protection inside z()) is kept. The std returned by
#   fit_calibration is already max(measured, floor), so the floor inside z()
#   effectively never fires (double protection). Passing floor=0 explicitly
#   restores the old behaviour (measured std).
_FLOOR_PQ = 0.05
_FLOOR_SB = 0.05

# Rule of thumb: at or above this pool size the std estimate is stable and the
# SB guard is considered trustworthy. The measurement that gave per-song SB-Mix
# std mean 0.12 in sb_sensitivity_probe (musdb18, 8 songs x 19 perturbations)
# used a 20-mix pool including the baseline. We recommend a threshold of at
# least 12 items containing sufficient perturbation (below that, the measured
# std falls under the floor and SB goes dull).
#
# Note : this threshold looks **only at the count**, not at the
# **magnitude** of the perturbations. Production runs always satisfy it with
# n_samples=13 (200/200 records), yet the floor still fires for PQ 47.5% /
# SB 22.5% because the pool's perturbations are weak. Meeting the count
# condition does not mean "calibration is healthy".
_MIN_RELIABLE_N = 12


def fit_calibration(pq_samples: Sequence[float],
                    sb_samples: Sequence[float],
                    floor_pq: float = _FLOOR_PQ,
                    floor_sb: float = _FLOOR_SB) -> SongCalibration:
    """Estimate mean/std from the calibration pool (dry + baseline + initial
    random candidates).

    The floor is a safety net against the **explosion** of z=delta/std. The
    floor firing (measured < floor) is a sign that "the pool's perturbations are
    weak / there are too few samples", and we tell the caller through
    SongCalibration.{pq,sb}_floored=True.

    **Condition under which the floor is a no-op (corrected **: only
    when the pool contains perturbations as large as sb_sensitivity_probe
    (gain +/-8 dB / EQ +/-9 dB) does per-song SB-Mix std ~=0.12 >> floor 0.05
    hold. The production runner (agreement_loop_all.py:508) builds the pool with
    the same _propose_action as the search (gain +/-4 dB / EQ +/-5 dB), so it
    **does not meet that premise**. The measured firing rate is
    PQ 47.5% / SB 22.5% (the main random-proposer run, 200 seed-records).
    We therefore retract the old claim that "not firing is normal for a
    discriminative pool".

    Args:
        pq_samples: the Audiobox-PQ score series for that song.
        sb_samples: the SongBench-Mixing score series for that song (same
            candidate set).
        floor_pq: lower bound on PQ std. Prevents the noise amplification of
                  z=delta/std (default 0.05).
        floor_sb: lower bound on SB-Mixing std (default 0.05). When the
                  perturbations are weak the measured SB-Mixing std gets small
                  and z explodes, so the floor raises it.
                  Passing floor=0 gives the old behaviour (measured std as-is).
    """
    pq = np.asarray(list(pq_samples), dtype=np.float64)
    sb = np.asarray(list(sb_samples), dtype=np.float64)
    if pq.size == 0 or sb.size == 0:
        raise ValueError("calibration needs at least one PQ and one SB sample.")
    n = int(min(pq.size, sb.size))
    # ddof=0 (population std). With a single sample std=0, protected by the floor.
    pq_std_m = float(pq.std(ddof=0))
    sb_std_m = float(sb.std(ddof=0))
    pq_std = max(pq_std_m, float(floor_pq))
    sb_std = max(sb_std_m, float(floor_sb))
    return SongCalibration(
        pq_mean=float(pq.mean()), pq_std=pq_std,
        sb_mean=float(sb.mean()), sb_std=sb_std,
        n_samples=n,
        pq_std_measured=pq_std_m, sb_std_measured=sb_std_m,
        pq_floored=bool(pq_std_m < float(floor_pq)),
        sb_floored=bool(sb_std_m < float(floor_sb)),
    )


def warn_if_weak_calibration(calib: SongCalibration) -> List[str]:
    """Return WARNING strings when the calibration is weak (floor fired /
    insufficient samples).

    Prints them if the return value is non-empty. A hook for the runner to judge
    the reliability of the SB guard (is_sb_guard_reliable). Never raises (does
    not stop the experiment).
    """
    msgs: List[str] = []
    if calib.n_samples < _MIN_RELIABLE_N:
        msgs.append(
            f"[agreement] WARN: calib n={calib.n_samples} < {_MIN_RELIABLE_N}; "
            f"the std estimate is unstable and the SB guard may go dull. Enlarge the pool (perturbed candidates).")
    if calib.sb_floored:
        msgs.append(
            f"[agreement] WARN: SB std measured={calib.sb_std_measured:.4f} < "
            f"floor; the SB z is compressed and the anti-hack guard goes dull. Include "
            f"larger mixing perturbations in the calib pool (measured SB-Mix std~=0.12 is the target).")
    if calib.pq_floored:
        msgs.append(
            f"[agreement] WARN: PQ std measured={calib.pq_std_measured:.4f} < floor; "
            f"the PQ pool is weak (possibly a degenerate case).")
    for m in msgs:
        print(m)
    return msgs


def compute_agreement(pq: float, sb: float, calib: SongCalibration) -> float:
    """Return R = min(z(PQ), z(SB)) (the default reward).

    Design: SB is an anti-hack guard that uses min to **pull down**
    "reward-hacking candidates that are only high on PQ". min rates highly only
    candidates that are high on both z_PQ and z_SB.
    Even when the floor compresses the SB z, the noise explosion in which SB
    unfairly crushes PQ is prevented (a weak pool is detected via
    calib.sb_floored).

    **Correction **: the old comment's claim that "SB never pushes
    the reward up on its own (never decides the direction)" is wrong. When SB is
    the binding axis (z_SB < z_PQ), raising SB is the only way to raise the
    reward, so SB does decide the direction. Measured
    (the main random-proposer run, 5600 proposals): of the candidates that pq_only would
    accept in the same state, **SB vetoes 39.0% (752/1928)**, and conversely of
    the 1458 accepted by min, **19.3% (282) would be rejected by pq_only** = SB
    is driving acceptance. SB is the binding axis on 37.9% (552/1458) of the
    accepted steps.
    """
    return float(min(calib.z_pq(pq), calib.z_sb(sb)))


# ------------------------------------------------------------------
# ablation reward forms (matching the design in the note)
# ------------------------------------------------------------------
def compute_penalized(pq: float, sb: float, calib: SongCalibration,
                      lam: float = 0.5) -> float:
    """R = z(PQ) - lam * |z(PQ) - z(SB)|  (disagreement-penalty variant).

    **Warning **: this form is **not** a family that interpolates
    between mean and min. The penalty is asymmetric, anchored on PQ, so
    comparing it against the identity
    ``min(a,b) = (a+b)/2 - |a-b|/2`` gives

        lam=0   -> R = z_PQ                       (= pq_only, not mean)
        lam=0.5 -> mean for a>b, mean-|delta| for a<b (over-penalized vs min)
        lam=1   -> min for a>b, 2a-b for a<b      (does not match min)

    so **no lam makes it min** (it only agrees on the half-plane z_PQ>z_SB, and
    only at lam=1). If you need a symmetric family that correctly connects
    mean(lam=0) and min(lam=0.5), use

        R = (z_PQ + z_SB)/2 - lam * |z_PQ - z_SB|

    This function is currently not used by any run (also recorded as unused in
    paper_supplementary/reproducibility.md).
    """
    zp, zs = calib.z_pq(pq), calib.z_sb(sb)
    return float(zp - lam * abs(zp - zs))


def compute_weighted(pq: float, sb: float, calib: SongCalibration,
                     w: float = 0.5) -> float:
    """R = w * z(PQ) + (1 - w) * z(SB)  (weighted-average variant)."""
    return float(w * calib.z_pq(pq) + (1.0 - w) * calib.z_sb(sb))


@dataclass
class AgreementReward:
    """High-level API bundling the scoring functions.

    pq_scorer / sb_scorer are (audio: ndarray, sr: int) -> float.
    Example:
        from mix_orchestrator.ears.tier6_music_reward import SongBenchMixingEar
        from mix_orchestrator.eval.audiobox_pq import AudioboxPQScorer
        sb = SongBenchMixingEar()
        pq = AudioboxPQScorer()
        reward = AgreementReward(pq_scorer=pq.score, sb_scorer=sb.score)
        calib = reward.calibrate(calib_audios, sr)
        r = reward.reward(mix_audio, sr, calib)
    """
    pq_scorer: Callable[[np.ndarray, int], float]
    sb_scorer: Callable[[np.ndarray, int], float]

    def score_pair(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Score and return (PQ, SB)."""
        return float(self.pq_scorer(audio, sr)), float(self.sb_scorer(audio, sr))

    def calibrate(self, calib_audios: List[np.ndarray], sr: int,
                  warn: bool = True) -> SongCalibration:
        """per-song calibration: score dry + baseline + initial candidates and
        build the z reference.

        When ``warn=True``, emits a WARNING if the SB std was raised by the
        floor (= the pool is weak and SB goes dull as an anti-hack guard).
        Including sufficiently large perturbations in the pool is a
        precondition for the SB guard (internal notes).
        """
        pq_samples, sb_samples = [], []
        for a in calib_audios:
            p, s = self.score_pair(a, sr)
            pq_samples.append(p)
            sb_samples.append(s)
        calib = fit_calibration(pq_samples, sb_samples)
        if warn:
            warn_if_weak_calibration(calib)
        return calib

    def reward(self, audio: np.ndarray, sr: int, calib: SongCalibration,
               form: str = "min",
               precomputed: Optional[tuple[float, float]] = None) -> float:
        """Reward for one mix candidate. Pass precomputed=(pq, sb) to skip rescoring."""
        if precomputed is not None:
            pq, sb = precomputed
        else:
            pq, sb = self.score_pair(audio, sr)
        if form == "min":
            return compute_agreement(pq, sb, calib)
        if form == "penalized":
            return compute_penalized(pq, sb, calib)
        if form == "weighted":
            return compute_weighted(pq, sb, calib)
        raise ValueError(f"unknown reward form: {form!r}")
