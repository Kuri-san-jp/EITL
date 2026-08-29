"""Hard-gate safety checks on a rendered master bus.

The orchestrator enforces these BEFORE any state is accepted. Violations
raise MixSafetyError so the LLM is forced to respond rather than silently
ship a broken mix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class MixSafetyError(RuntimeError):
    """Raised when a rendered mix violates a hard gate."""


@dataclass
class SafetyLimits:
    true_peak_db_max: float = -1.0        # ITU-R BS.1770 true peak ceiling
    lufs_target: float = -14.0
    lufs_tolerance: float = 6.0           # ±LU around target
    stereo_corr_min: float = -0.3         # phase correlation floor


def check_true_peak(audio: np.ndarray, limits: SafetyLimits) -> Optional[str]:
    """audio: (C, N) float32 in roughly [-1, 1]."""
    if audio.size == 0:
        return "empty audio"
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return None
    peak_db = 20.0 * np.log10(peak + 1e-12)
    if peak_db > limits.true_peak_db_max:
        return f"true peak {peak_db:.2f} dB > {limits.true_peak_db_max} dB"
    return None


def check_lufs(integrated_lufs: float, limits: SafetyLimits) -> Optional[str]:
    if not np.isfinite(integrated_lufs):
        return None  # silence — skip
    delta = abs(integrated_lufs - limits.lufs_target)
    if delta > limits.lufs_tolerance:
        return (f"integrated LUFS {integrated_lufs:.2f} outside "
                f"target {limits.lufs_target} ± {limits.lufs_tolerance}")
    return None


def check_stereo_correlation(audio: np.ndarray, limits: SafetyLimits) -> Optional[str]:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return None
    l, r = audio[0], audio[1]
    if np.std(l) < 1e-6 or np.std(r) < 1e-6:
        return None
    corr = float(np.corrcoef(l, r)[0, 1])
    if not np.isfinite(corr):
        return None
    if corr < limits.stereo_corr_min:
        return f"stereo correlation {corr:.3f} < {limits.stereo_corr_min}"
    return None
