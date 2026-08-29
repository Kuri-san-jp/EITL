"""Window enumeration for multi-resolution evaluation."""
from __future__ import annotations

from typing import List, Tuple


def section_windows(boundaries: List[Tuple[float, str]],
                    end_sec: float) -> List[Tuple[Tuple[float, float], str]]:
    """Pair adjacent boundaries into (window, label)."""
    if not boundaries:
        return [((0.0, end_sec), "whole")]
    out = []
    sorted_b = sorted(boundaries, key=lambda x: x[0])
    for i, (t, label) in enumerate(sorted_b):
        t_next = sorted_b[i+1][0] if i+1 < len(sorted_b) else end_sec
        if t_next > t:
            out.append(((float(t), float(t_next)), label))
    return out


def short_term_windows(duration_sec: float, win_sec: float = 3.0, hop_sec: float = 1.0) -> List[Tuple[float, float]]:
    out = []
    t = 0.0
    while t + win_sec <= duration_sec + 1e-6:
        out.append((t, t + win_sec))
        t += hop_sec
    if not out and duration_sec > 0.5:
        out.append((0.0, duration_sec))
    return out
