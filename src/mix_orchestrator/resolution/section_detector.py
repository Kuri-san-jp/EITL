"""Music structure / section detection.

Phase 1: try MSAF / allinone if available, else a deterministic
energy-based segmenter. The LLM is told the boundaries may be wrong
and can override them via tool calls.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def detect_sections(audio: np.ndarray, sr: int,
                    n_sections: int = 6,
                    min_section_sec: float = 4.0) -> List[Tuple[float, str]]:
    """Return [(start_sec, label), ...]. Last boundary marks end_of_song implicitly."""
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    duration = len(mono) / sr
    if duration < 2 * min_section_sec:
        return [(0.0, "whole")]
    # Try MSAF first
    try:
        return _msaf_segment(mono, sr)
    except Exception:                                  # noqa: BLE001
        pass
    return _energy_segment(mono, sr, n_sections, min_section_sec)


def _msaf_segment(mono: np.ndarray, sr: int) -> List[Tuple[float, str]]:
    import msaf, tempfile, soundfile as sf, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, mono, sr)
        try:
            boundaries, labels = msaf.process(f.name, boundaries_id="sf", labels_id="fmc2d")
        finally:
            os.unlink(f.name)
    # MSAF returns the end as the last boundary; drop it
    out = []
    for i, t in enumerate(boundaries[:-1]):
        lab = f"section_{labels[i]}" if i < len(labels) else f"section_{i}"
        out.append((float(t), lab))
    return out


def _energy_segment(mono: np.ndarray, sr: int,
                    n_sections: int, min_sec: float) -> List[Tuple[float, str]]:
    """Naive RMS-change segmentation as a fallback."""
    hop = int(sr * 0.5)
    win = int(sr * 1.0)
    rms = []
    for i in range(0, len(mono) - win, hop):
        rms.append(np.sqrt(np.mean(mono[i:i+win]**2)))
    rms = np.array(rms)
    if len(rms) < n_sections * 2:
        return [(0.0, "whole")]
    # Find n-1 largest absolute first-differences as boundaries
    diff = np.abs(np.diff(rms))
    cand = np.argsort(diff)[::-1]
    times = []
    min_samples = int(min_sec / 0.5)
    for idx in cand:
        t = float(idx * 0.5 + 0.5)
        if all(abs(t - t2) >= min_sec for t2 in times):
            times.append(t)
        if len(times) >= n_sections - 1:
            break
    times = sorted(times)
    boundaries = [(0.0, "intro")]
    labels = ["verse_1", "chorus_1", "verse_2", "chorus_2", "bridge", "outro"]
    for i, t in enumerate(times):
        boundaries.append((t, labels[i] if i < len(labels) else f"section_{i+1}"))
    return boundaries
