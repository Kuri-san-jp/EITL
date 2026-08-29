"""Section segmentation based on musical structure (for P5, addressing review H-3/C-2).

The energy fallback in the old detect_sections merely took the top points of the
first difference of the RMS as boundaries, which is not musical structure, and its
labels (verse/chorus) were meaningless. This module applies librosa's agglomerative
clustering to MFCC+chroma features and returns musically more coherent boundaries.
It also works in environments without MSAF/all-in-one.

Policy:
  - Labels are **neutral and unique** (seg_0, seg_1, ...). They are not named
    verse/chorus (functional labels are not trustworthy without MSAF/allin1; this
    avoids making a false musical claim).
  - Return which method was used (its name) so it can be recorded in the
    experiment log.
  - Enforce a minimum section length (with the ~8-10 s stability floor of Audiobox
    and friends in mind).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def segment_structure(audio: np.ndarray, sr: int,
                      n_sections: int = 5,
                      min_section_sec: float = 8.0
                      ) -> Tuple[List[Tuple[float, str]], str]:
    """Return boundaries based on musical structure.

    Returns:
        (boundaries, method):
          boundaries = [(start_sec, unique_label), ...] (ascending by start, first is 0.0)
          method     = name of the segmentation method used (for logging)
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float32)
    dur = len(mono) / sr
    if dur < 2 * min_section_sec:
        return [(0.0, "seg_0")], "single_too_short"

    k_max = max(2, int(dur // min_section_sec))
    k = min(n_sections, k_max)

    try:
        import librosa
        hop = 2048
        mfcc = librosa.feature.mfcc(y=mono, sr=sr, hop_length=hop, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop)
        feat = np.vstack([
            librosa.util.normalize(mfcc, axis=1),
            librosa.util.normalize(chroma, axis=1),
        ])
        # agglomerative: returns the boundary frames that split feat into k segments
        bound_frames = librosa.segment.agglomerative(feat, k)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop)
        method = "agglomerative_mfcc_chroma"
    except Exception as ex:                              # noqa: BLE001
        # Last-resort fallback: uniform segmentation (neutral labels)
        bound_times = np.linspace(0.0, dur, k, endpoint=False)
        method = f"uniform_fallback({type(ex).__name__})"

    # Guarantee a 0.0 start, sort ascending, and drop duplicates
    times = sorted(set([0.0] + [float(t) for t in bound_times if 0.0 < t < dur]))

    # Enforce the minimum length: merge boundaries that are too close together
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] >= min_section_sec:
            merged.append(t)
    # If the last section is too short, drop the final boundary
    if len(merged) >= 2 and (dur - merged[-1]) < min_section_sec:
        merged.pop()

    boundaries = [(float(t), f"seg_{i}") for i, t in enumerate(merged)]
    return boundaries, method
