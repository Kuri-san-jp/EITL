"""Top-level stem loader dispatching on dataset name."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .datasets.synthetic import synthetic_stems


def load_stems(name: str, **kwargs) -> Tuple[Dict[str, np.ndarray], int]:
    """Return (stems_dict, sample_rate)."""
    name = name.lower()
    if name == "synthetic":
        sr = int(kwargs.get("sample_rate", 44100))
        dur = float(kwargs.get("duration_sec", 6.0))
        seed = int(kwargs.get("seed", 0))
        return synthetic_stems(duration_sec=dur, sr=sr, seed=seed), sr
    if name == "medleydb":
        from .datasets.medleydb import load_medleydb_song
        return load_medleydb_song(**kwargs)
    if name in ("musdb18", "musdb18hq", "musdb18-hq", "musdb"):
        from .datasets.musdb18 import load_musdb18_song
        return load_musdb18_song(**kwargs)
    if name in ("cambridge_mt", "cambridge-mt", "cambridge"):
        from .datasets.cambridge_mt import load_cambridge_mt_song
        return load_cambridge_mt_song(**kwargs)
    raise ValueError(f"unknown dataset {name!r}")


def list_songs(name: str, **kwargs) -> list[str]:
    """Enumerate songs in a dataset (where supported)."""
    name = name.lower()
    if name in ("musdb18", "musdb18hq", "musdb18-hq", "musdb"):
        from .datasets.musdb18 import list_songs as _ls
        return _ls(**kwargs)
    if name == "medleydb":
        from .datasets.medleydb import list_songs as _ls
        return _ls(**kwargs)
    if name in ("cambridge_mt", "cambridge-mt", "cambridge"):
        from .datasets.cambridge_mt import list_songs as _ls
        return _ls(**kwargs)
    raise ValueError(f"list_songs not supported for {name!r}")


def load_section_boundaries(name: str, song_id: str, **kwargs):
    """Return [(start_sec, label), ...] from a dataset's annotations."""
    name = name.lower()
    if name == "medleydb":
        from .datasets.medleydb import load_section_boundaries as _ls
        return _ls(song_id, **kwargs)
    return []
