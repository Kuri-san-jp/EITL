"""MUSDB18-HQ loader (open-access multitrack dataset, Zenodo 3338373).

Directory layout (after extracting `musdb18hq.zip`):

    $MUSDB18_ROOT/
        train/
            <song_name>/
                mixture.wav
                drums.wav
                bass.wav
                vocals.wav
                other.wav
        test/
            <song_name>/
                mixture.wav
                drums.wav
                bass.wav
                vocals.wav
                other.wav

Each stem is stereo, 44.1 kHz, 16/24-bit PCM.

Public API (matches the rest of `mix_orchestrator.data.loaders`):

    stems, sr = load_musdb18_song("A Classic Education - NightOwl", split="test")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


# MUSDB18-HQ ships these 4 named stems + a `mixture` reference.
STEM_NAMES = ("drums", "bass", "vocals", "other")


def musdb18_root(root: Optional[str] = None) -> Path:
    p = root or os.environ.get("MUSDB18_ROOT")
    if not p:
        raise RuntimeError("MUSDB18_ROOT not set (and no `root` arg given)")
    return Path(p)


def list_songs(split: str = "test", root: Optional[str] = None) -> List[str]:
    base = musdb18_root(root) / split
    if not base.exists():
        raise FileNotFoundError(f"MUSDB18-HQ split dir not found: {base}")
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def load_musdb18_song(song_id: str,
                      root: Optional[str] = None,
                      split: str = "test",
                      sample_rate: int = 44100,
                      include_mixture: bool = False
                      ) -> Tuple[Dict[str, np.ndarray], int]:
    """Return (stems_dict, sr) for one song.

    - `stems_dict[name]` is shape (channels, samples), float32 in [-1, 1].
    - If `include_mixture=True`, the original engineer's mix is added under
      the key `"mixture"`. By default we omit it because the orchestrator
      builds its own mix from the 4 sources.
    """
    song_dir = musdb18_root(root) / split / song_id
    if not song_dir.exists():
        # Try the other split as a courtesy
        alt = musdb18_root(root) / ("train" if split == "test" else "test") / song_id
        if alt.exists():
            song_dir = alt
        else:
            raise FileNotFoundError(f"song dir not found: {song_dir}")

    stems: Dict[str, np.ndarray] = {}
    names = list(STEM_NAMES) + (["mixture"] if include_mixture else [])
    for stem_name in names:
        wav_path = song_dir / f"{stem_name}.wav"
        if not wav_path.exists():
            # Some MUSDB18 dumps store `other.wav` as `accompaniment.wav` —
            # don't crash, just note the absence.
            continue
        audio, sr_file = sf.read(str(wav_path), dtype="float32", always_2d=True)
        audio = audio.T            # (N, C) -> (C, N)
        if sr_file != sample_rate:
            audio = _resample(audio, sr_file, sample_rate)
        stems[stem_name] = audio.astype(np.float32)

    if not stems:
        raise RuntimeError(f"no stems readable under {song_dir}")
    return stems, sample_rate


def load_mixture(song_id: str, root: Optional[str] = None,
                 split: str = "test", sample_rate: int = 44100
                 ) -> Tuple[np.ndarray, int]:
    """Convenience: load just the engineer's reference mixture."""
    p = musdb18_root(root) / split / song_id / "mixture.wav"
    audio, sr_file = sf.read(str(p), dtype="float32", always_2d=True)
    audio = audio.T
    if sr_file != sample_rate:
        audio = _resample(audio, sr_file, sample_rate)
    return audio.astype(np.float32), sample_rate


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        import librosa
        return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out, axis=-1)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        up, down = sr_out // g, sr_in // g
        if x.ndim == 1:
            return resample_poly(x, up, down)
        return np.stack([resample_poly(c, up, down) for c in x])
