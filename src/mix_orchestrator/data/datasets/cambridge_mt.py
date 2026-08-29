"""Cambridge MT (Mike Senior's Mixing Secrets) loader.

The "Cambridge-MT" multitrack library is hosted at
https://cambridge-mt.com/ms-mtk.htm. Each song is distributed as a ZIP
of WAV/AIFF stems. Layout on disk (after extraction):

    $CAMBRIDGE_MT_ROOT/
        <song_id>/
            *.wav   (or *.flac, *.aif)
            *.txt   (Mike's per-track note, optional)

We use the directory name as `song_id` and the file stem as the
track / instrument name. Where the file name follows the typical
Cambridge-MT convention (`01_Vocals_LeadFemale.wav`,
`02_Drums_KickIn.wav`, ...), we strip the leading index and category
to produce a friendly short name and infer the instrument tag.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


_AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff")


def cambridge_root(root: Optional[str] = None) -> Path:
    p = root or os.environ.get("CAMBRIDGE_MT_ROOT")
    if not p:
        raise RuntimeError("CAMBRIDGE_MT_ROOT not set (and no `root` arg given)")
    return Path(p)


def list_songs(root: Optional[str] = None) -> List[str]:
    base = cambridge_root(root)
    if not base.exists():
        raise FileNotFoundError(f"Cambridge-MT root not found: {base}")
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(("._", "."))
    )


# Track-name normaliser ------------------------------------------------


_CATEGORY_KEYWORDS = {
    "vocal":   ("vocal", "voc", "lead", "sing", "rap", "voice"),
    "drums":   ("drum", "kick", "snare", "hihat", "hat", "tom", "ride",
                 "cymbal", "overhead", "perc"),
    "bass":    ("bass", "subbass"),
    "guitar":  ("guitar", "gtr", "gtrs"),
    "keys":    ("keys", "piano", "synth", "rhodes", "organ", "pad"),
    "strings": ("strings", "violin", "viola", "cello"),
    "horns":   ("horn", "trumpet", "sax", "trombone"),
    "fx":      ("fx", "sfx", "ambient", "noise"),
}


def _normalise_track_name(stem: str) -> Tuple[str, str]:
    """Return (short_name, instrument_tag).

    For `"01_Vocals_LeadFemale"` -> (`Vocals_LeadFemale`, `vocal`).
    """
    name = re.sub(r"^\d+[\s_\-]*", "", stem).strip("_- ")
    if not name:
        name = stem
    lower = name.lower()
    inst = "other"
    for tag, keys in _CATEGORY_KEYWORDS.items():
        if any(k in lower for k in keys):
            inst = tag
            break
    return name, inst


def load_cambridge_mt_song(song_id: str,
                           root: Optional[str] = None,
                           sample_rate: int = 44100,
                           ) -> Tuple[Dict[str, np.ndarray], int]:
    base = cambridge_root(root)
    song_dir = base / song_id
    if not song_dir.exists():
        raise FileNotFoundError(f"song dir not found: {song_dir}")

    stems: Dict[str, np.ndarray] = {}
    for f in sorted(song_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _AUDIO_EXTS:
            continue
        if f.name.startswith("._"):
            continue
        audio, sr = sf.read(str(f), dtype="float32", always_2d=True)
        audio = audio.T
        if sr != sample_rate:
            audio = _resample(audio, sr, sample_rate)
        track, _inst = _normalise_track_name(f.stem)
        # Avoid overwriting if two files normalise to the same name
        if track in stems:
            track = f.stem
        stems[track] = audio.astype(np.float32)
    if not stems:
        raise RuntimeError(f"no audio under {song_dir}")
    return stems, sample_rate


def load_instrument_map(song_id: str,
                        root: Optional[str] = None) -> Dict[str, str]:
    """Infer per-stem instrument tags from file names."""
    base = cambridge_root(root)
    song_dir = base / song_id
    if not song_dir.exists():
        return {}
    out: Dict[str, str] = {}
    for f in sorted(song_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _AUDIO_EXTS:
            continue
        if f.name.startswith("._"):
            continue
        track, inst = _normalise_track_name(f.stem)
        out[track] = inst
    return out


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
