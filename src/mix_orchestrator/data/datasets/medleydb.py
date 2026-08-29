"""MedleyDB V1/V2 loader.

Expected on-disk layout (after `download_medleydb.sh`):

    $MEDLEYDB_ROOT/
        Audio/                          # V1
            <song>/
                <song>_MIX.wav
                <song>_STEMS/
                    <song>_STEM_NN_Inst.wav
                <song>_RAW/
                    <song>_RAW_NN_M.wav
        AudioV2/                        # V2 (optional)
            <song>/
                ... same triple ...
        annotations/                    # cloned from github.com/marl/medleydb
            Annotations/
                Instrument_Activations/
                Sections/<song>_SECTIONS.lab
                Melody_Annotations/
                ...

Section .lab format (one line per boundary):
    <start_sec>\t<end_sec>\t<label>
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


def medleydb_root(root: Optional[str] = None) -> Path:
    p = root or os.environ.get("MEDLEYDB_ROOT")
    if not p:
        raise RuntimeError("MEDLEYDB_ROOT not set (and no `root` arg given)")
    return Path(p)


def list_songs(root: Optional[str] = None,
               include_v2: bool = True) -> List[str]:
    base = medleydb_root(root)
    songs = set()
    for d in (base / "Audio", base / "AudioV2"):
        if d.exists():
            for p in d.iterdir():
                if p.is_dir() and not p.name.startswith("._"):
                    songs.add(p.name)
        if not include_v2:
            break
    return sorted(songs)


def load_medleydb_song(song_id: str,
                       root: Optional[str] = None,
                       sample_rate: int = 44100,
                       prefer_stems: str = "STEMS",     # or "RAW"
                       ) -> Tuple[Dict[str, np.ndarray], int]:
    base = medleydb_root(root)
    song_dir = _find_song_dir(base, song_id)
    if song_dir is None:
        raise FileNotFoundError(f"song {song_id!r} not found under {base}/Audio or AudioV2")

    stems_dir = song_dir / f"{song_id}_{prefer_stems}"
    if not stems_dir.exists():
        # fall back to the other type
        other = "RAW" if prefer_stems == "STEMS" else "STEMS"
        stems_dir = song_dir / f"{song_id}_{other}"
    if not stems_dir.exists():
        raise FileNotFoundError(f"no STEMS/RAW dir under {song_dir}")

    stems: Dict[str, np.ndarray] = {}
    for wav in sorted(stems_dir.glob("*.wav")):
        if wav.name.startswith("._"):           # macOS resource fork — skip
            continue
        name = _track_name_from(wav.stem)
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        audio = audio.T
        if sr != sample_rate:
            audio = _resample(audio, sr, sample_rate)
        stems[name] = audio.astype(np.float32)

    if not stems:
        raise RuntimeError(f"no stems readable under {stems_dir}")
    return stems, sample_rate


def load_section_boundaries(song_id: str,
                            root: Optional[str] = None,
                            ) -> List[Tuple[float, str]]:
    """Parse the song's SECTIONS.lab if available.

    Returns [(start_sec, label), ...] sorted by start_sec. Empty list if
    no annotation file is found (callers should then fall back to
    `resolution.section_detector.detect_sections`).
    """
    base = medleydb_root(root)
    candidates = [
        base / "annotations" / "Annotations" / "Sections" / f"{song_id}_SECTIONS.lab",
        base / "annotations" / "Sections" / f"{song_id}_SECTIONS.lab",
        base / "Sections" / f"{song_id}_SECTIONS.lab",
    ]
    lab = next((p for p in candidates if p.exists()), None)
    if lab is None:
        return []
    out: List[Tuple[float, str]] = []
    for line in lab.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t0 = float(parts[0])
        except ValueError:
            continue
        label = parts[-1] if len(parts) >= 3 else f"section_{len(out)}"
        out.append((t0, label))
    out.sort(key=lambda x: x[0])
    return out


def load_instrument_map(song_id: str,
                        root: Optional[str] = None,
                        prefer_stems: str = "STEMS"
                        ) -> Dict[str, str]:
    """Return {track_name: instrument} from <song>_METADATA.yaml.

    `track_name` matches the key produced by `_track_name_from(stem)`
    so the result can be looked up against load_medleydb_song()'s
    returned stems dict.
    """
    import yaml
    base = medleydb_root(root)
    song_dir = _find_song_dir(base, song_id)
    if song_dir is None:
        return {}
    meta_path = song_dir / f"{song_id}_METADATA.yaml"
    if not meta_path.exists():
        return {}
    try:
        meta = yaml.safe_load(meta_path.read_text("utf-8"))
    except Exception:                                                    # noqa: BLE001
        return {}
    out: Dict[str, str] = {}
    stems = (meta or {}).get("stems", {}) or {}
    use_raw = (prefer_stems.upper() == "RAW")
    for s_key, s in stems.items():
        # STEMS branch
        if not use_raw:
            fname = s.get("filename")
            inst  = s.get("instrument")
            if fname and inst:
                track_name = _track_name_from(Path(fname).stem)
                out[track_name] = inst
            continue
        # RAW branch
        raw = s.get("raw") or {}
        for r_key, r in raw.items():
            fname = r.get("filename"); inst = r.get("instrument")
            if fname and inst:
                track_name = _track_name_from(Path(fname).stem)
                out[track_name] = inst
    return out


def load_reference_mix(song_id: str,
                       root: Optional[str] = None,
                       sample_rate: int = 44100,
                       ) -> Tuple[np.ndarray, int]:
    base = medleydb_root(root)
    song_dir = _find_song_dir(base, song_id)
    if song_dir is None:
        raise FileNotFoundError(f"song {song_id!r} not found")
    mix = song_dir / f"{song_id}_MIX.wav"
    audio, sr = sf.read(str(mix), dtype="float32", always_2d=True)
    audio = audio.T
    if sr != sample_rate:
        audio = _resample(audio, sr, sample_rate)
    return audio.astype(np.float32), sample_rate


# ---------- helpers ----------


def _find_song_dir(base: Path, song_id: str) -> Optional[Path]:
    for sub in ("Audio", "AudioV2"):
        d = base / sub / song_id
        if d.exists():
            return d
    return None


def _track_name_from(stem: str) -> str:
    """`MusicDelta_Beatles_STEM_03_VocalsMale` -> `VocalsMale`."""
    parts = stem.split("_")
    for marker in ("STEM", "RAW"):
        if marker in parts:
            i = parts.index(marker)
            tail = "_".join(parts[i + 2:])
            return tail or stem
    return stem


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
