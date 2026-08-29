"""Target-profile harvesting (interpretation B): extract a recoverable target mix
profile (relative balance / spectrum / stereo / dynamics) from wet (already
mixed) audio, and at inference time match the dry stems to that target
(user proposal .

The profile consists of relative quantities that do not depend on dry/wet.
Dependencies are numpy + pyloudnorm (CPU).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _measure_lufs(audio: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln
    x = audio.T if audio.ndim == 2 else audio
    try:
        return float(pyln.Meter(sr).integrated_loudness(x))
    except Exception:                                          # noqa: BLE001
        rms = float(np.sqrt(np.mean(np.asarray(audio, float) ** 2) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12)


def _band_energies_db(mono: np.ndarray, sr: int) -> Dict[str, float]:
    spec = np.abs(np.fft.rfft(mono.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    tot = float((spec ** 2).sum()) + 1e-12
    out = {}
    for name, lo, hi in [("low", 0, 250), ("mid", 250, 4000), ("high", 4000, sr / 2)]:
        m = (freqs >= lo) & (freqs < hi)
        out[name] = round(10.0 * np.log10(float((spec[m] ** 2).sum()) / tot + 1e-12), 2)
    return out


def _stereo_corr(audio: np.ndarray) -> float:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return 1.0
    l, r = audio[0], audio[1]
    if np.std(l) < 1e-7 or np.std(r) < 1e-7:
        return 1.0
    c = float(np.corrcoef(l, r)[0, 1])
    return c if np.isfinite(c) else 1.0


def _ms_ratio_db(audio: np.ndarray) -> float:
    if audio.ndim != 2 or audio.shape[0] != 2:
        return -60.0
    l, r = audio[0], audio[1]
    s = 0.5 * (l - r); m = 0.5 * (l + r)
    rms_s = float(np.sqrt(np.mean(s ** 2) + 1e-12))
    rms_m = float(np.sqrt(np.mean(m ** 2) + 1e-12))
    return 20.0 * np.log10((rms_s + 1e-12) / (rms_m + 1e-12))


def _lra(mono: np.ndarray, sr: int, win: float = 3.0) -> float:
    n = int(win * sr)
    if n <= 0 or len(mono) < n:
        return 0.0
    vals = []
    for i in range(0, len(mono) - n, n):
        seg = mono[i:i + n]
        vals.append(20.0 * np.log10(np.sqrt(np.mean(seg ** 2) + 1e-12) + 1e-12))
    if len(vals) < 2:
        return 0.0
    return float(np.percentile(vals, 95) - np.percentile(vals, 10))


def extract_profile_from_stems(stems: Dict[str, np.ndarray], sr: int, role_of=None) -> Dict[str, Any]:
    """Extract the target profile from ground-truth/separated stems (relative to mix = sum(stems))."""
    from .analysis_driven import role_of_combined
    main = {k: v for k, v in stems.items() if k != "mixture"}
    mix = np.sum([np.asarray(v, np.float32) for v in main.values()], axis=0)
    mono_mix = mix.mean(axis=0) if mix.ndim == 2 else mix
    mix_lufs = _measure_lufs(mix, sr)
    role_audio: Dict[str, np.ndarray] = {}
    for name, v in main.items():
        v = np.asarray(v, np.float32)
        r = role_of(name) if role_of else role_of_combined(name, v, sr)
        role_audio[r] = v if r not in role_audio else role_audio[r] + v
    rel_loud = {r: round(_measure_lufs(a, sr) - mix_lufs, 2) for r, a in role_audio.items()}
    role_width = {r: round(_ms_ratio_db(a), 2) for r, a in role_audio.items()}
    return {
        "relative_loudness_db": rel_loud,
        "spectral_balance_db": _band_energies_db(mono_mix, sr),
        "stereo": {"correlation": round(_stereo_corr(mix), 3), "role_width_db": role_width},
        "dynamics": {"lra": round(_lra(mono_mix, sr), 2)},
        "roles": sorted(role_audio.keys()),
    }


def match_profile_gains(dry_stems: Dict[str, np.ndarray], sr: int,
                        target_profile: Dict[str, Any], role_of=None) -> Dict[str, float]:
    """Inference: per-stem gain (dB) that matches the dry stems to the target's relative_loudness."""
    from .analysis_driven import role_of_combined
    main = {k: v for k, v in dry_stems.items() if k != "mixture"}
    mix = np.sum([np.asarray(v, np.float32) for v in main.values()], axis=0)
    mix_lufs = _measure_lufs(mix, sr)
    tgt = target_profile.get("relative_loudness_db", {})
    gains: Dict[str, float] = {}
    for name, v in main.items():
        v = np.asarray(v, np.float32)
        r = role_of(name) if role_of else role_of_combined(name, v, sr)
        cur_rel = _measure_lufs(v, sr) - mix_lufs
        if r in tgt:
            gains[name] = round(float(np.clip(tgt[r] - cur_rel, -12.0, 12.0)), 2)
    return gains


def build_profile_matched_mix(dry_stems: Dict[str, np.ndarray], sr: int,
                              target_library, genre=None, roles=None, k: int = 1,
                              role_of=None):
    """P6: RAG-retrieve a target profile -> base mix that matches the dry stems to the target balance.

    Returns (mix_audio (C,L), meta) or (None, meta) if retrieval fails.
    target_library = ExperienceLibrary (contains target_profile cases).
    """
    from ..dsp.mix_state import MixState
    from ..dsp.renderer import render
    from ..tools import action_tools as A
    from .analysis_driven import _role_of as default_role_of
    role_of = role_of or default_role_of

    main = {k_: v for k_, v in dry_stems.items() if k_ != "mixture"}
    if roles is None:
        roles = sorted({role_of(n) for n in main})
    res = target_library.retrieve(genre=genre, stem_roles=roles, k=k)
    res = [r for r in res
           if r["case"].get("method", {}).get("strategy") == "target_profile"]
    if not res:
        return None, {"error": "no_target_profile_retrieved", "genre": genre, "roles": roles}
    case = res[0]["case"]
    target = case["method"]["target"]
    gains = match_profile_gains(dry_stems, sr, target, role_of=role_of)
    state = MixState.initial_from_stems(main, sr)
    for name, g in gains.items():
        if abs(g) > 1e-6:
            state = A.apply_static_gain(state, {"track": name, "gain_db": g})
    return render(state), {"target_case": case.get("case_id"),
                           "similarity": res[0].get("similarity"),
                           "gains": gains, "genre": genre, "roles": roles}


def profile_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ra, rb = a.get("relative_loudness_db", {}), b.get("relative_loudness_db", {})
    keys = set(ra) & set(rb)
    d_loud = np.mean([abs(ra[k] - rb[k]) for k in keys]) if keys else float("nan")
    sa, sb = a.get("spectral_balance_db", {}), b.get("spectral_balance_db", {})
    sk = set(sa) & set(sb)
    d_spec = np.mean([abs(sa[k] - sb[k]) for k in sk]) if sk else float("nan")
    return float(np.nanmean([d_loud, d_spec]))
