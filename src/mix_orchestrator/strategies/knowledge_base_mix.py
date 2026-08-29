"""Knowledge-based initial mix generator.

Applies human mixing tips (HPF / complementary EQ / loudness pyramid) as rules
to build the **starting point (initial mix)** for the LLM's trial and error.
It does not optimise the reward directly; it is assembled deterministically
from mixing convention alone (avoiding metric-gaming; cf. analysis_driven P2).

Input : dry stems {name: ndarray(C, N)}, sr.
Output: master mix (via render) + per-stem log of the LUFS/EQ/gain decisions.

Design notes:
  - Only combines existing DSP actions (apply_static_eq / apply_static_gain).
    **No new DSP is written.** Since there is no dedicated HPF effect, the
    roll-off is approximated by stacking 1-2 stages of a low-shelf cut that
    strongly attenuates just below the cutoff (StaticEQ; for freq<200 the
    renderer materializes it as a LowShelfFilter). Note explicitly that this
    is a shelf-cut approximation, not a true brick-wall HPF (it is real DSP,
    not a proxy, but it does not have the ideal HPF response).
  - LUFS uses the real pyloudnorm (_measure_lufs). Each stem's integrated LUFS
    is measured and pulled toward a per-role relative target with gain
    (loudness pyramid).
  - The master is expected to be normalised to a common LUFS by the caller via
    normalize_for_eval (so loudness cannot buy an advantage). This function
    returns the raw master mix.

The constant tables (_HPF_HZ / _EQ_RULES / _LUFS_TARGET) can be tuned later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..dsp.mix_state import MixState
from ..dsp.renderer import render
from ..dsp.loudness_norm import _measure_lufs
from ..tools import action_tools as A


# ============================================================================
# 1. stem role classification (from the name, case-insensitive substring match)
# ============================================================================
# Priority-ordered keyword table. kick / hihat are broken out as sub-roles of drums.
# Matching goes top to bottom and takes the first role that matches (kick/hihat
# are checked before drums so they get their own role. bass is not placed after
# the drum entries to avoid being misread as kick -- "bass" is a standalone
# keyword, so there is no collision).
_ROLE_KEYWORDS: List[Tuple[str, List[str]]] = [
    # drums sub-roles (checked first)
    ("hihat", ["hat", "hihat", "hi-hat", "cymbal", "ride", "crash", "overhead"]),
    ("kick",  ["kick", "bd", "bassdrum", "bass drum"]),
    ("drums", ["drum", "snare", "tom", "perc", "clap", "shaker", "tambourine"]),
    # remaining roles
    ("bass",  ["bass", "sub", "808"]),
    ("vocals", ["vocal", "vox", "lead", "voice", "sing", "choir", "harmony"]),
    ("guitar", ["guitar", "gtr", "acoustic", "electric"]),
    ("keys",  ["piano", "key", "synth", "organ", "pad", "rhodes", "epiano", "wurli"]),
]

# Parent category of each role (used by the LUFS / mud-cut policy). kick/hihat -> drums.
_PARENT_ROLE = {
    "hihat": "drums", "kick": "drums", "drums": "drums",
    "bass": "bass", "vocals": "vocals", "guitar": "guitar",
    "keys": "keys", "other": "other",
}


def classify_role(stem_name: str) -> str:
    """stem name -> role. Case-insensitive substring match. Falls back to 'other'.

    The returned role is the fine-grained one: vocals/bass/kick/hihat/drums/guitar/keys/other.
    """
    s = stem_name.lower()
    for role, kws in _ROLE_KEYWORDS:
        if any(k in s for k in kws):
            return role
    return "other"


# ============================================================================
# 2. High-pass filter (remove low-end masking/mud, reclaim headroom)
# ============================================================================
# role -> cutoff Hz. Attenuates just below the cutoff with a low-shelf cut
# (approximate HPF).
_HPF_HZ: Dict[str, float] = {
    "bass":   30.0,
    "kick":   35.0,
    "vocals": 90.0,
    "guitar": 110.0,   # centre of 90-120Hz
    "keys":   90.0,
    "other":  90.0,
    "drums":  60.0,    # snare/tom etc. Above kick, removes low rumble
    "hihat":  500.0,   # centre of 400-600Hz. Cuts a lot of low end
}
# Shelf cut per HPF stage (dB). StaticEQ.gain_db is clamped to [-12,+12].
# A shallow HPF is a single stage (-9dB); deep HPFs such as hihat stack two
# stages for a steeper roll-off.
_HPF_SHELF_GAIN_DB = -9.0
_HPF_SHELF_Q = 0.7
# Roles that stack two stages (percussive high-frequency content needing deep
# low-end removal)
_HPF_TWO_STAGE = {"hihat"}


# ============================================================================
# 3. Complementary / subtractive EQ (avoid frequency masking)
# ============================================================================
# Each rule = (set of parent_roles or roles, freq, gain_db, q, label).
# gain_db stays within [-12,+12]. A role matches either as a fine-grained role
# or as a parent role.
@dataclass(frozen=True)
class EQRule:
    roles: Tuple[str, ...]       # targets (fine-grained role or parent role name)
    freq: float
    gain_db: float
    q: float
    label: str


_EQ_RULES: Tuple[EQRule, ...] = (
    # --- kick <-> bass low-end separation ---
    # kick: give it the 60-100Hz thump (boost) / cut 100-120Hz slightly to yield to bass
    EQRule(("kick",), 80.0, +2.5, 1.0, "kick_thump_boost"),
    EQRule(("kick",), 110.0, -2.0, 1.2, "kick_cut_for_bass_sustain"),
    # bass: give it the 80-120Hz sustain (boost) / cut 60-100Hz slightly to yield to kick
    EQRule(("bass",), 100.0, +2.0, 1.0, "bass_sustain_boost"),
    EQRule(("bass",), 70.0, -2.0, 1.2, "bass_cut_for_kick_thump"),
    # --- mud band 200-500Hz: gentle cut on guitar/keys/other ---
    EQRule(("guitar", "keys", "other"), 300.0, -3.0, 0.8, "mud_cut"),
    # vocals keep their warmth (no cut) -> no rule
    # --- vocal presence 2-5kHz ---
    EQRule(("vocals",), 3500.0, +2.5, 1.0, "vocal_presence_boost"),
    EQRule(("guitar", "keys"), 3500.0, -2.0, 1.2, "presence_cut_for_vocal"),
    # --- air 10-12kHz: gentle high-shelf boost on vocals ---
    EQRule(("vocals",), 11000.0, +1.5, 0.7, "vocal_air_shelf"),
)


# ============================================================================
# 4. per-stem LUFS balance (loudness pyramid)
# ============================================================================
# role -> relative integrated LUFS target. The relative balance matters more
# than the absolute values.
# vocals up front / kick and snare prominent / bass as the foundation /
# guitar and keys supporting / other.
_LUFS_TARGET: Dict[str, float] = {
    "vocals": -15.0,
    "kick":   -16.0,
    "drums":  -16.0,   # includes snare
    "hihat":  -20.0,   # hihat stays controlled, never too prominent
    "bass":   -17.0,
    "guitar": -20.0,
    "keys":   -20.0,
    "other":  -20.0,
}
# Limits on the gain adjustment (clamped so an essentially silent stem does not
# get +40dB or similar).
_MAX_GAIN_DB = 12.0
_MIN_GAIN_DB = -18.0
# Stems quieter than this LUFS are treated as "effectively silent" and excluded
# from the LUFS balancing.
_SILENCE_LUFS = -50.0


# ============================================================================
# main
# ============================================================================
@dataclass
class StemDecision:
    role: str
    lufs_before: float
    hpf: List[dict] = field(default_factory=list)       # [{freq, gain_db, q}]
    eq: List[dict] = field(default_factory=list)        # [{freq, gain_db, q, label}]
    gain_db: float = 0.0
    lufs_after: float = float("nan")


def _stack_hpf(role: str) -> List[dict]:
    """Return the role's HPF as a list of low-shelf cuts."""
    cutoff = _HPF_HZ.get(role, _HPF_HZ.get(_PARENT_ROLE.get(role, "other"), 90.0))
    n_stage = 2 if role in _HPF_TWO_STAGE else 1
    return [{"freq": cutoff, "gain_db": _HPF_SHELF_GAIN_DB, "q": _HPF_SHELF_Q}
            for _ in range(n_stage)]


def _eq_for_role(role: str) -> List[dict]:
    """Match EQ rules on both the fine-grained role and the parent role, and return the list to apply."""
    parent = _PARENT_ROLE.get(role, "other")
    out: List[dict] = []
    for r in _EQ_RULES:
        if role in r.roles or parent in r.roles:
            out.append({"freq": r.freq, "gain_db": r.gain_db,
                        "q": r.q, "label": r.label})
    return out


def build_knowledge_base_mix(
        stems: Dict[str, np.ndarray], sr: int,
        exclude: Tuple[str, ...] = ("mixture",),
        eq: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Build the knowledge-based initial mix.

    Args:
        eq: If True, apply the HPF + complementary/subtractive EQ (the original
            behaviour). If False, stack no EQ at all and build the initial mix
            from the per-stem LUFS balance (loudness pyramid) alone. Following
            the design policy "if EQ does not change the evaluation score, do
            not EQ" (eval-gated EQ), eq=False can be used as the starting point
            of the agreement reward loop.

    Returns:
        (master_mix (C, N) float32, meta dict).
        meta = {"method", "eq", "per_stem": {name: StemDecision-as-dict}, "master_lufs"}.
        master_mix is raw (not normalised). Call normalize_for_eval before evaluating.
    """
    main = {k: v for k, v in stems.items() if k not in exclude}
    if not main:
        raise ValueError(f"no stems after excluding {exclude}: {list(stems)}")

    decisions: Dict[str, StemDecision] = {}

    # --- (a) role classification per stem + LUFS before processing ---
    for name, x in main.items():
        role = classify_role(name)
        try:
            lufs0 = _measure_lufs(np.asarray(x, dtype=np.float32), sr)
        except Exception:                                   # noqa: BLE001
            lufs0 = float("nan")
        decisions[name] = StemDecision(role=role, lufs_before=lufs0)

    # --- (b) stack HPF + complementary EQ onto the state (only when eq=True) ---
    state = MixState.initial_from_stems(main, sr)
    if eq:
        for name, dec in decisions.items():
            # HPF (approximated with a low-shelf cut)
            for hpf in _stack_hpf(dec.role):
                state = A.apply_static_eq(state, {"track": name, **hpf})
                dec.hpf.append(hpf)
            # Complementary / subtractive EQ
            for eqb in _eq_for_role(dec.role):
                state = A.apply_static_eq(
                    state, {"track": name, "freq": eqb["freq"],
                            "gain_db": eqb["gain_db"], "q": eqb["q"]})
                dec.eq.append(eqb)

    # --- (c) per-stem LUFS balance (gain) ---
    # Render each stem individually after HPF/EQ, measure its LUFS, and pull it
    # toward the target.
    for name, dec in decisions.items():
        if not (np.isfinite(dec.lufs_before) and dec.lufs_before > _SILENCE_LUFS):
            # Effectively silent -> no gain adjustment
            dec.gain_db = 0.0
            dec.lufs_after = dec.lufs_before
            continue
        # Build the single-stem mix after EQ and measure its LUFS (HPF/EQ change the LUFS)
        single = _render_single_stem(state, name)
        try:
            lufs_eq = _measure_lufs(single, sr)
        except Exception:                                  # noqa: BLE001
            lufs_eq = dec.lufs_before
        target = _LUFS_TARGET.get(
            dec.role, _LUFS_TARGET.get(_PARENT_ROLE.get(dec.role, "other"), -20.0))
        gain = float(np.clip(target - lufs_eq, _MIN_GAIN_DB, _MAX_GAIN_DB))
        state = A.apply_static_gain(state, {"track": name, "gain_db": gain})
        dec.gain_db = gain
        # LUFS after adjustment (gain is linear so lufs_eq + gain matches, but
        # confirm against the measurement)
        dec.lufs_after = lufs_eq + gain

    # --- (d) render the master mix ---
    master = render(state)
    try:
        master_lufs = _measure_lufs(master, sr)
    except Exception:                                      # noqa: BLE001
        master_lufs = float("nan")

    meta = {
        "method": "knowledge_base_initial_mix",
        "eq": bool(eq),
        "sr": sr,
        "per_stem": {name: _decision_to_dict(d) for name, d in decisions.items()},
        "master_lufs": master_lufs,
    }
    return master, meta


def _render_single_stem(state: MixState, name: str, cache=None) -> np.ndarray:
    """Render just one stem while keeping the state's processors intact.

    Args:
        cache: when omitted (the default), renders without a cache as before.
            Passing a ``RenderCache`` enables incremental rendering (the output
            is bit-identical). **This solo state holds only a single stem**, so
            it must not share a cache with the full mix (that would trigger a
            rebind every time and end up slower). Pass a separate instance per
            stem name. ``dsp/stem_lufs_cache.StemLufsCache`` does exactly that.
    """
    from dataclasses import replace
    solo_stems = {name: state.stems[name]}
    solo_static = {name: state.static_processors.get(name, ())}
    solo_trans = {name: state.transient_processors.get(name, ())}
    solo_section = {name: state.section_processors.get(name, ())}
    solo = MixState(
        stems=solo_stems, sample_rate=state.sample_rate,
        section_boundaries=state.section_boundaries,
        static_processors=solo_static,
        section_processors=solo_section,
        transient_processors=solo_trans,
        master_chain=(),
    )
    return render(solo) if cache is None else render(solo, cache=cache)


def _decision_to_dict(d: StemDecision) -> dict:
    return {
        "role": d.role,
        "lufs_before": d.lufs_before,
        "hpf": d.hpf,
        "eq": d.eq,
        "gain_db": d.gain_db,
        "lufs_after": d.lufs_after,
    }
