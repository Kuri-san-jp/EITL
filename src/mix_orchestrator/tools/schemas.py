"""Pydantic schemas for every tool the LLM can call.

These schemas drive both:
  (a) the Anthropic / OpenAI tool-use spec exposed to the model, and
  (b) runtime validation before the action is applied.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ---------- Generic envelope ----------

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


# ---------- Action tools (static) ----------

class ApplyStaticEQArgs(BaseModel):
    track: str
    freq: float = Field(gt=0, le=20000)
    gain_db: float = Field(ge=-12, le=12)
    q: float = Field(gt=0, le=10)


class ApplyStaticCompressorArgs(BaseModel):
    track: str
    threshold_db: float = Field(ge=-60, le=0)
    ratio: float = Field(ge=1.0, le=20.0)
    attack_ms: float = Field(gt=0, le=500)
    release_ms: float = Field(gt=0, le=5000)
    knee_db: float = Field(default=6.0, ge=0, le=18)


class ApplyStaticGainArgs(BaseModel):
    track: str
    gain_db: float = Field(ge=-18, le=12)


class ApplyStaticPanArgs(BaseModel):
    track: str
    pan: float = Field(ge=-1.0, le=1.0)


class ApplyStaticWidthArgs(BaseModel):
    track: str
    width: float = Field(ge=0.0, le=2.0)


# ---------- Action tools (section) ----------

class ApplySectionGainArgs(BaseModel):
    track: str
    section_label: str
    gain_db: float = Field(ge=-12, le=12)
    crossfade_ms: float = Field(default=200, ge=0, le=2000)


class ApplySectionEQArgs(BaseModel):
    track: str
    section_label: str
    freq: float = Field(gt=0, le=20000)
    gain_db: float = Field(ge=-12, le=12)
    q: float = Field(gt=0, le=10)
    crossfade_ms: float = Field(default=200, ge=0, le=2000)


class ApplySectionPanArgs(BaseModel):
    track: str
    section_label: str
    pan: float = Field(ge=-1.0, le=1.0)
    crossfade_ms: float = Field(default=200, ge=0, le=2000)


class ApplySectionWidthArgs(BaseModel):
    track: str
    section_label: str
    width: float = Field(ge=0.0, le=2.0)
    crossfade_ms: float = Field(default=200, ge=0, le=2000)


class ApplySectionCompressorArgs(BaseModel):
    track: str
    section_label: str
    threshold_db: float = Field(ge=-40, le=-6)     # extreme values such as -60 are disallowed to prevent the pumping cheat
    ratio: float = Field(ge=1.0, le=4.0)
    attack_ms: float = Field(default=10.0, gt=0, le=100)
    release_ms: float = Field(default=100.0, gt=0, le=1000)
    knee_db: float = Field(default=6.0, ge=0, le=12)
    crossfade_ms: float = Field(default=200, ge=0, le=2000)


# ---------- Action tools (master) ----------

class ApplyMasterEQArgs(BaseModel):
    freq: float = Field(gt=0, le=20000)
    gain_db: float = Field(ge=-6, le=6)
    q: float = Field(gt=0, le=10)


class ApplyMasterCompressorArgs(BaseModel):
    threshold_db: float = Field(ge=-30, le=0)
    ratio: float = Field(ge=1.0, le=4.0)
    attack_ms: float = Field(gt=0, le=200)
    release_ms: float = Field(gt=0, le=2000)


class ApplyMasterLimiterArgs(BaseModel):
    ceiling_db: float = Field(default=-1.0, ge=-3, le=-0.3)
    release_ms: float = Field(default=100, ge=50, le=500)


# ---------- Action tools (transient / automation) ----------

class ApplyTransientShaperArgs(BaseModel):
    track: str
    attack_gain_db: float = Field(ge=-12, le=12)
    sustain_gain_db: float = Field(ge=-12, le=12)


class ApplyGainAutomationArgs(BaseModel):
    track: str
    breakpoints: List[Tuple[float, float]] = Field(min_length=2)   # (time_sec, gain_db)

    # : this was the only place missing the validation that pan/width
    # automation already had. renderer._apply_gain_automation calls
    # np.interp(x, times, gains), and np.interp **only assumes ``times`` is
    # ascending, it does not check**
    # (docs: "if `xp` is not increasing, the results are nonsense").
    # Non-ascending breakpoints do not raise; they silently produce a different
    # gain curve. Reject non-monotonic times explicitly, by the same rule as
    # pan/width.
    #
    # The gain_db range matches ApplyStaticGainArgs (-18..+12). Without that, the
    # loophole "automation means unlimited gain" opens up: (a) it could use a wider
    # range than static gain, which breaks comparison across actions, and (b) reward
    # hacking that pushes one stem to +40 dB and effectively silences the rest would
    # pass unconstrained.
    @field_validator("breakpoints")
    @classmethod
    def _check_gain(cls, v):
        ts = [t for t, _ in v]
        if ts != sorted(ts) or ts[0] < 0:
            raise ValueError("breakpoints must be time-ascending and start at t>=0")
        if any(not (-18.0 <= g <= 12.0) for _, g in v):
            raise ValueError("gain_db values must be in [-18, 12]")
        return v


class ApplyPanAutomationArgs(BaseModel):
    track: str
    breakpoints: List[Tuple[float, float]] = Field(min_length=2)   # (time_sec, pan -1..1)

    @field_validator("breakpoints")
    @classmethod
    def _check_pan(cls, v):
        ts = [t for t, _ in v]
        if ts != sorted(ts) or ts[0] < 0:
            raise ValueError("breakpoints must be time-ascending and start at t>=0")
        if any(not (-1.0 <= p <= 1.0) for _, p in v):
            raise ValueError("pan values must be in [-1, 1]")
        return v


class ApplyWidthAutomationArgs(BaseModel):
    track: str
    breakpoints: List[Tuple[float, float]] = Field(min_length=2)   # (time_sec, width 0..2)

    @field_validator("breakpoints")
    @classmethod
    def _check_width(cls, v):
        ts = [t for t, _ in v]
        if ts != sorted(ts) or ts[0] < 0:
            raise ValueError("breakpoints must be time-ascending and start at t>=0")
        if any(not (0.0 <= w <= 2.0) for _, w in v):
            raise ValueError("width values must be in [0, 2]")
        return v


# ---------- Dynamic / multiband / sidechain ----------

class ApplyDynamicEQArgs(BaseModel):
    track: str
    freq: float = Field(gt=0, le=20000)
    threshold_db: float = Field(ge=-60, le=0)
    gain_db: float = Field(ge=-12, le=12)
    q: float = Field(default=1.0, gt=0, le=10)
    attack_ms: float = Field(default=10.0, gt=0, le=200)
    release_ms: float = Field(default=100.0, gt=0, le=2000)


class ApplyMultibandCompArgs(BaseModel):
    track: str
    crossover_lo: float = Field(default=200.0, gt=20, le=2000)
    crossover_hi: float = Field(default=3000.0, gt=500, le=12000)
    low_threshold_db:  float = Field(default=-24.0, ge=-60, le=0)
    low_ratio:         float = Field(default=2.0, ge=1.0, le=10.0)
    low_attack_ms:     float = Field(default=10.0, gt=0, le=200)
    low_release_ms:    float = Field(default=200.0, gt=0, le=2000)
    mid_threshold_db:  float = Field(default=-18.0, ge=-60, le=0)
    mid_ratio:         float = Field(default=2.0, ge=1.0, le=10.0)
    mid_attack_ms:     float = Field(default=10.0, gt=0, le=200)
    mid_release_ms:    float = Field(default=200.0, gt=0, le=2000)
    high_threshold_db: float = Field(default=-18.0, ge=-60, le=0)
    high_ratio:        float = Field(default=2.0, ge=1.0, le=10.0)
    high_attack_ms:    float = Field(default=5.0, gt=0, le=200)
    high_release_ms:   float = Field(default=100.0, gt=0, le=2000)


class ApplySidechainCompArgs(BaseModel):
    track: str
    sidechain_track: str
    threshold_db: float = Field(ge=-60, le=0)
    ratio: float = Field(ge=1.0, le=20.0)
    attack_ms: float = Field(default=5.0, gt=0, le=200)
    release_ms: float = Field(default=100.0, gt=0, le=2000)


# ---------- Time-based / harmonic insert FX ----------

class ApplyReverbArgs(BaseModel):
    track: str
    room_size: float = Field(default=0.5, ge=0.0, le=1.0)
    damping: float = Field(default=0.5, ge=0.0, le=1.0)
    # anti-cheat: wet_level capped at 0.5 — a fully-wet wash would mask the
    # dry-stem balance and let the model "cheat" reverb-heavy submixes.
    wet_level: float = Field(default=0.3, ge=0.0, le=0.5)
    dry_level: float = Field(default=0.8, ge=0.0, le=1.0)
    width: float = Field(default=1.0, ge=0.0, le=1.0)
    # 0 disables the pre-reverb highpass; otherwise restricted to low-mid.
    highpass_hz: float = Field(default=0.0, ge=0.0, le=1000.0)


class ApplyDelayArgs(BaseModel):
    track: str
    # anti-cheat: keep delays musical, not seconds-long ambience washes.
    delay_seconds: float = Field(default=0.25, gt=0.0, le=1.0)
    feedback: float = Field(default=0.3, ge=0.0, le=0.6)   # <1 to stay stable; ≤0.6 avoids runaway repeats
    mix: float = Field(default=0.25, ge=0.0, le=0.5)        # ≤0.5 so dry stays dominant


class ApplySaturationArgs(BaseModel):
    track: str
    # anti-cheat: drive capped at 12 dB — light harmonic warmth only, not
    # a destructive fuzz that would falsely boost perceived loudness/THD.
    drive_db: float = Field(default=6.0, ge=0.0, le=12.0)


class ApplyDeEsserArgs(BaseModel):
    track: str
    center_hz: float = Field(default=7000.0, ge=5000.0, le=9000.0)   # sibilance band
    threshold_db: float = Field(default=-30.0, ge=-60.0, le=0.0)
    ratio: float = Field(default=4.0, ge=1.0, le=10.0)
    # anti-cheat: max reduction depth capped at 12 dB so the de-esser cannot
    # null the entire high band into a fake-smooth (lisping) high end.
    range_db: float = Field(default=6.0, gt=0.0, le=12.0)
    attack_ms: float = Field(default=1.0, gt=0.0, le=50.0)
    release_ms: float = Field(default=50.0, gt=0.0, le=500.0)


# ---------- Perception / evaluation tools ----------

class EvaluateMultiResolutionArgs(BaseModel):
    resolutions: List[Literal["global", "section", "short_term", "momentary"]] = Field(
        default_factory=lambda: ["global", "section", "short_term", "momentary"]
    )


class ScoreEarArgs(BaseModel):
    ear: str
    window_sec: Optional[Tuple[float, float]] = None


# ---------- State tools ----------

class SaveCheckpointArgs(BaseModel):
    label: str


class RollbackArgs(BaseModel):
    target_state_id: str


# ---------- Registry exposed to LLM ----------

TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    "apply_static_eq":          {"args": ApplyStaticEQArgs,          "scope": "static"},
    "apply_static_compressor":  {"args": ApplyStaticCompressorArgs,  "scope": "static"},
    "apply_static_gain":        {"args": ApplyStaticGainArgs,        "scope": "static"},
    "apply_static_pan":         {"args": ApplyStaticPanArgs,         "scope": "static"},
    "apply_static_width":       {"args": ApplyStaticWidthArgs,       "scope": "static"},
    "apply_section_gain":       {"args": ApplySectionGainArgs,       "scope": "section"},
    "apply_section_eq":         {"args": ApplySectionEQArgs,         "scope": "section"},
    "apply_section_pan":        {"args": ApplySectionPanArgs,        "scope": "section"},
    "apply_section_width":      {"args": ApplySectionWidthArgs,      "scope": "section"},
    "apply_section_compressor": {"args": ApplySectionCompressorArgs, "scope": "section"},
    "apply_master_eq":          {"args": ApplyMasterEQArgs,          "scope": "master"},
    "apply_master_compressor":  {"args": ApplyMasterCompressorArgs,  "scope": "master"},
    "apply_master_limiter":     {"args": ApplyMasterLimiterArgs,     "scope": "master"},
    "apply_transient_shaper":   {"args": ApplyTransientShaperArgs,   "scope": "transient"},
    "apply_gain_automation":    {"args": ApplyGainAutomationArgs,    "scope": "automation"},
    "apply_pan_automation":     {"args": ApplyPanAutomationArgs,     "scope": "automation"},
    "apply_width_automation":   {"args": ApplyWidthAutomationArgs,   "scope": "automation"},
    "apply_dynamic_eq":         {"args": ApplyDynamicEQArgs,         "scope": "dynamic"},
    "apply_multiband_compressor": {"args": ApplyMultibandCompArgs,   "scope": "dynamic"},
    "apply_sidechain_compressor": {"args": ApplySidechainCompArgs,   "scope": "dynamic"},
    "apply_reverb":             {"args": ApplyReverbArgs,             "scope": "insert"},
    "apply_delay":              {"args": ApplyDelayArgs,              "scope": "insert"},
    "apply_saturation":         {"args": ApplySaturationArgs,         "scope": "insert"},
    "apply_deesser":            {"args": ApplyDeEsserArgs,            "scope": "insert"},
    "evaluate_multi_resolution":{"args": EvaluateMultiResolutionArgs,"scope": "perception"},
    "score_ear":                {"args": ScoreEarArgs,               "scope": "perception"},
    "save_checkpoint":          {"args": SaveCheckpointArgs,         "scope": "state"},
    "rollback":                 {"args": RollbackArgs,               "scope": "state"},
}


def anthropic_tool_specs() -> List[Dict[str, Any]]:
    """Convert TOOL_CATALOG to Anthropic tool-use schema list."""
    specs = []
    for name, info in TOOL_CATALOG.items():
        schema = info["args"].model_json_schema()
        # Remove pydantic-specific fields Anthropic doesn't need
        schema.pop("$defs", None)
        schema.pop("title", None)
        specs.append({
            "name": name,
            "description": _doc_for(name, info["scope"]),
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        })
    return specs


def _doc_for(name: str, scope: str) -> str:
    base = {
        "apply_static_eq": "Apply a parametric EQ band to one stem for the whole song.",
        "apply_static_compressor": "Apply a compressor to one stem (full song).",
        "apply_static_gain": "Trim a stem's level (full song).",
        "apply_static_pan": "Pan a stem L↔R (full song).",
        "apply_static_width": "Adjust stereo width of a stem (0=mono, 1=neutral, 2=wide).",
        "apply_section_gain": "Boost / cut a stem within ONE labeled section (verse_1, chorus_1, ...).",
        "apply_section_eq": "EQ a stem within ONE labeled section, crossfaded at boundaries.",
        "apply_section_pan": "Pan a stem within ONE labeled section, crossfaded at boundaries (time-varying pan).",
        "apply_section_width": "Adjust stereo width of a stem within ONE labeled section, crossfaded (time-varying width).",
        "apply_section_compressor": "Compress a stem within ONE labeled section, crossfaded (time-varying dynamics).",
        "apply_master_eq": "EQ the master bus (small moves only).",
        "apply_master_compressor": "Glue compression on the master bus.",
        "apply_master_limiter": "Brick-wall ceiling on master to prevent clipping.",
        "apply_transient_shaper": "Shape attack/sustain envelope on a stem (drums-style).",
        # Why the ranges and units are spelled out in the description: breakpoints
        # is a List[Tuple[float, float]], so the JSON schema carries no information
        # that "the first element is seconds and the second is dB". This action is
        # not shown to the LLM by default, so this wording does not change the
        # prompt of any existing run.
        "apply_gain_automation": (
            "Time-varying gain on a stem: a breakpoint list "
            "[[time_sec, gain_db], ...], linearly interpolated between "
            "breakpoints and held constant outside the first/last breakpoint. "
            "Times are seconds from the START of the audio being mixed and must "
            "be strictly non-decreasing; gain_db must be within [-18, +12] "
            "(same range as apply_static_gain). At least 2 breakpoints."),
        "apply_pan_automation": "Continuous time-varying pan on a stem (breakpoint list, linearly interpolated).",
        "apply_width_automation": "Continuous time-varying stereo width on a stem (breakpoint list, linearly interpolated).",
        "apply_dynamic_eq": "Dynamic EQ: cut/boost only when band envelope crosses threshold (handles only-loud-parts problems).",
        "apply_multiband_compressor": "3-band compressor (low/mid/high) — independent compression in each band.",
        "apply_sidechain_compressor": "Compress one stem keyed by another stem's envelope (kick→bass ducking, etc.).",
        "apply_reverb": "Add algorithmic reverb to one stem (per-stem insert). wet_level capped ≤0.5; optional pre-reverb highpass keeps the tail clean.",
        "apply_delay": "Add a feedback delay (echo) to one stem (per-stem insert). delay_seconds≤1.0, feedback≤0.6, mix≤0.5.",
        "apply_saturation": "Add light harmonic saturation/warmth to one stem (drive_db≤12, non-destructive).",
        "apply_deesser": "De-ess one stem: dynamically duck the sibilance band (~5–9 kHz) when it gets harsh. Reduction depth capped by range_db.",
        "evaluate_multi_resolution": "Compute full multi-resolution score dict (global+section+short_term+momentary).",
        "score_ear": "Run ONE ear by name on the current mix (optionally windowed).",
        "save_checkpoint": "Save the current state under a human label so you can rollback later.",
        "rollback": "Restore a previous state by id.",
    }
    return base.get(name, f"{scope} action: {name}")
