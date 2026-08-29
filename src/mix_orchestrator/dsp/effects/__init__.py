"""DSP effect data models + pedalboard adapters.

Effects are *frozen* dataclasses carrying only parameters. The renderer
materializes them into pedalboard chains at render-time. This keeps
MixState fully picklable and content-hashable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---------- Base markers ----------


@dataclass(frozen=True)
class Effect:
    """Marker base for any effect (static / section / master / etc.)."""


@dataclass(frozen=True)
class StaticEffect(Effect):
    """Applied to the whole stem before summing."""


@dataclass(frozen=True)
class SectionEffect(Effect):
    """Applied only within [start_sec, end_sec], crossfaded at boundaries."""
    start_sec: float
    end_sec: float
    crossfade_ms: float = 200.0


@dataclass(frozen=True)
class MasterEffect(Effect):
    """Applied to the summed bus after all stem processing."""


@dataclass(frozen=True)
class TransientEffect(Effect):
    """Sub-100ms shaping on a stem."""


# ---------- Static (per-stem, full-song) ----------


@dataclass(frozen=True)
class StaticGain(StaticEffect):
    gain_db: float


@dataclass(frozen=True)
class StaticPan(StaticEffect):
    pan: float                # -1..+1


@dataclass(frozen=True)
class StaticWidth(StaticEffect):
    width: float              # 0 (mono) .. 2 (extra-wide)


@dataclass(frozen=True)
class StaticEQ(StaticEffect):
    freq: float
    gain_db: float
    q: float


@dataclass(frozen=True)
class StaticCompressor(StaticEffect):
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    knee_db: float = 6.0


# ---------- Section (per-stem, time-bounded) ----------


@dataclass(frozen=True)
class SectionGain(SectionEffect):
    gain_db: float = 0.0


@dataclass(frozen=True)
class SectionEQ(SectionEffect):
    freq: float = 1000.0
    gain_db: float = 0.0
    q: float = 1.0


@dataclass(frozen=True)
class SectionPan(SectionEffect):
    """Time-bounded pan (made continuous by crossfading)."""
    pan: float = 0.0          # -1..+1


@dataclass(frozen=True)
class SectionWidth(SectionEffect):
    """Time-bounded stereo width (made continuous by crossfading)."""
    width: float = 1.0        # 0 (mono) .. 2 (extra-wide)


@dataclass(frozen=True)
class SectionCompressor(SectionEffect):
    """Time-bounded compressor (made continuous by crossfading)."""
    threshold_db: float = -18.0
    ratio: float = 2.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    knee_db: float = 6.0


# ---------- Master bus ----------


@dataclass(frozen=True)
class MasterEQ(MasterEffect):
    freq: float
    gain_db: float
    q: float


@dataclass(frozen=True)
class MasterCompressor(MasterEffect):
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float


@dataclass(frozen=True)
class MasterLimiter(MasterEffect):
    ceiling_db: float = -1.0
    release_ms: float = 100.0


# ---------- Automation (breakpoints) ----------


@dataclass(frozen=True)
class GainAutomation(StaticEffect):
    """Breakpoints: List[(time_sec, gain_db)] (linearly interpolated)."""
    breakpoints: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class PanAutomation(StaticEffect):
    """Continuous pan automation. Breakpoints: List[(time_sec, pan -1..1)] (linearly interpolated)."""
    breakpoints: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class WidthAutomation(StaticEffect):
    """Continuous width automation. Breakpoints: List[(time_sec, width 0..2)] (linearly interpolated)."""
    breakpoints: Tuple[Tuple[float, float], ...]


# ---------- Transient ----------


@dataclass(frozen=True)
class TransientShaper(TransientEffect):
    attack_gain_db: float = 0.0
    sustain_gain_db: float = 0.0


# ---------- Dynamic / multiband / sidechain ----------


@dataclass(frozen=True)
class DynamicEQ(StaticEffect):
    """Reduces (or boosts) `gain_db` at `freq` only when the band envelope
    crosses `threshold_db`."""
    freq: float
    threshold_db: float
    gain_db: float
    q: float = 1.0
    attack_ms: float = 10.0
    release_ms: float = 100.0


@dataclass(frozen=True)
class MultibandComp(StaticEffect):
    """3-band compressor (low/mid/high) — bands are dicts but we keep them
    as a fixed 3-tuple to stay hashable & frozen-safe."""
    crossover_lo: float = 200.0
    crossover_hi: float = 3000.0
    low_threshold_db:  float = -24.0
    low_ratio:         float = 2.0
    low_attack_ms:     float = 10.0
    low_release_ms:    float = 200.0
    mid_threshold_db:  float = -18.0
    mid_ratio:         float = 2.0
    mid_attack_ms:     float = 10.0
    mid_release_ms:    float = 200.0
    high_threshold_db: float = -18.0
    high_ratio:        float = 2.0
    high_attack_ms:    float = 5.0
    high_release_ms:   float = 100.0


@dataclass(frozen=True)
class SidechainComp(StaticEffect):
    """Compress `track` keyed by another stem's envelope (pre-fader)."""
    sidechain_track: str
    threshold_db: float
    ratio: float
    attack_ms: float = 5.0
    release_ms: float = 100.0


# ---------- Time-based / harmonic insert FX (per-stem) ----------


@dataclass(frozen=True)
class Reverb(StaticEffect):
    """Algorithmic reverb (pedalboard.Reverb) as a per-stem insert.

    `wet_level` is intentionally capped upstream (schema ≤0.5) so the mix
    cannot be "cheated" into a fully-wet wash that masks dry-stem balance.
    `highpass_hz` (optional, >0) runs a HighpassFilter on the wet path only
    to keep low-frequency build-up out of the reverb tail (pre-reverb HPF).
    """
    room_size: float = 0.5
    damping: float = 0.5
    wet_level: float = 0.3
    dry_level: float = 0.8
    width: float = 1.0
    highpass_hz: float = 0.0     # 0 = disabled


@dataclass(frozen=True)
class Delay(StaticEffect):
    """Feedback delay line (pedalboard.Delay) as a per-stem insert."""
    delay_seconds: float = 0.25
    feedback: float = 0.3
    mix: float = 0.25


@dataclass(frozen=True)
class Saturation(StaticEffect):
    """Light harmonic saturation (pedalboard.Distortion driven gently).

    Intended for *warmth* (added low-order harmonics), NOT destruction;
    `drive_db` is capped low upstream (schema ≤12 dB)."""
    drive_db: float = 6.0


@dataclass(frozen=True)
class DeEsser(StaticEffect):
    """Sibilance de-esser: dynamic gain-reduction restricted to the
    sibilance band (~5–9 kHz). No native pedalboard plugin exists, so the
    renderer band-isolates [center/√2, center·√2], follows its envelope,
    and ducks only that band when it exceeds `threshold_db`.

    `range_db` caps the maximum reduction (downward expander style) so an
    over-aggressive setting cannot null the band entirely (anti-cheat:
    keeps consonant intelligibility, prevents lisping the mix into a
    fake-smooth high end)."""
    center_hz: float = 7000.0
    threshold_db: float = -30.0
    ratio: float = 4.0
    range_db: float = 6.0        # max reduction depth (>0)
    attack_ms: float = 1.0
    release_ms: float = 50.0


__all__ = [
    "Effect", "StaticEffect", "SectionEffect", "MasterEffect", "TransientEffect",
    "StaticGain", "StaticPan", "StaticWidth", "StaticEQ", "StaticCompressor",
    "SectionGain", "SectionEQ", "SectionPan", "SectionWidth", "SectionCompressor",
    "MasterEQ", "MasterCompressor", "MasterLimiter",
    "GainAutomation", "PanAutomation", "WidthAutomation", "TransientShaper",
    "DynamicEQ", "MultibandComp", "SidechainComp",
    "Reverb", "Delay", "Saturation", "DeEsser",
]
