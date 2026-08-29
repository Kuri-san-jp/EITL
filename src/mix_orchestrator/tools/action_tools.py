"""Handlers that translate ToolCall(name=..., args=...) → new MixState."""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..dsp.mix_state import MixState
from ..dsp.effects import (
    StaticGain, StaticPan, StaticWidth, StaticEQ, StaticCompressor,
    SectionGain, SectionEQ, SectionPan, SectionWidth, SectionCompressor,
    MasterEQ, MasterCompressor, MasterLimiter,
    GainAutomation, PanAutomation, WidthAutomation, TransientShaper,
    DynamicEQ, MultibandComp, SidechainComp,
    Reverb, Delay, Saturation, DeEsser,
)
from . import schemas as S


# ---------- helpers ----------

def _find_section(state: MixState, label: str) -> Tuple[float, float]:
    """Return (start_sec, end_sec) of a labeled section."""
    if not state.section_boundaries:
        raise ValueError("no section boundaries set on this state")
    sb = sorted(state.section_boundaries, key=lambda x: x[0])
    duration = max(s.shape[-1] for s in state.stems.values()) / state.sample_rate
    for i, (t, lab) in enumerate(sb):
        if lab == label:
            t_next = sb[i+1][0] if i+1 < len(sb) else duration
            return float(t), float(t_next)
    raise ValueError(f"section {label!r} not found in {[lab for _, lab in sb]}")


# ---------- handlers ----------

def apply_static_eq(state: MixState, args: dict) -> MixState:
    a = S.ApplyStaticEQArgs(**args)
    eff = StaticEQ(freq=a.freq, gain_db=a.gain_db, q=a.q)
    return state.add_static(a.track, eff,
                            summary=f"static_eq[{a.track}] f={a.freq:.0f} g={a.gain_db:+.1f}dB Q={a.q:.2f}")


def apply_static_compressor(state: MixState, args: dict) -> MixState:
    a = S.ApplyStaticCompressorArgs(**args)
    eff = StaticCompressor(threshold_db=a.threshold_db, ratio=a.ratio,
                           attack_ms=a.attack_ms, release_ms=a.release_ms, knee_db=a.knee_db)
    return state.add_static(a.track, eff,
                            summary=f"comp[{a.track}] T={a.threshold_db:.1f} R={a.ratio:.1f}")


def apply_static_gain(state: MixState, args: dict) -> MixState:
    a = S.ApplyStaticGainArgs(**args)
    return state.add_static(a.track, StaticGain(gain_db=a.gain_db),
                            summary=f"gain[{a.track}] {a.gain_db:+.1f}dB")


def apply_static_pan(state: MixState, args: dict) -> MixState:
    a = S.ApplyStaticPanArgs(**args)
    return state.add_static(a.track, StaticPan(pan=a.pan),
                            summary=f"pan[{a.track}] {a.pan:+.2f}")


def apply_static_width(state: MixState, args: dict) -> MixState:
    a = S.ApplyStaticWidthArgs(**args)
    return state.add_static(a.track, StaticWidth(width=a.width),
                            summary=f"width[{a.track}] {a.width:.2f}")


def apply_section_gain(state: MixState, args: dict) -> MixState:
    a = S.ApplySectionGainArgs(**args)
    start, end = _find_section(state, a.section_label)
    eff = SectionGain(start_sec=start, end_sec=end, crossfade_ms=a.crossfade_ms, gain_db=a.gain_db)
    return state.add_section(a.track, eff,
                             summary=f"sec_gain[{a.track}@{a.section_label}] {a.gain_db:+.1f}dB")


def apply_section_eq(state: MixState, args: dict) -> MixState:
    a = S.ApplySectionEQArgs(**args)
    start, end = _find_section(state, a.section_label)
    eff = SectionEQ(start_sec=start, end_sec=end, crossfade_ms=a.crossfade_ms,
                    freq=a.freq, gain_db=a.gain_db, q=a.q)
    return state.add_section(a.track, eff,
                             summary=f"sec_eq[{a.track}@{a.section_label}] f={a.freq:.0f} g={a.gain_db:+.1f}dB")


def apply_section_pan(state: MixState, args: dict) -> MixState:
    a = S.ApplySectionPanArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    start, end = _find_section(state, a.section_label)
    eff = SectionPan(start_sec=start, end_sec=end, crossfade_ms=a.crossfade_ms, pan=a.pan)
    return state.add_section(a.track, eff,
                             summary=f"sec_pan[{a.track}@{a.section_label}] {a.pan:+.2f}")


def apply_section_width(state: MixState, args: dict) -> MixState:
    a = S.ApplySectionWidthArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    start, end = _find_section(state, a.section_label)
    eff = SectionWidth(start_sec=start, end_sec=end, crossfade_ms=a.crossfade_ms, width=a.width)
    return state.add_section(a.track, eff,
                             summary=f"sec_width[{a.track}@{a.section_label}] {a.width:.2f}")


def apply_section_compressor(state: MixState, args: dict) -> MixState:
    a = S.ApplySectionCompressorArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    start, end = _find_section(state, a.section_label)
    eff = SectionCompressor(start_sec=start, end_sec=end, crossfade_ms=a.crossfade_ms,
                            threshold_db=a.threshold_db, ratio=a.ratio,
                            attack_ms=a.attack_ms, release_ms=a.release_ms, knee_db=a.knee_db)
    return state.add_section(a.track, eff,
                             summary=f"sec_comp[{a.track}@{a.section_label}] T={a.threshold_db:.1f} R={a.ratio:.1f}")


def apply_pan_automation(state: MixState, args: dict) -> MixState:
    a = S.ApplyPanAutomationArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    bp = tuple((float(t), float(p)) for t, p in a.breakpoints)
    return state.add_static(a.track, PanAutomation(breakpoints=bp),
                            summary=f"pan_auto[{a.track}] {len(bp)} pts")


def apply_width_automation(state: MixState, args: dict) -> MixState:
    a = S.ApplyWidthAutomationArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    bp = tuple((float(t), float(w)) for t, w in a.breakpoints)
    return state.add_static(a.track, WidthAutomation(breakpoints=bp),
                            summary=f"width_auto[{a.track}] {len(bp)} pts")


def apply_master_eq(state: MixState, args: dict) -> MixState:
    a = S.ApplyMasterEQArgs(**args)
    return state.add_master(MasterEQ(freq=a.freq, gain_db=a.gain_db, q=a.q),
                            summary=f"master_eq f={a.freq:.0f} g={a.gain_db:+.1f}dB")


def apply_master_compressor(state: MixState, args: dict) -> MixState:
    a = S.ApplyMasterCompressorArgs(**args)
    return state.add_master(MasterCompressor(threshold_db=a.threshold_db, ratio=a.ratio,
                                              attack_ms=a.attack_ms, release_ms=a.release_ms),
                            summary=f"master_comp T={a.threshold_db:.1f} R={a.ratio:.1f}")


def apply_master_limiter(state: MixState, args: dict) -> MixState:
    a = S.ApplyMasterLimiterArgs(**args)
    return state.add_master(MasterLimiter(ceiling_db=a.ceiling_db, release_ms=a.release_ms),
                            summary=f"master_limit ceil={a.ceiling_db:.2f}dB")


def apply_transient_shaper(state: MixState, args: dict) -> MixState:
    a = S.ApplyTransientShaperArgs(**args)
    return state.add_transient(a.track,
                               TransientShaper(attack_gain_db=a.attack_gain_db,
                                               sustain_gain_db=a.sustain_gain_db),
                               summary=f"transient[{a.track}] atk={a.attack_gain_db:+.1f} sus={a.sustain_gain_db:+.1f}")


def apply_gain_automation(state: MixState, args: dict) -> MixState:
    a = S.ApplyGainAutomationArgs(**args)
    bp = tuple((float(t), float(g)) for t, g in a.breakpoints)
    return state.add_static(a.track, GainAutomation(breakpoints=bp),
                            summary=f"gain_auto[{a.track}] {len(bp)} pts")


def apply_dynamic_eq(state: MixState, args: dict) -> MixState:
    a = S.ApplyDynamicEQArgs(**args)
    return state.add_static(a.track,
                            DynamicEQ(freq=a.freq, threshold_db=a.threshold_db,
                                      gain_db=a.gain_db, q=a.q,
                                      attack_ms=a.attack_ms, release_ms=a.release_ms),
                            summary=f"dyn_eq[{a.track}] f={a.freq:.0f} thr={a.threshold_db:.1f} g={a.gain_db:+.1f}")


def apply_multiband_compressor(state: MixState, args: dict) -> MixState:
    a = S.ApplyMultibandCompArgs(**args)
    return state.add_static(
        a.track,
        MultibandComp(
            crossover_lo=a.crossover_lo, crossover_hi=a.crossover_hi,
            low_threshold_db=a.low_threshold_db, low_ratio=a.low_ratio,
            low_attack_ms=a.low_attack_ms, low_release_ms=a.low_release_ms,
            mid_threshold_db=a.mid_threshold_db, mid_ratio=a.mid_ratio,
            mid_attack_ms=a.mid_attack_ms, mid_release_ms=a.mid_release_ms,
            high_threshold_db=a.high_threshold_db, high_ratio=a.high_ratio,
            high_attack_ms=a.high_attack_ms, high_release_ms=a.high_release_ms,
        ),
        summary=f"mb_comp[{a.track}] xover={a.crossover_lo:.0f}/{a.crossover_hi:.0f}",
    )


def apply_sidechain_compressor(state: MixState, args: dict) -> MixState:
    a = S.ApplySidechainCompArgs(**args)
    if a.sidechain_track not in state.stems:
        raise KeyError(f"sidechain_track {a.sidechain_track!r} not in stems")
    return state.add_static(
        a.track,
        SidechainComp(sidechain_track=a.sidechain_track,
                      threshold_db=a.threshold_db, ratio=a.ratio,
                      attack_ms=a.attack_ms, release_ms=a.release_ms),
        summary=f"sc_comp[{a.track}<-{a.sidechain_track}] thr={a.threshold_db:.1f} R={a.ratio:.1f}",
    )


def apply_reverb(state: MixState, args: dict) -> MixState:
    a = S.ApplyReverbArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    return state.add_static(
        a.track,
        Reverb(room_size=a.room_size, damping=a.damping,
               wet_level=a.wet_level, dry_level=a.dry_level,
               width=a.width, highpass_hz=a.highpass_hz),
        summary=f"reverb[{a.track}] room={a.room_size:.2f} wet={a.wet_level:.2f}",
    )


def apply_delay(state: MixState, args: dict) -> MixState:
    a = S.ApplyDelayArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    return state.add_static(
        a.track,
        Delay(delay_seconds=a.delay_seconds, feedback=a.feedback, mix=a.mix),
        summary=f"delay[{a.track}] t={a.delay_seconds:.3f}s fb={a.feedback:.2f} mix={a.mix:.2f}",
    )


def apply_saturation(state: MixState, args: dict) -> MixState:
    a = S.ApplySaturationArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    return state.add_static(
        a.track,
        Saturation(drive_db=a.drive_db),
        summary=f"saturation[{a.track}] drive={a.drive_db:.1f}dB",
    )


def apply_deesser(state: MixState, args: dict) -> MixState:
    a = S.ApplyDeEsserArgs(**args)
    if a.track not in state.stems:
        raise KeyError(f"track {a.track!r} not in stems")
    return state.add_static(
        a.track,
        DeEsser(center_hz=a.center_hz, threshold_db=a.threshold_db,
                ratio=a.ratio, range_db=a.range_db,
                attack_ms=a.attack_ms, release_ms=a.release_ms),
        summary=f"deesser[{a.track}] f={a.center_hz:.0f} thr={a.threshold_db:.1f} range={a.range_db:.1f}dB",
    )


ACTION_HANDLERS = {
    "apply_static_eq":         apply_static_eq,
    "apply_static_compressor": apply_static_compressor,
    "apply_static_gain":       apply_static_gain,
    "apply_static_pan":        apply_static_pan,
    "apply_static_width":      apply_static_width,
    "apply_section_gain":      apply_section_gain,
    "apply_section_eq":        apply_section_eq,
    "apply_section_pan":       apply_section_pan,
    "apply_section_width":     apply_section_width,
    "apply_section_compressor": apply_section_compressor,
    "apply_master_eq":         apply_master_eq,
    "apply_master_compressor": apply_master_compressor,
    "apply_master_limiter":    apply_master_limiter,
    "apply_transient_shaper":  apply_transient_shaper,
    "apply_gain_automation":   apply_gain_automation,
    "apply_pan_automation":    apply_pan_automation,
    "apply_width_automation":  apply_width_automation,
    "apply_dynamic_eq":        apply_dynamic_eq,
    "apply_multiband_compressor": apply_multiband_compressor,
    "apply_sidechain_compressor": apply_sidechain_compressor,
    "apply_reverb":            apply_reverb,
    "apply_delay":             apply_delay,
    "apply_saturation":        apply_saturation,
    "apply_deesser":           apply_deesser,
}
