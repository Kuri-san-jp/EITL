"""RuleAdvisor -- advisory layer that turns ear scores into concrete DSP ToolCalls.

Core of the design:
  From the absolute values returned by each ear and their deltas against the
  previous turn, rules generate the DSP operations that should be applied
  (in ToolCall form) and hand them to the orchestrator's LLM prompt as
  "expert advice". The LLM is free to adopt, modify, or reject that advice.

This secures a path in which ear deltas feed directly into actual parameter
control, instead of relying on the LLM's judgement alone (the core of the
novelty claim for the ICASSP submission).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Pair of the concrete ToolCall a rule proposes (name + arguments) and why it
# was proposed (a human-readable message).
@dataclass
class Suggestion:
    tool_call: Dict[str, Any]   # {"name": "...", "arguments": {...}}
    reason: str                 # "lufs is -19 LU, far from the -14 target"
    priority: int = 5           # 1=urgent (clipping etc.), 5=normal, 9=do last
    source_ear: str = ""        # which ear produced this suggestion


@dataclass
class AdviceReport:
    """All advice for one turn + which numbers it was derived from."""
    suggestions: List[Suggestion] = field(default_factory=list)
    flagged_signals: Dict[str, float] = field(default_factory=dict)

    def sorted_by_priority(self) -> List[Suggestion]:
        return sorted(self.suggestions, key=lambda s: s.priority)

    def to_prompt_block(self, max_items: int = 6) -> str:
        """Build the advice block to be inserted into the LLM prompt."""
        if not self.suggestions:
            return "(No strong advice. Proceed at the LLM's own discretion.)"
        items = self.sorted_by_priority()[:max_items]
        lines = []
        for i, s in enumerate(items, 1):
            args_str = ", ".join(f"{k}={v}" for k, v in s.tool_call["arguments"].items())
            lines.append(
                f"  {i}. [{s.source_ear}] {s.reason}\n"
                f"     -> suggestion: {s.tool_call['name']}({args_str})"
            )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Individual rules
# ----------------------------------------------------------------------


def _r(score_flat: Dict[str, float], key: str, default: Optional[float] = None) -> Optional[float]:
    """Fetch a value from the flat score dict (NaN is treated as None)."""
    v = score_flat.get(key)
    if v is None:
        return default
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return default
    if fv != fv:                 # NaN check
        return default
    return fv


def rule_safety_true_peak(scores: Dict[str, float]) -> List[Suggestion]:
    """Pull the master limiter down when true peak exceeds -1 dBTP."""
    tp = _r(scores, "true_peak.true_peak_db")
    if tp is None or tp <= -1.0:
        return []
    ceiling = round(min(-1.5, tp - 1.0), 2)
    return [Suggestion(
        tool_call={"name": "apply_master_limiter",
                   "arguments": {"ceiling_db": ceiling, "release_ms": 100}},
        reason=f"true peak is {tp:+.2f} dBTP, inside the clipping region (limit -1 dBTP)",
        priority=1,
        source_ear="true_peak",
    )]


def rule_safety_loudness(scores: Dict[str, float], target_lufs: float = -14.0,
                         tolerance: float = 3.0) -> List[Suggestion]:
    """Pull LUFS back with master gain when it is far off the target."""
    lufs = _r(scores, "lufs.integrated_lufs")
    if lufs is None:
        return []
    delta = lufs - target_lufs
    if abs(delta) < tolerance:
        return []
    if delta > 0:
        # too loud -> hold it back with the limiter
        return [Suggestion(
            tool_call={"name": "apply_master_limiter",
                       "arguments": {"ceiling_db": -1.0, "release_ms": 100}},
            reason=f"LUFS is {lufs:.1f} (target {target_lufs}), {delta:+.1f} LU too loud",
            priority=2,
            source_ear="lufs",
        )]
    else:
        # too quiet -> lift the highest-energy stem
        return [Suggestion(
            tool_call={"name": "apply_master_eq",
                       "arguments": {"freq": 80, "gain_db": 1.5, "q": 0.7}},
            reason=f"LUFS is {lufs:.1f} (target {target_lufs}), {delta:+.1f} LU short; lifting the low end is one option",
            priority=4,
            source_ear="lufs",
        )]


def rule_safety_stereo(scores: Dict[str, float]) -> List[Suggestion]:
    """Pull width back toward the centre when stereo correlation swings strongly negative."""
    rho = _r(scores, "stereo.stereo_correlation")
    if rho is None or rho >= -0.1:
        return []
    return [Suggestion(
        tool_call={"name": "apply_static_width",
                   "arguments": {"track": "__widest__", "width": 0.6}},
        reason=f"stereo correlation is {rho:+.2f}, risk of phase cancellation (track is meant to be the widest stem)",
        priority=2,
        source_ear="stereo",
    )]


def rule_audiobox_axes(prev: Dict[str, float], cur: Dict[str, float]) -> List[Suggestion]:
    """Propose a master EQ move per Audiobox axis (PQ/PC/CE/CU) that dropped."""
    out: List[Suggestion] = []
    axis_to_advice = {
        # axis name -> (centre frequency, expected effect)
        "PQ": (80,   "a low shelf that adds overall weight and thickness"),
        "PC": (8000, "a high shelf that brings brightness and air"),
        "CE": (2500, "an upper-mid boost that pushes voice and instruments forward"),
        "CU": (250,  "a low-mid cut that clears the mud (negative direction)"),
    }
    for axis, (freq, desc) in axis_to_advice.items():
        cur_v = _r(cur, f"audiobox.{axis}")
        prev_v = _r(prev, f"audiobox.{axis}")
        if cur_v is None or prev_v is None:
            continue
        delta = cur_v - prev_v
        if delta < -0.05:                 # dropped by 0.05 points or more
            gain = -2.0 if axis == "CU" else 1.5
            out.append(Suggestion(
                tool_call={"name": "apply_master_eq",
                           "arguments": {"freq": freq, "gain_db": gain, "q": 0.7}},
                reason=f"Audiobox {axis} dropped by {delta:+.3f} ({prev_v:.2f}->{cur_v:.2f}); recover with {desc} to compensate",
                priority=5,
                source_ear=f"audiobox.{axis}",
            ))
    return out


def rule_clap_drift(scores: Dict[str, float], min_cosine: float = 0.85) -> List[Suggestion]:
    """Push for a rollback when CLAP cosine similarity drifts far from the initial mix."""
    cos = _r(scores, "clap.clap_cosine")
    if cos is None or cos >= min_cosine:
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"CLAP similarity is {cos:.2f}, a large deviation from the initial mix (suspected style drift)",
        priority=2,
        source_ear="clap",
    )]


def rule_mert_drift(scores: Dict[str, float], max_distance: float = 0.3) -> List[Suggestion]:
    """A large increase in MERT embedding distance suggests the colour of the track has been broken."""
    dist = _r(scores, "mert.mert_distance")
    if dist is None or dist <= max_distance:
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"MERT distance is {dist:.2f}; harmony and timbre may have changed a lot from the start",
        priority=3,
        source_ear="mert",
    )]


def rule_vocal_intelligibility(scores: Dict[str, float], stems: List[str]) -> List[Suggestion]:
    """Lift EQ / gain on the vocal track when vocal intelligibility is low."""
    inte = _r(scores, "vocal_intelligibility.intelligibility")
    wer = _r(scores, "vocal_intelligibility.wer")
    vocal_track = next((s for s in stems if "vocal" in s.lower()), None)
    if vocal_track is None:
        return []
    out = []
    if inte is not None and inte < 0.5:
        out.append(Suggestion(
            tool_call={"name": "apply_static_eq",
                       "arguments": {"track": vocal_track, "freq": 3000, "gain_db": 2.0, "q": 1.0}},
            reason=f"vocal intelligibility {inte:.2f} is low; +2 dB on the 3 kHz presence band",
            priority=4,
            source_ear="vocal_intelligibility",
        ))
    if wer is not None and wer > 0.3:
        out.append(Suggestion(
            tool_call={"name": "apply_static_gain",
                       "arguments": {"track": vocal_track, "gain_db": 1.5}},
            reason=f"vocal WER is high at {wer:.2f}; lift the vocal by +1.5 dB",
            priority=4,
            source_ear="vocal_intelligibility",
        ))
    return out


def rule_mos_consensus_low(scores: Dict[str, float],
                           min_avg: float = 2.5) -> List[Suggestion]:
    """Propose a master-stage adjustment when the mean of the Tier 1 MOS family
    (NISQA/UTMOS/DNSMOS/SingMOS) is low. Several MOS being low at once is
    roughly broad-band quality degradation, so add brightness with a master_eq
    high shelf."""
    vals: List[float] = []
    for key in ("nisqa", "utmos", "dnsmos.ovrl_mos", "singmos"):
        v = _r(scores, key)
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return []
    avg = sum(vals) / len(vals)
    if avg >= min_avg:
        return []
    return [Suggestion(
        tool_call={"name": "apply_master_eq",
                   "arguments": {"freq": 10000, "gain_db": 1.5, "q": 0.7}},
        reason=f"Tier 1 MOS mean {avg:.2f} ({len(vals)} axes) is below {min_avg}; "
               "raise overall clarity with a +1.5 dB high shelf",
        priority=3,
        source_ear="mos_consensus",
    )]


def rule_dnsmos_bg_noise(scores: Dict[str, float]) -> List[Suggestion]:
    """When DNSMOS bak_mos is low (a lot of background noise), push for a
    noise-gate / downward-expander style treatment on the master."""
    bak = _r(scores, "dnsmos.bak_mos")
    if bak is None or bak >= 3.0:
        return []
    return [Suggestion(
        tool_call={"name": "apply_master_compressor",
                   "arguments": {"threshold_db": -30, "ratio": 1.5,
                                 "attack_ms": 10, "release_ms": 200}},
        reason=f"DNSMOS bak_mos {bak:.2f}, strong background murkiness; "
               "gain SNR with expander-leaning compression in the low-level region",
        priority=4,
        source_ear="dnsmos",
    )]


def rule_scoreq_low(scores: Dict[str, float],
                    min_quality: float = 0.5) -> List[Suggestion]:
    """When SCOREQ is low (quality degraded), make the recent operations a rollback candidate."""
    q = _r(scores, "scoreq")
    if q is None or q >= min_quality:
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"SCOREQ quality {q:.2f} is below {min_quality}; "
               "the recent operations may have hurt quality",
        priority=3,
        source_ear="scoreq",
    )]


def rule_visqol_drop(prev: Dict[str, float],
                     cur: Dict[str, float]) -> List[Suggestion]:
    """Rollback candidate when ViSQOL MOSLQO drops noticeably from the previous turn."""
    p = _r(prev, "visqol_music.moslqo")
    c = _r(cur, "visqol_music.moslqo")
    if p is None or c is None:
        return []
    delta = c - p
    if delta >= -0.2:
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"ViSQOL MOSLQO dropped by {delta:+.2f} ({p:.2f}->{c:.2f}); "
               "large perceptual degradation",
        priority=2,
        source_ear="visqol_music",
    )]


def rule_cdpam_drift(scores: Dict[str, float],
                     max_distance: float = 0.5) -> List[Suggestion]:
    """A large CDPAM distance is roughly a large perceptual distance from the
    original mix. Over-processing is suspected, so push for a rollback."""
    d = _r(scores, "cdpam.cdpam_distance")
    if d is None or d <= max_distance:
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"CDPAM distance {d:.2f} exceeds {max_distance}; "
               "perceptually far away now",
        priority=3,
        source_ear="cdpam",
    )]


def rule_fad_drift(scores: Dict[str, float],
                   max_fad: float = 5.0) -> List[Suggestion]:
    """A large increase in FAD means we have strayed too far from the reference distribution."""
    fad = _r(scores, "fad.fad")
    if fad is None or fad <= max_fad:
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"FAD {fad:.2f} is over {max_fad}, too far from the reference distribution "
               "(suspected over-processing)",
        priority=3,
        source_ear="fad",
    )]


def rule_qwen2_judge_low(scores: Dict[str, float],
                          min_score: float = 5.0) -> List[Suggestion]:
    """When the Qwen2-Audio judge rates the mix low on its own (< 5/10), emit a
    suggestion that reinforces the weakest audiobox axis."""
    s = _r(scores, "qwen2_audio_judge") or _r(scores, "qwen2_audio_judge.score")
    if s is None or s >= min_score:
        return []
    # find the weakest audiobox axis and reinforce it
    axes = {ax: _r(scores, f"audiobox.{ax}") for ax in ("PQ", "PC", "CE", "CU")}
    axes = {k: v for k, v in axes.items() if v is not None}
    if not axes:
        return []
    weakest = min(axes.items(), key=lambda kv: kv[1])[0]
    freq_map = {"PQ": 80, "PC": 8000, "CE": 2500, "CU": 250}
    freq = freq_map[weakest]
    gain = -2.0 if weakest == "CU" else 1.5
    return [Suggestion(
        tool_call={"name": "apply_master_eq",
                   "arguments": {"freq": freq, "gain_db": gain, "q": 0.7}},
        reason=f"Qwen2-Audio judge rates it low at {s:.1f}/10; "
               f"audiobox {weakest} ({axes[weakest]:.2f}) is the weakest, so "
               f"reinforce {freq} Hz by {gain:+.1f} dB",
        priority=4,
        source_ear="qwen2_audio_judge",
    )]


def rule_genre_drift(prev: Dict[str, float],
                     cur: Dict[str, float]) -> List[Suggestion]:
    """A large drop in genre confidence is roughly a sign the genre character may have collapsed."""
    p = _r(prev, "genre.genre_confidence")
    c = _r(cur, "genre.genre_confidence")
    if p is None or c is None:
        return []
    if c >= p - 0.15:
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"Genre confidence fell {p:.2f}->{c:.2f} (Δ{c-p:+.2f}); "
               "the genre character may have been broken",
        priority=3,
        source_ear="genre",
    )]


def rule_reseparation_drop(prev: Dict[str, float],
                           cur: Dict[str, float]) -> List[Suggestion]:
    """A large drop in Demucs re-separation SDR is roughly a sign the stems are bleeding together."""
    p = _r(prev, "reseparation_sdr.sdr_mean")
    c = _r(cur, "reseparation_sdr.sdr_mean")
    if p is None or c is None:
        return []
    if c >= p - 1.0:        # be wary once it drops by 1 dB or more
        return []
    return [Suggestion(
        tool_call={"name": "rollback", "arguments": {"target_state_id": "best"}},
        reason=f"re-separation SDR fell {p:.1f} dB -> {c:.1f} dB; "
               "the stem images are bleeding together (smearing)",
        priority=3,
        source_ear="reseparation_sdr",
    )]


def rule_lra_excess(scores: Dict[str, float],
                    min_lra: float = 4.0,
                    max_lra: float = 15.0) -> List[Suggestion]:
    """Loudness Range too narrow is roughly over-compression / too wide is roughly excessive dynamics."""
    lra = _r(scores, "lra.lra")
    if lra is None:
        return []
    if lra < min_lra:
        return [Suggestion(
            tool_call={"name": "apply_master_compressor",
                       "arguments": {"threshold_db": -18, "ratio": 1.2,
                                     "attack_ms": 30, "release_ms": 300}},
            reason=f"LRA {lra:.1f} LU is narrow (suspected over-compression); "
                   "try a master compressor that loosens the compression ratio",
            priority=4,
            source_ear="lra",
        )]
    if lra > max_lra:
        return [Suggestion(
            tool_call={"name": "apply_master_compressor",
                       "arguments": {"threshold_db": -20, "ratio": 2.0,
                                     "attack_ms": 15, "release_ms": 150}},
            reason=f"LRA {lra:.1f} LU is too wide; tidy up the dynamics with a "
                   "master compressor",
            priority=5,
            source_ear="lra",
        )]
    return []


def rule_section_variance(section_scores: Dict[str, Dict[str, float]]) -> List[Suggestion]:
    """Recommend a section-scoped operation when score variance across sections is large."""
    if not section_scores:
        return []
    # measure the spread using Audiobox.PQ as the representative value
    vals = []
    weakest = None
    weakest_v = None
    for sec_label, flat in section_scores.items():
        v = _r(flat, "audiobox.PQ")
        if v is None:
            continue
        vals.append(v)
        if weakest_v is None or v < weakest_v:
            weakest = sec_label
            weakest_v = v
    if len(vals) < 2 or weakest is None:
        return []
    var = max(vals) - min(vals)
    if var < 0.15:
        return []
    return [Suggestion(
        tool_call={"name": "apply_section_eq",
                   "arguments": {"track": "__weakest_in_section__",
                                 "section_label": weakest,
                                 "freq": 2500, "gain_db": 1.5, "q": 0.7}},
        reason=f"audiobox.PQ spreads {var:.2f} across sections; "
               f"lift the weakest section '{weakest}' ({weakest_v:.2f}) with a section EQ",
        priority=5,
        source_ear="section_variance",
    )]


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


class RuleAdvisor:
    """Takes the previous and current score dicts and returns an AdviceReport."""

    def __init__(self, target_lufs: float = -14.0):
        self.target_lufs = target_lufs

    def advise(self,
               cur_flat: Dict[str, float],
               prev_flat: Optional[Dict[str, float]] = None,
               section_scores: Optional[Dict[str, Dict[str, float]]] = None,
               stems: Optional[List[str]] = None,
               ) -> AdviceReport:
        prev_flat = prev_flat or {}
        stems = stems or []
        section_scores = section_scores or {}

        suggestions: List[Suggestion] = []
        # ----- existing: safety + audiobox delta + reference-based drift + section -----
        suggestions.extend(rule_safety_true_peak(cur_flat))
        suggestions.extend(rule_safety_loudness(cur_flat, self.target_lufs))
        suggestions.extend(rule_safety_stereo(cur_flat))
        suggestions.extend(rule_audiobox_axes(prev_flat, cur_flat))
        suggestions.extend(rule_clap_drift(cur_flat))
        suggestions.extend(rule_mert_drift(cur_flat))
        suggestions.extend(rule_vocal_intelligibility(cur_flat, stems))
        suggestions.extend(rule_section_variance(section_scores))
        # ----- added: advice from Tier 1 MOS / Tier 2 ref / Tier 3 judge /
        #         Tier 4 dyn / Tier 5 task -----
        suggestions.extend(rule_mos_consensus_low(cur_flat))
        suggestions.extend(rule_dnsmos_bg_noise(cur_flat))
        suggestions.extend(rule_scoreq_low(cur_flat))
        suggestions.extend(rule_visqol_drop(prev_flat, cur_flat))
        suggestions.extend(rule_cdpam_drift(cur_flat))
        suggestions.extend(rule_fad_drift(cur_flat))
        suggestions.extend(rule_qwen2_judge_low(cur_flat))
        suggestions.extend(rule_genre_drift(prev_flat, cur_flat))
        suggestions.extend(rule_reseparation_drop(prev_flat, cur_flat))
        suggestions.extend(rule_lra_excess(cur_flat))

        flagged = {
            "true_peak_db":          cur_flat.get("true_peak.true_peak_db", float("nan")),
            "integrated_lufs":       cur_flat.get("lufs.integrated_lufs", float("nan")),
            "stereo_correlation":    cur_flat.get("stereo.stereo_correlation", float("nan")),
            "lra":                   cur_flat.get("lra.lra", float("nan")),
            "clap_cosine":           cur_flat.get("clap.clap_cosine", float("nan")),
            "mert_distance":         cur_flat.get("mert.mert_distance", float("nan")),
            "visqol_moslqo":         cur_flat.get("visqol_music.moslqo", float("nan")),
            "cdpam_distance":        cur_flat.get("cdpam.cdpam_distance", float("nan")),
            "fad":                   cur_flat.get("fad.fad", float("nan")),
            "nisqa":                 cur_flat.get("nisqa", float("nan")),
            "utmos":                 cur_flat.get("utmos", float("nan")),
            "dnsmos_ovrl":           cur_flat.get("dnsmos.ovrl_mos", float("nan")),
            "singmos":               cur_flat.get("singmos", float("nan")),
            "scoreq":                cur_flat.get("scoreq", float("nan")),
            "qwen2_judge":           cur_flat.get("qwen2_audio_judge", float("nan")),
            "vocal_intelligibility": cur_flat.get(
                "vocal_intelligibility.intelligibility", float("nan")),
            "genre_confidence":      cur_flat.get("genre.genre_confidence", float("nan")),
            "reseparation_sdr":      cur_flat.get("reseparation_sdr.sdr_mean", float("nan")),
        }
        return AdviceReport(suggestions=suggestions, flagged_signals=flagged)
