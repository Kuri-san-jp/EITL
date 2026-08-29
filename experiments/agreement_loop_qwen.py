"""Agreement-reward (R = min(z_PQ, z_SB)) driven mixing loop — step 2, part 2: local Qwen driven.

Part 1 of step 2 (agreement_loop_v2.py) ran _propose_action with random/greedy and
demonstrated that the machinery — the extended 24-handler space, eval-gating, in-loop
full-song rendering, the min agreement reward, and partial saving — runs without
breaking on all 9 kinds. This runner replaces that _propose_action with tool-call
proposals from **a local Qwen (Qwen2.5-32B-Instruct-AWQ, vLLM OpenAI-compatible
server)**. Everything else (calibration / render / scoring / eval-gate / partial save /
cost aggregation / per-stem LUFS observation) is kept identical to v2.

Qwen proposal logic (following the "next-stage findings" in the internal notes):
  - At each iter, hand Qwen the context:
      (a) per-stem role / current LUFS (vocals/bass/drums/other)
      (b) current PQ / SongBench-Mixing / reward (= min(z_PQ, z_SB)) and z values
      (c) operations **accepted** / **rejected** so far and their reward effect (Δreward)
      (d) bias hints: reverb / saturation tend to lower Audiobox-PQ (all rejected in v2)
  - Qwen proposes the next **single operation** as a tool-call (24-action schema).
  - eval-gated: stack the proposal onto best_state, render the full song → PQ + SB → reward.
    Accept only when the reward improves (the same greedy gate as v2).
  - **de-esser gate**: even when Qwen proposes apply_deesser, reject it before rendering
    if the sibilance-band (5-9 kHz) energy ratio of the target vocal stem is low
    (a de-esser render takes ~11s; do not waste it on proposals against sibilance that
    does not exist).

Qwen server (co-location):
  IMAGE_NAME=vllm/vllm-openai:latest NUM_GPUS=1 JOB_TIME=03:00:00 \
    scripts/the local job wrapper python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-32B-Instruct-AWQ --host 0.0.0.0 --port 8000 \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --max-model-len 32768 --served-model-name qwen ...
  Startup check: curl http://127.0.0.1:8000/v1/models
  Because this is a single node (a single node) and the local job wrapper uses --network=host,
  127.0.0.1:<port> is reachable from another container (loop=songbench-image) too.

Running the loop (this runner) (PQ=audiobox + SongBench=MuQ; GPU required):
  LOCAL_LLM_URL=http://127.0.0.1:8000/v1 LOCAL_LLM_MODEL=qwen \
  IMAGE_NAME=songbench-image:latest JOB_TIME=01:30:00 \
    scripts/the local job wrapper python experiments/agreement_loop_qwen.py \
      --dataset musdb18 --split train --song-index 0 --n-steps 12

Hard policy (memory: feedback_no_proxy_no_experiment): no proxies allowed.
If the real PQ/SB models cannot be loaded, stop with an exception.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# experiments/_common puts ROOT/src on sys.path and configures the HF cache.
from _common import ROOT, load_one, build_llm  # noqa: E402

from mix_orchestrator.dsp.mix_state import MixState  # noqa: E402
from mix_orchestrator.dsp.renderer import render  # noqa: E402
# _measure_lufs is unused here now that per-stem LUFS moved into stem_lufs_cache, but
# it is kept because external code may do `from agreement_loop_qwen import _measure_lufs`.
from mix_orchestrator.dsp.loudness_norm import _measure_lufs, normalize_for_eval  # noqa: E402,F401
from mix_orchestrator.dsp.stem_lufs_cache import (  # noqa: E402,F401
    StemLufsCache, compute_per_stem_lufs,
)
from mix_orchestrator.tools import action_tools as A  # noqa: E402
from mix_orchestrator.tools.schemas import TOOL_CATALOG, anthropic_tool_specs  # noqa: E402
# classify_role / _render_single_stem are likewise unused here (stem_lufs_cache uses them).
# They are kept as re-exports for backward compatibility.
from mix_orchestrator.strategies.knowledge_base_mix import (  # noqa: E402,F401
    build_knowledge_base_mix, classify_role, _render_single_stem,
)


OUT_DIR = ROOT / "outputs" / "runs" / "agreement_loop_qwen"

# Restrict the actions Qwen can propose to the same static/insert space as v2.
# section/master/automation/dynamic/transient/perception/state carry little meaning in
# full-song 1-shot greedy (they need temporal structure / sidechaining), so they are
# excluded from the tool list to cut down on wasted proposals.
_ALLOWED_ACTIONS = [
    "apply_static_gain", "apply_static_pan", "apply_static_width",
    "apply_static_eq", "apply_static_compressor",
    "apply_reverb", "apply_delay", "apply_saturation", "apply_deesser",
]

# ---------------------------------------------------------------------------
# Switching the action space (added ; the default stays the 9 kinds above,
# not a single character changes)
# ---------------------------------------------------------------------------
#: The exclusion reason in the comment above — "automation carries little meaning in
#: full-song 1-shot greedy (it needs temporal structure)" — was **a judgement made under
#: the assumption of full-song search**. Now that search runs on 12-second excerpts the
#: assumption changes:
#:
#:   - 12 seconds is less than one section, so section_* still carries no meaning, but
#:     gain automation does work as "move it in time with the tempo inside 12 seconds".
#:   - The excerpt window is cut snapped to the beat grid (beat_snap in select_excerpts),
#:     so the start of the excerpt = a beat. Given the beat times you can write
#:     beat-synchronous automation.
#:
#: Switched with ``AUTOMIX_ACTION_SET``:
#:   base       : only the 9 kinds above (default; bit-identical to existing runs)
#:   automation : the 9 kinds + apply_gain_automation, i.e. 10 kinds
#:
#: **Important (cannot be transplanted)**: automation breakpoints carry absolute times,
#: so a chain obtained on an excerpt cannot be transplanted onto full-song stems
#: (agreement_loop_all.swap_state_stems refuses it explicitly).
#: Runs with automation enabled report only the 12-second excerpt results.
#: See the discussion of agreement_loop_all.EXCERPT_ONLY_ENV for details.
ACTION_SET_ENV = "AUTOMIX_ACTION_SET"
ACTION_SETS: Dict[str, List[str]] = {
    "base": [],
    "automation": ["apply_gain_automation"],
}
#: Actions that carry absolute times (breakpoints), so an excerpt chain cannot be
#: transplanted onto the full song.
TIME_ABSOLUTE_ACTIONS = frozenset({
    "apply_gain_automation", "apply_pan_automation", "apply_width_automation",
})


def action_set_default() -> str:
    """Read the ``AUTOMIX_ACTION_SET`` environment variable. Unset means 'base' (legacy behavior)."""
    v = (os.environ.get(ACTION_SET_ENV) or "base").strip().lower()
    if v not in ACTION_SETS:
        raise ValueError(f"unknown {ACTION_SET_ENV}={v!r}; "
                         f"valid={tuple(ACTION_SETS)}")
    return v


def allowed_actions(action_set: Optional[str] = None) -> List[str]:
    """List of action names shown to the LLM in this run.

    If ``action_set`` is None, use the environment-variable default
    (= 'base' = the legacy 9 kinds). Additions are **appended at the end**, so the
    relative order of the existing 9 kinds does not change (to avoid confounding with
    the position-bias experiment in ``AUTOMIX_TOOL_ORDER``).
    """
    mode = action_set_default() if action_set is None else str(action_set)
    if mode not in ACTION_SETS:
        raise ValueError(f"unknown action_set={mode!r}; valid={tuple(ACTION_SETS)}")
    return list(_ALLOWED_ACTIONS) + list(ACTION_SETS[mode])


def action_set_is_time_absolute(action_set: Optional[str] = None) -> bool:
    """Whether this action set contains actions with absolute times (= cannot be transplanted to the full song)."""
    return any(a in TIME_ABSOLUTE_ACTIONS for a in allowed_actions(action_set))


# ============================================================================
# tool specs (the actions shown to Qwen) and schema validation
# ============================================================================
#: Environment variable that switches the ordering of the tool list (added .
#:
#: **Why it is needed**: in TOOL_CATALOG's declaration order apply_static_eq comes
#: first and apply_static_gain third. Measured, Qwen filled almost all of 10000 moves
#: with just those two kinds and never once proposed the remaining 7.
#: Only the order is changed, to isolate whether position bias is the cause.
#:
#: **Not a single character of the prompt body changes**, so the disclosure in the
#: paper is just "we changed the order in which the tools are presented".
#:
#:   catalog : TOOL_CATALOG's declaration order (default; bit-identical to existing runs)
#:   reverse : reversed order; eq/gain end up at the end
#:   rotate  : rotate per step, so no action stays at a fixed position
#:   alpha   : alphabetical by name; the least arbitrary choice
TOOL_ORDER_ENV = "AUTOMIX_TOOL_ORDER"
TOOL_ORDER_MODES = ("catalog", "reverse", "rotate", "alpha")


def tool_order_default() -> str:
    v = (os.environ.get(TOOL_ORDER_ENV) or "catalog").strip().lower()
    return v if v in TOOL_ORDER_MODES else "catalog"


def _allowed_tool_specs(order: Optional[str] = None,
                        step: int = 0,
                        action_set: Optional[str] = None) -> List[Dict[str, Any]]:
    """Show Qwen only the allowed actions out of anthropic_tool_specs().

    ``order`` changes the ordering (default = catalog = as before).
    ``step`` is used only for ``rotate``, shifting the head per step.
    ``action_set`` (default None = the ``AUTOMIX_ACTION_SET`` environment variable,
    'base' when unset) switches the action space. With 'base' the return value is
    exactly identical to before.
    """
    allow = set(allowed_actions(action_set))
    specs = [s for s in anthropic_tool_specs() if s["name"] in allow]
    mode = tool_order_default() if order is None else order
    if mode == "reverse":
        return list(reversed(specs))
    if mode == "alpha":
        return sorted(specs, key=lambda s: s["name"])
    if mode == "rotate" and specs:
        k = int(step) % len(specs)
        return specs[k:] + specs[:k]
    return specs


#: Upper bound on the number of breakpoints per automation move (imposed on LLM
#: proposals only). Even at 12 seconds / 0.4 s per beat (=150 BPM) that is 31 points,
#: so 64 leaves plenty of headroom.
#: It is not put into the schema (ApplyGainAutomationArgs): pump7s in
#: explore_timevarying_v3, which targets full songs, grows its point count with song
#: length and reaches 64 points at 220 seconds, so putting the bound in the shared
#: schema would break existing scripts.
MAX_AUTOMATION_BREAKPOINTS = 64


def _validate_action(name: str, args: Dict[str, Any], tracks: List[str],
                     action_set: Optional[str] = None,
                     excerpt_sec: Optional[float] = None
                     ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Schema-validate Qwen's tool-call.

    Returns (ok, validated_args, reason).
      - ok=False if name is unknown / not allowed.
      - Validate against the pydantic schema (Field constraints). Out of range is ok=False.
      - ok=False if track is not an existing stem name.
      - For actions carrying ``breakpoints``, also check the point-count bound and
        whether at least one point falls inside the window (only when ``excerpt_sec``
        is passed).
    validated_args is the dict already coerced by pydantic (defaults filled in).

    ``action_set`` (default None = environment variable, 'base' when unset) and
    ``excerpt_sec`` (default None = no window check) both give **exactly the same
    behavior as before** at their defaults.
    """
    if name not in allowed_actions(action_set):
        return False, None, f"action {name!r} not in allowed set"
    spec = TOOL_CATALOG.get(name)
    if spec is None:
        return False, None, f"unknown action {name!r}"
    track = args.get("track")
    if track is not None and track not in tracks:
        return False, None, f"track {track!r} not in stems {tracks}"
    try:
        model = spec["args"](**args)
    except Exception as ex:                                       # noqa: BLE001
        return False, None, f"schema validation failed: {ex!r}"
    vargs = model.model_dump()
    bps = vargs.get("breakpoints")
    if bps is not None:
        # "ascending / t>=0 / value range" already passed on the schema side. What is
        # checked here is only the experiment-specific condition of **whether it is
        # meaningful with respect to the search window**.
        if len(bps) > MAX_AUTOMATION_BREAKPOINTS:
            return False, None, (
                f"breakpoints has {len(bps)} points; at most "
                f"{MAX_AUTOMATION_BREAKPOINTS} are allowed")
        if excerpt_sec is not None and float(excerpt_sec) > 0:
            # Outside the first/last breakpoint the interpolation is constant at the
            # endpoint value (np.interp's default). If the first point is past the end
            # of the window, the whole automation collapses to "constant at the first
            # gain value" = a plain static gain. That is meaningless for an action whose
            # point is time variation, so reject it.
            t0 = float(bps[0][0])
            if t0 >= float(excerpt_sec):
                return False, None, (
                    f"first breakpoint is at t={t0:.3f} s but the audio being "
                    f"mixed is only {float(excerpt_sec):.3f} s long, so the "
                    f"automation would be a constant gain; put breakpoints "
                    f"inside [0, {float(excerpt_sec):.3f}] seconds")
    return True, vargs, ""


# ============================================================================
# de-esser gate: cheaply measure the sibilance-band energy ratio of a vocal stem
# ============================================================================
def _sibilance_ratio(stem: np.ndarray, sr: int,
                     lo_hz: float = 5000.0, hi_hz: float = 9000.0) -> float:
    """Estimate 5-9 kHz band energy / total band energy with an rFFT (mono sum, single FFT).

    A pre-check to avoid the 11s de-esser render. If the value is small (< threshold),
    treat the sibilance as nonexistent and reject the de-esser proposal before rendering.
    """
    x = stem
    if x.ndim == 2:
        x = x.mean(axis=0)
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n < 16:
        return 0.0
    # A full FFT of a long song is expensive, and a central window of at most ~10 s is
    # enough (the band ratio is stationary).
    max_n = int(10.0 * sr)
    if n > max_n:
        start = (n - max_n) // 2
        x = x[start:start + max_n]
        n = max_n
    spec = np.abs(np.fft.rfft(x))
    power = spec * spec
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    total = float(power.sum()) + 1e-12
    band = float(power[(freqs >= lo_hz) & (freqs < hi_hz)].sum())
    return band / total


def _deesser_gate(state: MixState, args: Dict[str, Any], sr: int,
                  min_ratio: float) -> Tuple[bool, float, str]:
    """Whether to allow an apply_deesser proposal. Allowed if the target track's sibilance ratio >= min_ratio."""
    track = args.get("track")
    if track is None or track not in state.stems:
        return False, 0.0, "deesser target missing"
    ratio = _sibilance_ratio(state.stems[track], sr)
    if ratio < min_ratio:
        return False, ratio, (f"sibilance ratio {ratio:.4f} < {min_ratio:.4f} "
                              f"on {track!r}; skip de-esser (avoid ~11s render)")
    return True, ratio, ""


# ============================================================================
# Qwen proposal: build the context + obtain one tool-call
# ============================================================================
_SYSTEM_PROMPT = """You are a professional mixing engineer optimizing a multitrack \
music mix. At each step you propose EXACTLY ONE mixing action by calling ONE tool. \
A reward = min(z_PQ, z_SB) is computed by two independent models (Audiobox Production \
Quality and SongBench-Mixing); your action is kept only if the reward improves \
(eval-gated greedy). Make small, musically-motivated moves. Use the per-stem roles \
and loudness to decide what each stem needs. Call exactly one tool; do not output \
prose-only answers."""

# ---------------------------------------------------------------------------
# Guidance mode (C-3): how to compose the "prior knowledge" block of the prompt
# ---------------------------------------------------------------------------
# The default "legacy" **changes not a single character** (bit-identical to existing runs).
#
#   legacy : as it stands. Both system and user use the fixed wording
#            reward = min(z_PQ, z_SB), and Guidance is the 4 lines of
#            "observed tendencies on this reward".
#   objfix : align only the description of the reward with the actual --reward-form
#            (a bug fix). The Guidance block stays as in legacy. The only difference
#            from legacy is the description of the objective, so "how much the
#            misstatement of the objective mattered" can be isolated on its own.
#            **Known omission**: objfix fixes only the metrics line at the top; the
#            4th Guidance line "To raise the bottleneck metric, ..." stays as in legacy.
#            Therefore in runs with --reward-form pq_only/sb_only **a nonexistent
#            concept called the bottleneck keeps living in the prompt**.
#            This incompleteness is a bug, not an intention (confirmed on  by
#            rendering the actual prompt). To avoid breaking the reproducibility of
#            existing runs (guid_objfix_pq_off133, adopt_objfix_t10_s1234_off133, etc.),
#            objfix is **frozen as is** and the complete version is a separate mode,
#            objfix2.
#   objfix2: objfix plus aligning the bottleneck reference on the 4th Guidance line with
#            reward_form as well. When the reward is a single term (pq_only / sb_only),
#            "bottleneck metric" is undefined, so write "the reward" instead. For
#            min / weighted the bottleneck is a real concept, so keep the legacy wording.
#            The **content of the Guidance (which actions are better)** does not differ
#            from legacy by a single character. Only what the objective points at
#            changes. So legacy → objfix → objfix2 is a nested series in which only
#            "the correctness of the description of the objective" moves.
#   task   : objfix plus **removing from the prompt everything that ranks actions based
#            on measurements**. Instead pass only (a) the definition of the objective and
#            (b) the rules of the search loop (greedy acceptance, state unchanged on
#            rejection, deterministic render/scoring, the de-esser gate). (b) is a
#            specification of the environment and contains no information about "which
#            action is good", so repetition can be stopped without leaking the random
#            arm's results to the LLM (= answer leakage).
#
# **Why there is no mode that "rewrites the Guidance from measurements"**:
# The acceptance rate and the per-action reward gains are statistics obtained by running
# random for 220k moves on the same songs, the same reward and the same excerpts as the
# evaluation target. Writing those into the prompt would make the LLM's action
# distribution a distillation of the random arm's marginal, which trivially breaks the
# question "is the LLM better than random as a proposer". See the discussion in the note
# for details.
GUIDANCE_ENV = "AUTOMIX_GUIDANCE"
GUIDANCE_MODES = ("legacy", "objfix", "objfix2", "task")

# Modes that emit the legacy description of the objective (= ignore reward_form).
# Modes not listed here align the metrics line with reward_form.
_GUIDANCE_LEGACY_OBJ = ("legacy",)
# Modes that emit the Guidance block. 'task' does not (it drops all ranking statements).
_GUIDANCE_WITH_PRIORS = ("legacy", "objfix", "objfix2")
# reward_forms where the reward is a single term and the concept "bottleneck" does not exist.
_SINGLE_TERM_FORMS = ("pq_only", "sb_only")


def guidance_mode_default() -> str:
    """Read the ``AUTOMIX_GUIDANCE`` environment variable. Unset means 'legacy' (legacy behavior)."""
    v = (os.environ.get(GUIDANCE_ENV) or "legacy").strip().lower()
    if v not in GUIDANCE_MODES:
        raise ValueError(f"unknown {GUIDANCE_ENV}={v!r}; valid={GUIDANCE_MODES}")
    return v


def _reward_expr(reward_form: Optional[str]) -> str:
    """Return the formula notation of the reward corresponding to --reward-form."""
    return {"pq_only": "reward = z_PQ",
            "sb_only": "reward = z_SB",
            "weighted": "reward = (1-w)*z_PQ + w*z_SB",
            "min": "reward = min(z_PQ, z_SB)"}.get(reward_form or "min",
                                                   "reward = min(z_PQ, z_SB)")


# ---------------------------------------------------------------------------
# Family 4 (the removal direction), part 1: how much of the system prompt to strip
# ---------------------------------------------------------------------------
# The default 'full' **changes not a single character** (bit-identical to existing runs).
#
# Motivation: besides the search contract (one tool per move / accept only on
# improvement / no prose), the current system prompt contains a persona
# ("professional mixing engineer") and two pieces of advice
# ("Make small, musically-motivated moves." /
#  "Use the per-stem roles and loudness to decide what each stem needs.").
# The output when stalled is a **repetition of minuscule changes** — "step the bass EQ
# by 1 dB at a time, keeping q=2" — and the "small ... moves" instruction is suspected
# of inducing that repetition. Rather than adding an instruction to cancel it out,
# remove it and measure its contribution.
#
#   'full'    : as before (default).
#   'nosmall' : drop only the sentence "Make small, musically-motivated moves. ".
#               Nothing else changes by a single character, so the contribution of
#               that one sentence can be isolated on its own.
#   'minimal' : drop the persona and all advice, keeping only the contract with the
#               environment.
#
# The notation of the reward formula follows the guidance mode ('legacy' keeps legacy's
# (incorrect) formula), so the effect of "stripping the system prompt" and the effect of
# "fixing the misstated objective" are not confounded. None of the three modes contain
# any information about which actions are effective (if anything they remove advice, so
# there are fewer hand-written heuristics).
PROMPT_SYSTEM_ENV = "AUTOMIX_PROMPT_SYSTEM"
SYSTEM_MODES = ("full", "nosmall", "minimal")
# The one sentence dropped by 'nosmall'. It appears as the identical string in both the
# legacy constant and the objfix version.
_SMALL_MOVES_SENTENCE = "Make small, musically-motivated moves. "


def prompt_system_mode_default() -> str:
    """Read the ``AUTOMIX_PROMPT_SYSTEM`` environment variable. Unset means 'full' (legacy behavior)."""
    v = (os.environ.get(PROMPT_SYSTEM_ENV) or "full").strip().lower()
    if v not in SYSTEM_MODES:
        raise ValueError(f"unknown {PROMPT_SYSTEM_ENV}={v!r}; valid={SYSTEM_MODES}")
    return v


# ---------------------------------------------------------------------------
# Family 4 (the removal direction), part 2: drop the whole prior-knowledge block at the
# end of the design prompt
# ---------------------------------------------------------------------------
# The default False **changes not a single character** (bit-identical to existing runs).
#
# The current 4 Guidance lines — "reverb/saturation tend to lower PQ", "prefer
# level/pan/width/EQ" — are **rankings of actions**, and they amount to handing over by
# hand a part of the answer that the random arm needed 220k moves to obtain. Being
# consistent with the measurements is not an excuse; from the standpoint of fairness of
# the search it is on the illegitimate side.
# guidance='task' removes those 4 lines but **adds** the rules of the search loop
# instead, so the "removal" effect and the "addition" effect are confounded.
# drop_guidance only removes the block wholesale, so the removal effect can be measured
# on its own.
PROMPT_DROP_GUIDANCE_ENV = "AUTOMIX_PROMPT_DROP_GUIDANCE"


def prompt_drop_guidance_default() -> bool:
    """The ``AUTOMIX_PROMPT_DROP_GUIDANCE`` environment variable. Unset means False (legacy behavior)."""
    v = os.environ.get(PROMPT_DROP_GUIDANCE_ENV)
    if v is None:
        return False
    return v.strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Replacing the prompt from an external file (added 
# ---------------------------------------------------------------------------
# When putting prior knowledge derived from the 220k-move measurements into the prompt,
# the wording gets swapped many times. Editing this file each time **wipes out the
# running experiment, because the structure restarts python per song** (on ,
# 19 songs were in fact wiped out by a NameError). So the wording lives in a text file
# and the code is left alone; the code-side edit is finished with this single addition.
#
# File format (either section may be omitted):
#     # === SYSTEM ===
#     <the full text of the system prompt>
#     # === GUIDANCE ===
#     <the full text of the guidance block inserted into the design prompt>
#
# Omitting SYSTEM uses the legacy build_system_prompt. Omitting GUIDANCE uses the legacy
# guidance block (following the guidance mode). When GUIDANCE is given it **takes
# precedence over both** drop_guidance and the guidance mode.
#
# **If unset, the prompt does not change by a single character**
# (tools/check_prompt_bitexact.py).
# For reproducibility, the path and sha256 are printed to stdout at load time so they
# stay in the job log.
# ---------------------------------------------------------------------------
# Removing SongBench from the search (user instruction, 
# ---------------------------------------------------------------------------
# With reward_form=pq_only, SongBench has weight 0, so there is no need to score it on
# every move nor to show it in the prompt. When enabled, scoring in the search loop
# becomes PQ only (sb=nan) and the SongBench line and the system-prompt mention
# disappear from the prompt.
# **The final full-song scoring is left untouched, so the SongBench value for reporting
# is still measured once per song.**
# That number is needed as the counter-evidence against reward hacking (PQ +0.82 versus
# SB -0.064), and unlike per-move scoring it costs almost nothing, so it is kept.
# At the default (unset), neither the prompt nor the scoring changes by a single bit.
NO_SB_ENV = "AUTOMIX_NO_SB_IN_SEARCH"


def no_sb_in_search(default: bool = False) -> bool:
    """The ``AUTOMIX_NO_SB_IN_SEARCH`` environment variable. Unset means ``default``.

    **Calls with reward_form=pq_only pass default=True.**
    This addresses the design's point from . See the docstring of
    agreement_loop_all._no_sb_in_search for details.
    """
    v = os.environ.get(NO_SB_ENV)
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Saving the actual prompts (user instruction, 
# ---------------------------------------------------------------------------
# Keep **the very strings assembled during the real run**, not a "reconstruction".
# Pass a directory in the AUTOMIX_DUMP_PROMPT_DIR environment variable and the system
# prompt is written out once and the design prompt at the designated steps. Unset does
# nothing.
#
# A reconstruction cannot reproduce the per-stem role/LUFS (they are not kept in
# search_log). Keeping the real thing makes that problem go away.
DUMP_DIR_ENV = "AUTOMIX_DUMP_PROMPT_DIR"
#: Steps at which the design prompt is kept: one each from early, middle and late.
#: Keeping all of them would be enormous.
_DUMP_STEPS = (0, 1, 50, 120, 300)
_dumped: set = set()


def _dump_prompt(kind: str, text: str, step: Optional[int] = None) -> None:
    """Write out the prompt if AUTOMIX_DUMP_PROMPT_DIR is set.

    A failure does not stop the search (this is an incidental diagnostic feature, and
    killing a run over it would defeat the purpose).
    """
    d = (os.environ.get(DUMP_DIR_ENV) or "").strip()
    if not d:
        return
    key = f"{kind}_{step}"
    if key in _dumped:
        return
    _dumped.add(key)
    try:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        name = f"{kind}.txt" if step is None else f"{kind}_step{step:04d}.txt"
        (p / name).write_text(text, encoding="utf-8")
    except OSError:
        pass


GUIDANCE_FILE_ENV = "AUTOMIX_GUIDANCE_FILE"
_SEC_SYSTEM = "# === SYSTEM ==="
_SEC_GUIDANCE = "# === GUIDANCE ==="
#: This is called on every move, so read it once keyed by (path, mtime, size).
_guidance_file_cache: Dict[Tuple[str, float, int], Dict[str, str]] = {}


def guidance_file_path() -> Optional[str]:
    """The ``AUTOMIX_GUIDANCE_FILE`` environment variable. Unset or empty means None (legacy behavior)."""
    v = (os.environ.get(GUIDANCE_FILE_ENV) or "").strip()
    return v or None


def load_guidance_file(path: Optional[str] = None) -> Dict[str, str]:
    """Read the guidance file and return ``{'system': ..., 'guidance': ...}``.

    A missing section simply has no key. **Raise if the file is not found.**
    Silently running with the legacy prompt would produce a "run you thought you had
    swapped", indistinguishable from the JSON afterwards.
    """
    import hashlib

    p = guidance_file_path() if path is None else path
    if not p:
        return {}
    fp = Path(p)
    if not fp.is_absolute():
        fp = Path(ROOT) / p
    if not fp.is_file():
        raise FileNotFoundError(f"{GUIDANCE_FILE_ENV}={p!r} not found: {fp}")
    st = fp.stat()
    key = (str(fp), st.st_mtime, st.st_size)
    hit = _guidance_file_cache.get(key)
    if hit is not None:
        return hit
    raw = fp.read_text("utf-8")
    out: Dict[str, str] = {}
    cur: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        if cur is not None:
            out[cur] = "\n".join(buf).strip("\n")

    for line in raw.splitlines():
        s = line.strip()
        if s == _SEC_SYSTEM:
            flush()
            cur, buf = "system", []
        elif s == _SEC_GUIDANCE:
            flush()
            cur, buf = "guidance", []
        elif cur is not None:
            buf.append(line)
    flush()
    out = {k: v for k, v in out.items() if v.strip()}
    if not out:
        raise ValueError(f"{fp} has neither {_SEC_SYSTEM} nor {_SEC_GUIDANCE} section")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    print(f"[prompt] guidance_file={fp} sha256={digest} sections={sorted(out)}",
          flush=True)
    _guidance_file_cache[key] = out
    return out


def _drop_small_moves(text: str) -> str:
    """Drop "Make small, musically-motivated moves. " from the system prompt.

    Passing silently when the target is not found would produce a "run you thought you
    had stripped", so raise unless exactly one occurrence is found.
    """
    if text.count(_SMALL_MOVES_SENTENCE) != 1:
        raise ValueError("the sentence to remove was not found in the system prompt: "
                         f"{_SMALL_MOVES_SENTENCE!r}")
    return text.replace(_SMALL_MOVES_SENTENCE, "", 1)


def _minimal_system_prompt(reward_form: Optional[str], guidance: str) -> str:
    """A system prompt with the persona and advice dropped, keeping only the contract with the environment."""
    expr = ("reward = min(z_PQ, z_SB)" if guidance in _GUIDANCE_LEGACY_OBJ
            else _reward_expr(reward_form))
    return ("You propose EXACTLY ONE mixing action per step by calling ONE tool. "
            f"The action is applied to the mix, the mix is scored ({expr}), and "
            "the action is kept only if the reward improves. Call exactly one "
            "tool; do not output prose-only answers.")


def build_system_prompt(reward_form: Optional[str] = None,
                        guidance: Optional[str] = None,
                        system_mode: Optional[str] = None) -> str:
    """The system prompt for the given guidance mode. 'legacy' is the legacy string verbatim.

    'objfix' / 'objfix2' / 'task' return the same system prompt. The only error in the
    system prompt was the single sentence "reward = min(z_PQ, z_SB)", so the fix objfix2
    adds on top of objfix (the 4th Guidance line) exists only on the design-prompt side.

    ``system_mode`` (default None = the ``AUTOMIX_PROMPT_SYSTEM`` environment variable,
    'full' when unset = legacy behavior, byte-for-byte unchanged). For the meaning of
    'nosmall' / 'minimal' see the discussion around :data:`SYSTEM_MODES`.
    """
    _gf = load_guidance_file()
    if _gf.get("system"):
        # If the external file has a system section, that is the whole text.
        # system_mode ('nosmall' etc.) is not applied: the contract is that the file
        # side is the final form.
        _dump_prompt("system", _gf["system"])
        return _gf["system"]
    if guidance is None:
        guidance = guidance_mode_default()
    smode = (prompt_system_mode_default() if system_mode is None
             else str(system_mode).strip().lower())
    if smode not in SYSTEM_MODES:
        raise ValueError(f"unknown system_mode={smode!r}; valid={SYSTEM_MODES}")
    if smode == "minimal":
        return _minimal_system_prompt(reward_form, guidance)
    if guidance in _GUIDANCE_LEGACY_OBJ:
        return (_SYSTEM_PROMPT if smode == "full"
                else _drop_small_moves(_SYSTEM_PROMPT))
    expr = _reward_expr(reward_form)
    if reward_form == "pq_only":
        # In runs that hide SongBench, do not mention a metric that is not there.
        # At the default (not hidden) the string is the legacy one, byte-identical.
        scope = ("A reward = z_PQ is computed from a single model (Audiobox "
                 "Production Quality)."
                 if no_sb_in_search(default=True) else
                 "A reward = z_PQ is computed from a single model (Audiobox "
                 "Production Quality). SongBench-Mixing is also reported to you "
                 "but has ZERO weight in the reward; do not trade PQ away for it.")
    elif reward_form == "sb_only":
        scope = ("A reward = z_SB is computed from a single model "
                 "(SongBench-Mixing). Audiobox Production Quality is also "
                 "reported to you but has ZERO weight in the reward.")
    else:
        scope = (f"A {expr} is computed by two independent models (Audiobox "
                 "Production Quality and SongBench-Mixing).")
    base = ("You are a professional mixing engineer optimizing a multitrack "
            "music mix. At each step you propose EXACTLY ONE mixing action by "
            f"calling ONE tool. {scope} Your action is kept only if the reward "
            "improves (eval-gated greedy). Make small, musically-motivated "
            "moves. Use the per-stem roles and loudness to decide what each "
            "stem needs. Call exactly one tool; do not output prose-only "
            "answers.")
    return base if smode == "full" else _drop_small_moves(base)


# ---------------------------------------------------------------------------
# Whether to show the total budget (n_steps) in the prompt (C-2: recovering the prefix
# property)
# ---------------------------------------------------------------------------
# Greedy search is inherently **prefix-exact**: the state at move k equals "the result of
# running with budget k". So one run of 5000 moves yields the results for
# k=28/100/500/1000/5000 all at zero extra cost. However, the first line of the prompt is
# "Step {k}/{K}", which **reveals the total budget K to the LLM**, so the LLM's behavior
# depends on K and the prefix property breaks (there can be a behavioral difference such
# as hurrying at K=28 and taking it slow at K=5000).
#
# The default is True as before (= "Step k/K"). **False is recommended** (= "Step k").
# False changes the LLM's output, so it is chosen explicitly with a flag rather than by
# switching the default. Use False in runs that newly measure a budget curve, and
# distinguish runs from each other by the prompt_show_budget recorded in the run JSON.
PROMPT_SHOW_BUDGET_ENV = "AUTOMIX_PROMPT_SHOW_BUDGET"


def prompt_show_budget_default() -> bool:
    """Read the ``AUTOMIX_PROMPT_SHOW_BUDGET`` environment variable. Unset means True."""
    v = os.environ.get(PROMPT_SHOW_BUDGET_ENV)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Option 4: stating the output format explicitly + an action catalogue (presenting the
# existence of the 9 kinds and the argument ranges)
# ---------------------------------------------------------------------------
# The default is False = **the system prompt does not change by a single character**
# (bit-identical to existing runs).
#
# Motivation (based on measurements; see also "what this does not do" below):
#   * tool-call parsing succeeded 500/500, finish_reason was tool_calls in every case,
#     and tokens_out was 32-45 (3% of max_tokens=1536). **Neither format failures nor
#     truncation are happening.** So the aim of this block is not "repairing the format".
#   * The actual deficiency is in the coverage of the choices. Over 500 moves the tools
#     that appeared were only 2 kinds, apply_static_eq 498 / apply_static_gain 2, with
#     the remaining 7 kinds appearing zero times. The arguments had degenerated too:
#     gain_db ∈ {-1.0, -2.0} (never a boost) and q ≡ 2.0. The tool spec's JSON schema
#     does contain minimum/maximum, but there is no description beyond the name (title),
#     and neither "all 9 kinds are equally selectable at every move" nor "the whole
#     value range is legal" was stated in prose.
#
# The content of the block is **generated mechanically from the schema**:
#   (a) output format: state explicitly "call exactly one tool / emit no prose".
#   (b) action catalogue: enumerate all 9 kinds **in the same order as the tool specs**.
#       Reordering would amount to manipulating position bias, so the order of
#       _allowed_tool_specs() (= apply_static_eq first) is used as is.
#   (c) the **legal range** of each argument, written straight from the pydantic
#       schema's min/max. Optional arguments are shown in [], and defaults are written
#       straight from the schema's default.
#
# **What this does not do (avoiding answer leakage)**:
#   * It never writes which action is more likely to raise the reward. The 9 kinds are
#     listed on completely equal footing, with no ranking and no recommendation.
#   * It gives no concrete "good values" as examples. Values are angle-bracket
#     placeholders only (e.g. ``gain_db=<-12..12>``), never a specific number.
#     Writing concrete few-shot values would signal "that value is good", and since
#     parsing already succeeds 100% of the time, the gain on the format side is zero.
#   * It does not touch the Guidance block (the reverb/saturation/deesser statements).
#     That is the subject of a separate isolation, and moving it together here would
#     confound them.
PROMPT_FEWSHOT_ENV = "AUTOMIX_PROMPT_FEWSHOT"


def prompt_fewshot_default() -> bool:
    """Read the ``AUTOMIX_PROMPT_FEWSHOT`` environment variable. Unset means False (legacy behavior)."""
    v = os.environ.get(PROMPT_FEWSHOT_ENV)
    if v is None:
        return False
    return v.strip().lower() not in ("0", "false", "no", "off")


def _fmt_num(x: float) -> str:
    """Write a schema boundary value compactly for the prompt (1.0 -> 1, 0.5 -> 0.5)."""
    f = float(x)
    return str(int(f)) if f == int(f) else f"{f:g}"


def _range_placeholder(prop: Dict[str, Any]) -> str:
    """Build a ``<lo..hi>`` placeholder from one property of the JSON schema.

    No concrete values are written at all (so as not to teach which values are good).
    The bounds are copied straight from the pydantic schema, so they can never disagree
    with the validator.
    """
    lo = prop.get("minimum", prop.get("exclusiveMinimum"))
    hi = prop.get("maximum", prop.get("exclusiveMaximum"))
    if lo is None and hi is None:
        return "<string>" if prop.get("type") == "string" else "<number>"
    lo_s = _fmt_num(lo) if lo is not None else "-inf"
    hi_s = _fmt_num(hi) if hi is not None else "+inf"
    return f"<{lo_s}..{hi_s}>"


def _action_catalog_block(tool_specs: Optional[List[Dict[str, Any]]] = None) -> str:
    """Generate the catalogue of the 9 actions mechanically from the tool specs.

    The order of ``_allowed_tool_specs()`` (= TOOL_CATALOG's declaration order) is used
    as is. Reordering would amount to manipulating position bias, so it is not done.
    """
    specs = _allowed_tool_specs() if tool_specs is None else tool_specs
    lines: List[str] = []
    for s in specs:
        schema = s["input_schema"]
        props: Dict[str, Any] = schema.get("properties", {})
        req = set(schema.get("required", []))
        parts: List[str] = []
        for k, v in props.items():
            if k == "track":
                item = "track=<stem>"
            else:
                item = f"{k}={_range_placeholder(v)}"
                if "default" in v:
                    item += f" default={_fmt_num(v['default'])}"
            parts.append(item if k in req else f"[{item}]")
        lines.append(f"  {s['name']:<24s} " + ", ".join(parts))
    return "\n".join(lines)


_FEWSHOT_HEADER = """\


OUTPUT FORMAT. Reply with exactly ONE tool call and no prose. Do not emit two tool \
calls, and do not answer in text instead of calling a tool.

ACTION CATALOGUE. All NINE tools below are available to you at EVERY step, on equal \
footing. This list states what is LEGAL, not what is effective: it does not rank the \
tools and it does not recommend any of them. Angle brackets give the full legal range \
of each argument -- every value inside the range is accepted, including positive \
gain_db (a boost) as well as negative gain_db (a cut), and the whole span of freq and \
q. They are placeholders, not suggested values. Arguments in [square brackets] are \
optional and fall back to the stated default; <stem> must be replaced by one of the \
stem names listed in the design message.
"""


def fewshot_system_suffix() -> str:
    """The "format + catalogue" block appended to the system prompt."""
    return _FEWSHOT_HEADER + "\n" + _action_catalog_block() + "\n"


# ---------------------------------------------------------------------------
# Family 1: how the history is presented (how accepted / rejected are shown)
# ---------------------------------------------------------------------------
# The default is None = **the prompt is byte-for-byte identical to before**.
#
# Motivation (findings from eyeballing the actual prompts of a stalled run):
#   * The accepted history is the raw time series ``accepted[-8:]``. In a stalled run all
#     8 entries were apply_static_eq, and moreover they formed an **arithmetic
#     progression** stepping the bass 500→600→700→800→900 Hz. Placing "1000" as the next
#     token is the most natural continuation, i.e. the prompt itself teaches the EQ
#     repetition in-context.
#   * The rejected history is the raw time series ``rejected[-10:]``. After stalling, the
#     same line is repeated 10 times, so a single kind occupies all 10 slots and the clue
#     of "what else has been tried" disappears. It says "avoid repeats" and yet carries
#     zero information.
#
# What changes here is **only the way things are presented**; nothing is said about which
# actions are effective. The only material used is "the agent's own action history" and
# the specification that "the pipeline is deterministic", so this does not amount to
# passing along the per-action rankings the random arm obtained over 220k moves
# (= answer leakage).
#
# Keys of the dict:
#   accepted_mode: 'recent' (default, legacy)  emit the most recent accepted_n entries as
#                             a raw time series.
#                  'diverse'  keep at most accepted_per_kind entries per tool, newest
#                             first, then restore chronological order (this breaks the
#                             arithmetic progression).
#                  'by_kind'  aggregate per tool into "count / target stems / cumulative
#                             Δreward". Individual argument values (500,600,700…) are not
#                             shown.
#                  'none'     do not emit the accepted history.
#   accepted_n:    max entries emitted for 'recent'/'diverse' (default 8 = legacy).
#   accepted_per_kind: entries kept per tool for 'diverse' (default 1).
#   rejected_mode: None (default)  as before, decided by anti_repeat.rejected_distinct_n.
#                  'recent'    emit the most recent rejected_n entries as a raw time
#                              series (the legacy default).
#                  'distinct'  emit rejected_n distinct (action,args), folding duplicates
#                              into "(proposed Nx)".
#                  'none'      do not emit the rejected history.
#   rejected_n:    the count above (default 10 = legacy).
#   forbid_last_n: explicitly forbid the N most recent (distinct) rejected proposals with
#                  "do not propose these this time" (default 0 = do not emit). Not
#                  emitted when anti_repeat's ``forbidden`` is already present, since it
#                  would be redundant.
#
# **About ordering**: the 'by_kind' aggregation is listed **in order of first appearance**,
# not "in descending cumulative Δ". Sorting by cumulative Δ would put a ranking table of
# tools into the prompt, even though it is based on the agent's own measurements. A change
# to the presentation should not go that far, so the order is fixed to first appearance.
PROMPT_HISTORY_ENV = "AUTOMIX_PROMPT_HISTORY"
HISTORY_ACCEPTED_MODES = ("recent", "diverse", "by_kind", "none")
HISTORY_REJECTED_MODES = ("recent", "distinct", "none")


def prompt_history_default() -> Optional[Dict[str, Any]]:
    """Read the ``AUTOMIX_PROMPT_HISTORY`` environment variable (JSON). Unset means None."""
    raw = (os.environ.get(PROMPT_HISTORY_ENV) or "").strip()
    if not raw:
        return None
    return parse_prompt_history(raw)


def parse_prompt_history(raw: str) -> Optional[Dict[str, Any]]:
    """Validate the JSON string from the CLI / environment variable and turn it into a dict."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as ex:
        raise ValueError(f"{PROMPT_HISTORY_ENV} is not readable as JSON: {ex}") from ex
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError(f"{PROMPT_HISTORY_ENV} must be a JSON object: {d!r}")
    known = {"accepted_mode", "accepted_n", "accepted_per_kind",
             "rejected_mode", "rejected_n", "forbid_last_n",
             # --- added . At the default False the prompt is byte-identical ---
             # last_outcome: state the outcome of the immediately preceding move on one
             #   line at the head of the history.
             #   The accepted history shows the most recent 8 and the rejected history
             #   shows 10 distinct entries, so **"what happened to the move I just made"
             #   was not guaranteed to appear**. If acceptances continue the rejected
             #   side goes stale; if rejections continue the accepted side is not updated.
             # rejected_delta: also emit Δ (how far below the current best) for rejected
             #   moves. Previously rejections carried only the absolute reward, so a move
             #   that missed by 0.001 and one that fell 0.5 short looked the same.
             "last_outcome", "rejected_delta"}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"prompt_history: unknown keys {sorted(unknown)}; "
                         f"valid={sorted(known)}")
    am = d.get("accepted_mode")
    if am is not None and am not in HISTORY_ACCEPTED_MODES:
        raise ValueError(f"unknown accepted_mode={am!r}; "
                         f"valid={HISTORY_ACCEPTED_MODES}")
    rm = d.get("rejected_mode")
    if rm is not None and rm not in HISTORY_REJECTED_MODES:
        raise ValueError(f"unknown rejected_mode={rm!r}; "
                         f"valid={HISTORY_REJECTED_MODES}")
    return d


def _append_accepted_block(lines: List[str], accepted: List[Dict[str, Any]],
                           hcfg: Dict[str, Any], fa) -> None:
    """Append the accepted-history block to ``lines`` following ``hcfg['accepted_mode']``.

    This path is taken only when the ``history`` argument is given. The default
    (= ``history`` not specified) does not call this function, so bit-identity with
    existing runs is unaffected.
    For the meaning of the modes see the discussion of :data:`PROMPT_HISTORY_ENV`.
    """
    mode = str(hcfg.get("accepted_mode"))
    if mode not in HISTORY_ACCEPTED_MODES:
        raise ValueError(f"unknown accepted_mode={mode!r}; "
                         f"valid={HISTORY_ACCEPTED_MODES}")
    if mode == "none":
        return
    n = int(hcfg.get("accepted_n") or 8)
    if not accepted:
        lines.append("No action accepted yet.")
        return
    if mode == "recent":
        lines.append("Actions ACCEPTED so far (reward improved):")
        for r in accepted[-n:]:
            lines.append(f"  + {r['action']}({fa(r['args'])}) "
                         f"-> reward {r['reward']:+.4f} (Δ{r['delta']:+.4f})")
        return
    if mode == "by_kind":
        # Aggregate in order of first appearance. Sorting by descending cumulative Δ
        # would put a ranking table of tools into the prompt, so the order is fixed to
        # chronological (first appearance).
        agg: Dict[str, Dict[str, Any]] = {}
        for r in accepted:
            d = agg.setdefault(r["action"], {"n": 0, "gain": 0.0, "stems": []})
            d["n"] += 1
            d["gain"] += float(r.get("delta") or 0.0)
            trk = str(r["args"].get("track", ""))
            if trk and trk not in d["stems"]:
                d["stems"].append(trk)
        lines.append(f"Actions ACCEPTED so far, grouped by tool "
                     f"({len(accepted)} accepted in total; individual parameter "
                     "values and their order are not shown):")
        for a, d in agg.items():
            on = (" on " + ", ".join(d["stems"])) if d["stems"] else ""
            lines.append(f"  + {a} x{d['n']}{on} "
                         f"(cumulative Δreward {d['gain']:+.4f})")
        return
    # mode == "diverse": scan newest first and keep at most `per` entries per tool.
    # An arithmetic run like "bass at 500,600,700,800,900 Hz" collapses to one entry.
    per = max(1, int(hcfg.get("accepted_per_kind") or 1))
    cnt: Dict[str, int] = {}
    keep: List[int] = []
    for i in range(len(accepted) - 1, -1, -1):
        a = accepted[i]["action"]
        if cnt.get(a, 0) >= per:
            continue
        cnt[a] = cnt.get(a, 0) + 1
        keep.append(i)
        if len(keep) >= n:
            break
    keep.sort()
    hidden = len(accepted) - len(keep)
    tail = (f"; {hidden} further entries using the same tools are not shown"
            if hidden else "")
    lines.append(f"Actions ACCEPTED so far (reward improved; {len(accepted)} "
                 f"accepted in total, {len(keep)} shown, chosen so that each tool "
                 f"appears at most {per}x{tail}):")
    for i in keep:
        r = accepted[i]
        lines.append(f"  + {r['action']}({fa(r['args'])}) "
                     f"-> reward {r['reward']:+.4f} (Δ{r['delta']:+.4f})")


def _build_user_prompt(stem_lufs: Dict[str, Dict[str, float]],
                       cur_pq: float, cur_sb: float,
                       z_pq: float, z_sb: float, cur_reward: float,
                       accepted: List[Dict[str, Any]],
                       rejected: List[Dict[str, Any]],
                       step: int, n_steps: int,
                       show_budget: Optional[bool] = None,
                       anti_repeat: Optional[Dict[str, Any]] = None,
                       reward_form: Optional[str] = None,
                       guidance: Optional[str] = None,
                       excerpt_sec: Optional[float] = None,
                       numfmt: Optional[str] = None,
                       history: Optional[Dict[str, Any]] = None,
                       drop_guidance: Optional[bool] = None,
                       beat_times: Optional[List[float]] = None,
                       tempo_bpm: Optional[float] = None) -> str:
    """Assemble the context for each iter.

    ``beat_times`` / ``tempo_bpm`` (default None = do not emit the block; the output is
        byte-for-byte identical to before):
        The **beat times** [s] of the search domain (the 12-second excerpt). Pass them as
        times relative to the start of the excerpt (start = 0). They are needed to place
        automation breakpoints on beats.

        **This is not answer leakage**: beat positions are a physical property of the
        audio itself (the output of librosa.beat.beat_track), not a search result such as
        "how often each action gets accepted" obtained by running the random arm for
        220k moves. What is passed is only the fact that "this audio has this tempo and
        beats at these positions", something anyone can tell by listening, and nothing is
        said about which actions are effective.

    ``drop_guidance`` (default None = the ``AUTOMIX_PROMPT_DROP_GUIDANCE`` environment
        variable, False when unset = legacy behavior, byte-for-byte unchanged):
        Do not emit the prior-knowledge block at the end at all (family 4).
        With 'legacy'/'objfix'/'objfix2' the 4 hand-written action-ranking lines
        disappear; with 'task' the search-loop rules do. It is **orthogonal to the
        guidance mode** and does not touch how the reward formula is written (legacy's
        misstatement or objfix's fix), so it is not confounded with the isolation of the
        objective. objfix and task branch identically on the reward formula and differ
        only in this block, so with drop_guidance=True their user prompts are the same
        string.

    ``history`` (default None = legacy behavior; the output is byte-for-byte unchanged):
        Replace the **presentation** of the accepted / rejected histories (family 1).
        For the meaning of the keys see the discussion of :data:`PROMPT_HISTORY_ENV`.
        Nothing is said about which actions are effective, so this does not amount to
        handing over the random arm's statistics (answer leakage). All that is passed is
        "the agent's own action history" and the specification that "the pipeline is
        deterministic".

    ``guidance`` (default None = the environment variable; 'legacy' when that is unset
        too):
        With 'legacy' the output of this function is **byte-for-byte identical to
        before**. For the meaning of 'objfix' / 'objfix2' / 'task' see the discussion of
        :data:`GUIDANCE_MODES`. ``reward_form`` / ``excerpt_sec`` are never referenced
        under 'legacy'.

    ``numfmt`` (default None = the environment variable; 'legacy' when unset):
        The numeric notation of the arguments shown in the accepted/rejected histories.
        'legacy' is the legacy ``%.3g`` (exponent notation + 3 significant digits);
        'plain' does neither exponent notation nor digit loss.
        See the discussion of :data:`PROMPT_NUMFMT_ENV` for details.

    ``anti_repeat`` (default None = legacy behavior; the output is byte-for-byte
        unchanged):
        Extra context that structurally forbids repeated proposals. The dict contains

          ``rejected_distinct_n`` (int):
              Emit the REJECTED block as N "distinct (action,args)" entries.
              The legacy ``rejected[-10:]`` carries zero information once identical
              proposals line up. Duplicates are folded into a repetition count ``xN``.
              0/None keeps the legacy behavior.
          ``forbidden`` (List[str]):
              Signatures of proposals **already evaluated from the current best_state and
              rejected**. The pipeline is deterministic, so re-evaluating the same
              (action,args) from the same state always yields the same reward and is
              always rejected. Re-proposing therefore has zero information gain, and
              forbidding it costs the search nothing.
          ``tried_counts`` (Dict[str,int]):
              Per-tool counts of proposals made so far on this song (the agent's own
              action history).
          ``untried`` (List[str]):
              Names of tools not proposed even once yet.
          ``stall`` (int) / ``stall_window`` (int):
              The number of consecutive rejections since the last acceptance, and the
              threshold at which to warn. When ``stall >= stall_window``, append the
              stall-warning block.

        None of these blocks say "which action is good" (the Guidance stays unchanged).
        What is added is only the agent's own action history and a deterministic
        prohibition of duplicates.

    ``show_budget``:
        Whether the first line is ``Step {step}/{n_steps}`` (True) or ``Step {step}``
        (False). ``None`` follows :func:`prompt_show_budget_default`
        (= the environment variable, default True).

        **The default is True = the legacy behavior.** Setting it to False hides the
        total budget from the LLM and restores the prefix property of greedy search (the
        state at move k = the result of running with budget k), so a single run of 5000
        moves yields the results for k=28/100/500/1000/5000 at zero extra cost. But it is
        a change that **alters the LLM's output itself**, so it must not be mixed with
        existing runs. ``False`` is recommended for runs that re-measure a budget curve.
    """
    if show_budget is None:
        show_budget = prompt_show_budget_default()
    if guidance is None:
        guidance = guidance_mode_default()
    if numfmt is None:
        numfmt = prompt_numfmt_default()
    drop_g = (prompt_drop_guidance_default() if drop_guidance is None
              else bool(drop_guidance))

    def fa(a: Dict[str, Any]) -> str:
        return _fmt_args(a, numfmt)

    head = f"Step {step}/{n_steps}" if show_budget else f"Step {step}"
    lines: List[str] = []
    lines.append(f"{head}. Current mix metrics:")
    lines.append(f"  Audiobox-PQ = {cur_pq:.3f}  (z_PQ = {z_pq:+.3f})")
    # Emit the SongBench line **only when sb is finite**. In runs with
    # AUTOMIX_NO_SB_IN_SEARCH=1 the search loop returns sb=nan, so this line disappears.
    # With reward_form=pq_only, SongBench has weight 0, and showing its value only means
    # handing a 7B model "a number it must not look at" on every move
    # (user instruction : the premise is that the SongBench value is not
    # referenced).
    # The check is on nan, so existing runs (finite values) do not change by a single
    # character.
    if cur_sb == cur_sb:  # NaN is not equal to itself
        lines.append(f"  SongBench-Mixing = {cur_sb:.3f}  (z_SB = {z_sb:+.3f})")
    if guidance in _GUIDANCE_LEGACY_OBJ:
        lines.append(f"  reward = min(z_PQ, z_SB) = {cur_reward:+.4f}  "
                     f"(the bottleneck metric is {'PQ' if z_pq <= z_sb else 'SB'}).")
    elif reward_form == "pq_only":
        lines.append(f"  reward = z_PQ = {cur_reward:+.4f}   <-- THIS is what you "
                     "must raise. z_SB is shown for reference only and has ZERO "
                     "weight in the reward.")
    elif reward_form == "sb_only":
        lines.append(f"  reward = z_SB = {cur_reward:+.4f}   <-- THIS is what you "
                     "must raise. z_PQ is shown for reference only and has ZERO "
                     "weight in the reward.")
    elif reward_form == "min":
        lines.append(f"  reward = min(z_PQ, z_SB) = {cur_reward:+.4f}  "
                     f"(the bottleneck metric is {'PQ' if z_pq <= z_sb else 'SB'}).")
    else:
        lines.append(f"  {_reward_expr(reward_form)} = {cur_reward:+.4f}   "
                     "<-- THIS is what you must raise.")
    lines.append("")
    lines.append("Per-stem state (role and integrated loudness of the soloed stem):")
    for nm, info in stem_lufs.items():
        lines.append(f"  - {nm}: role={info['role']}, LUFS={info['lufs']:.2f}")
    lines.append("")
    # ------------------------------------------------------------------
    # Beat grid (added ; with beat_times=None not a single line is emitted =
    # identical to before)
    #
    # Information that lets automation breakpoints be placed at musical positions.
    # What is passed is the beat times librosa detected from the audio itself; no search
    # statistics are included.
    # Times are **seconds relative to the start of the excerpt** (the same coordinate
    # system passed to breakpoints).
    # ------------------------------------------------------------------
    if beat_times:
        bt = [float(t) for t in beat_times]
        tempo_txt = f", tempo ~{float(tempo_bpm):.1f} BPM" if tempo_bpm else ""
        span = (f" of this {excerpt_sec:.0f}-second excerpt"
                if excerpt_sec else "")
        lines.append(f"Beat grid{span} (detected from the audio; "
                     f"{len(bt)} beats{tempo_txt}). Times are seconds from the "
                     f"start of the audio you are mixing — the same coordinate "
                     f"system as automation breakpoints:")
        # Wrap at 8 beats per line (even 12 s x 150 BPM = 30 beats fits in 4 lines).
        for i in range(0, len(bt), 8):
            lines.append("  " + ", ".join(f"{t:.3f}" for t in bt[i:i + 8]))
        lines.append("")
    _hist_mark = len(lines)      # used to tell whether no history line was emitted at all
    # ------------------------------------------------------------------
    # Presentation of the accepted history (added 
    #
    # **This was the main cause of the stalling.** Reconstructing the actual prompts
    # showed that all 8 accepted entries were EQ, and moreover formed an arithmetic
    # progression stepping the bass 500→600→700→800→900 Hz. That is the same as teaching
    # the continuation "next is 1000 Hz" in-context, and even at temperature=1.0, 8/8
    # replies returned 1000 Hz (tools/replay_real_prompt.py --sweep).
    # Having 1000 Hz listed 10 times in the rejected history does not overturn it.
    #
    #   raw    : 8 entries in chronological order as before (default; bit-identical to
    #            existing runs)
    #   summary: aggregate by action kind and **do not show the chronological ordering**.
    #            Only "what was accepted and how many times" is passed. The arithmetic
    #            pattern disappears.
    #
    # The per-kind acceptance counts are the search's self-knowledge, not external
    # knowledge of "which action is good", so this does not amount to leaking the random
    # arm's results.
    # ------------------------------------------------------------------
    #
    # Addendum  (family 1): the env variable above is an earlier implementation
    # that switches only the "accepted history" between raw/summary. This function's
    # ``history`` argument generalizes it: **presentation of the rejected history, hiding
    # the histories, and explicitly forbidding the most recent proposals** can all be
    # specified in one dict, and it can be varied per call (an env variable applies to the
    # whole process, so candidates cannot be compared within one process). When
    # ``history`` is not passed, the env-variable behavior is used as is, **without
    # changing a single character**.
    hcfg = history or {}
    # --- Outcome of the immediately preceding move (added ; at the default
    # False not a single character changes) ---
    # The accepted history shows the most recent N and the rejected history N distinct
    # entries, so "what happened to the move I just made" is **not guaranteed by either
    # block**. If acceptances continue the rejected side goes stale; if rejections
    # continue the accepted side is not updated.
    # Not seeing this in sequential decision-making is a gap in the feedback.
    if hcfg.get("last_outcome"):
        la = accepted[-1] if accepted else None
        lr = rejected[-1] if rejected else None
        sa = int(la.get("step", -1)) if la else -1
        sr = int(lr.get("step", -1)) if lr else -1
        if la is not None and sa >= sr:
            lines.append(
                f"Your last proposal {la['action']}({fa(la['args'])}) was "
                f"ACCEPTED: reward {la['reward']:+.4f} "
                f"(improved by {float(la.get('delta') or 0.0):+.4f}).")
        elif lr is not None:
            dtxt = (f", {float(lr['delta']):+.4f} below the current best"
                    if lr.get("delta") is not None else "")
            # Write not just the fact of the rejection but also **what must not be done
            # next**. The pipeline is deterministic, so re-proposing the same
            # (action, args) against the same state always yields the same reward and is
            # always rejected. It merely throws a move away, with zero information gain.
            # anti_repeat's HARD CONSTRAINT block carries a prohibition to the same
            # effect, but **the immediately preceding move is named explicitly**
            # (user instruction, .
            lines.append(
                f"Your last proposal {lr['action']}({fa(lr['args'])}) was "
                f"REJECTED: reward {lr.get('reward', float('nan')):+.4f}{dtxt}. "
                "The mix is unchanged.")
            lines.append(
                f"Do NOT propose {lr['action']}({fa(lr['args'])}) again. "
                "Rendering and scoring are deterministic, so it would give the "
                "identical rejected result and waste this step. Change the tool, "
                "the target stem, or at least one argument value.")
        lines.append("")
    if hcfg.get("accepted_mode") is not None:
        _append_accepted_block(lines, accepted, hcfg, fa)
    elif accepted:
        hist_mode = (os.environ.get("AUTOMIX_ACCEPTED_HISTORY")
                     or "raw").strip().lower()
        if hist_mode == "summary":
            agg: Dict[str, Dict[str, float]] = {}
            for r in accepted:
                a = r["action"]
                d = agg.setdefault(a, {"n": 0.0, "gain": 0.0})
                d["n"] += 1
                d["gain"] += float(r.get("delta") or 0.0)
            lines.append(f"Accepted so far: {len(accepted)} actions "
                         f"(counts by type; order not shown):")
            for a in sorted(agg, key=lambda x: -agg[x]["n"]):
                d = agg[a]
                lines.append(f"  + {a}: {int(d['n'])}x "
                             f"(total reward gain {d['gain']:+.4f})")
        else:
            lines.append("Actions ACCEPTED so far (reward improved):")
            for r in accepted[-8:]:
                lines.append(f"  + {r['action']}({fa(r['args'])}) "
                             f"-> reward {r['reward']:+.4f} (Δ{r['delta']:+.4f})")
    else:
        lines.append("No action accepted yet.")
    ar = anti_repeat or {}
    # Presentation of the rejected history. If ``rejected_mode`` is unspecified this
    # depends on anti_repeat as before.
    rej_mode = hcfg.get("rejected_mode")
    rej_n = int(hcfg.get("rejected_n") or 10)
    if rej_mode == "distinct":
        n_distinct = rej_n
    elif rej_mode in ("recent", "none"):
        n_distinct = 0
    else:
        n_distinct = int(ar.get("rejected_distinct_n") or 0)
    rejected = [] if rej_mode == "none" else rejected
    if rejected and n_distinct > 0:
        # Pick N distinct (action,args) newest first, then restore chronological order
        # for output. Identical proposals are folded with an explicit repetition count
        # (the point is to make "you have proposed the same move N times" visible to the
        # LLM).
        seen: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        counts: Dict[str, int] = {}
        for r in rejected:
            key = f"{r['action']}({fa(r['args'])})"
            counts[key] = counts.get(key, 0) + 1
            if key not in seen:
                order.append(key)
            seen[key] = r
        keep = order[-n_distinct:]
        lines.append(f"Recent REJECTED actions ({len(rejected)} rejections so far, "
                     f"{len(order)} distinct; reward did NOT improve):")
        # rejected_delta: attach how far below the current best it fell (default False).
        # Previously only the absolute reward was shown, so **a move that missed by 0.001
        # and one that fell 0.5 short looked the same**. This removes the asymmetry where
        # accepted moves carry a Δ but rejected ones do not.
        want_d = bool(hcfg.get("rejected_delta"))
        for key in keep:
            r = seen[key]
            note = r.get("note", "")
            tail = f" [{note}]" if note else ""
            rep = f"  (proposed {counts[key]}x)" if counts[key] > 1 else ""
            dd = ""
            if want_d and r.get("delta") is not None:
                dd = f" (Δ{float(r['delta']):+.4f})"
            lines.append(f"  - {key} -> reward "
                         f"{r.get('reward', float('nan')):+.4f}{dd}{tail}{rep}")
    elif rejected:
        lines.append("Recent REJECTED actions (reward did NOT improve — avoid repeats):")
        for r in rejected[-rej_n:]:
            note = r.get("note", "")
            tail = f" [{note}]" if note else ""
            lines.append(f"  - {r['action']}({fa(r['args'])}) "
                         f"-> reward {r.get('reward', float('nan')):+.4f}{tail}")
    if ar.get("tried_counts") is not None:
        lines.append("")
        tc = ar.get("tried_counts") or {}
        tot = sum(tc.values())
        hist = ", ".join(f"{k} x{v}" for k, v in
                         sorted(tc.items(), key=lambda kv: -kv[1])) or "(none)"
        lines.append(f"Your own tool usage so far ({tot} proposals): {hist}")
        untried = ar.get("untried") or []
        if untried:
            lines.append("You have NOT yet tried: " + ", ".join(untried))
    forb = ar.get("forbidden") or []
    if forb:
        lines.append("")
        lines.append("HARD CONSTRAINT — the following were ALREADY evaluated from the "
                     "CURRENT mix state and rejected. The pipeline is deterministic, so "
                     "re-proposing any of them gives the exact same rejected result. "
                     "They are FORBIDDEN this step:")
        for key in forb:
            lines.append(f"  x {key}")
        lines.append("  Any new proposal must differ from every line above in its tool "
                     "name or in at least one argument value.")
    # Family 1: the minimal version of "the same action+args as the most recent proposal
    # is forbidden". When anti_repeat's ``forbidden`` (every entry already evaluated from
    # the current state) is present it is a superset, so this is not emitted (do not write
    # the same content twice).
    n_last = int(hcfg.get("forbid_last_n") or 0)
    if n_last > 0 and not forb and rejected:
        last: List[str] = []
        for r in reversed(rejected):
            key = f"{r['action']}({fa(r['args'])})"
            if key not in last:
                last.append(key)
            if len(last) >= n_last:
                break
        lines.append("")
        lines.append("HARD CONSTRAINT — your most recent rejected "
                     f"{'proposal is' if len(last) == 1 else 'proposals are'} listed "
                     "below. Do NOT propose "
                     f"{'it' if len(last) == 1 else 'any of them'} again this step; "
                     "your next proposal must differ in its tool name or in at least "
                     "one argument value:")
        for key in reversed(last):
            lines.append(f"  x {key}")
    stall = int(ar.get("stall") or 0)
    win = int(ar.get("stall_window") or 0)
    if win > 0 and stall >= win:
        lines.append("")
        lines.append(f"[STALLED] {stall} consecutive proposals have been rejected and "
                     "the mix state has not changed. Small variations of the same move "
                     "are not working. Try a DIFFERENT KIND of operation this step "
                     "(a different tool, or a different target stem), rather than "
                     "another variation of what you have been proposing.")
    if len(lines) == _hist_mark:
        # The 'none' variants of family 1: not a single history line was emitted. The
        # separating blank line would be left dangling as "a trace of something removed",
        # so drop that blank line too.
        lines.pop()
    _gf = load_guidance_file()
    if _gf.get("guidance"):
        # Replacement by an external file. It takes precedence over both drop_guidance
        # and the guidance mode. The surrounding blank lines follow the legacy block.
        lines.append("")
        lines.append(_gf["guidance"])
    elif drop_g:
        # Family 4: do not emit the prior-knowledge block at all. With
        # legacy/objfix/objfix2 the 4 hand-written action-ranking lines disappear; with
        # task the search-loop rules do. The surrounding blank lines are not emitted
        # either, so no trace of the removed block is left.
        pass
    elif guidance in _GUIDANCE_WITH_PRIORS:
        lines.append("")
        # The legacy Guidance block (action rankings derived from observations in the
        # full-song era).
        lines.append("Guidance (observed tendencies on this reward):")
        lines.append("  * apply_reverb and apply_saturation tend to LOWER Audiobox-PQ "
                     "(wet/harmonic content hurts the 'clean' axis); propose them only "
                     "with a clear reason.")
        lines.append("  * apply_deesser is expensive and only useful when a vocal stem "
                     "actually has harsh sibilance (5-9 kHz); do not propose it blindly.")
        lines.append("  * Prefer level/pan/width/EQ moves and light compression first.")
        # legacy / objfix write "bottleneck metric" on this line too. But when
        # reward_form is pq_only / sb_only the reward is a single term, so no bottleneck
        # exists. **objfix forgot to fix this line**, so even after fixing the objective
        # on the metrics line, the Guidance side kept pointing at a zero-weight metric.
        # objfix2 fixes it here as well.
        if guidance == "objfix2" and (reward_form in _SINGLE_TERM_FORMS):
            lines.append("  * To raise the reward, target the stem most likely "
                         "responsible (e.g. EQ a masking stem, rebalance loudness).")
        else:
            lines.append("  * To raise the bottleneck metric, target the stem most likely "
                         "responsible (e.g. EQ a masking stem, rebalance loudness).")
    else:
        # 'task': say nothing at all about the ranking of actions; state only the rules
        # of the search loop. Every fact written here is a specification derivable from
        # the code and contains none of the random arm's search results (how much each
        # action earns).
        exc = (f"a {excerpt_sec:.0f}-second excerpt of the song"
               if excerpt_sec else "a short excerpt of the song")
        lines.append("")
        lines.append("Search rules (how your proposal will be used):")
        lines.append("  * Your action is applied to the mix state shown above, the "
                     "result is rendered and scored, and it is KEPT only if the "
                     "reward strictly improves. Otherwise it is discarded and the "
                     "mix returns to exactly the state shown above.")
        lines.append("  * Rendering and scoring are deterministic. Re-proposing a "
                     "tool+arguments combination that is already listed as REJECTED "
                     "above will reproduce the identical reward and be rejected "
                     "again, spending a step and changing nothing.")
        lines.append(f"  * All metrics above are measured on {exc}, not on the "
                     "whole song.")
        lines.append("  * Every tool in the tool list is available at every step; "
                     "there is no ordering or preference among them beyond what you "
                     "judge musically. One exception: an apply_deesser proposal is "
                     "screened before rendering and is returned to you unused if the "
                     "target vocal stem has no measurable sibilance.")
    lines.append("")
    lines.append("Propose the single best next action now by calling one tool.")
    _txt = "\n".join(lines)
    if int(step) in _DUMP_STEPS:
        _dump_prompt("user", _txt, int(step))
    return _txt


# ---------------------------------------------------------------------------
# Numeric notation of arguments inside the prompt (family 2: part of the bug fixes
# around the objective)
# ---------------------------------------------------------------------------
# The default is "legacy" = the legacy ``f"{v:.3g}"``. **Not a single character changes.**
#
# legacy ('%.3g') does two kinds of real damage (measured :
#   1. **Exponent notation.** freq=1000.0 -> "1e+03", 3000.0 -> "3e+03",
#      12500.0 -> "1.25e+04". The tool schema requires plain numbers, so the history
#      block alone is written differently from the LLM's own output format.
#      Worse, when the accepted history lines up arithmetically as 500/600/700/800/900,
#      only 1000 becomes "1e+03" and the visible progression breaks.
#   2. **Digit loss.** With 3 significant digits, freq=1230.0 and 1234.0 both collapse to
#      "1.23e+03". size_sweep_run._sig uses _fmt_args directly as the signature of a
#      proposal, so **distinct proposals collide on the same signature**.
#      In anti-repeat's forbidden set this becomes the error of "forbidding a move that
#      was never evaluated as if it had been rejected" (cutting the search space for no
#      reason).
#
# "plain" drops the exponent notation and removes the digit loss. **It says nothing at
# all about which action is good, so it is not answer leakage** (a formatting-only fix).
PROMPT_NUMFMT_ENV = "AUTOMIX_PROMPT_NUMFMT"
PROMPT_NUMFMT_MODES = ("legacy", "plain")


def prompt_numfmt_default() -> str:
    """Read the ``AUTOMIX_PROMPT_NUMFMT`` environment variable. Unset means 'legacy' (legacy behavior)."""
    v = (os.environ.get(PROMPT_NUMFMT_ENV) or "legacy").strip().lower()
    if v not in PROMPT_NUMFMT_MODES:
        raise ValueError(f"unknown {PROMPT_NUMFMT_ENV}={v!r}; "
                         f"valid={PROMPT_NUMFMT_MODES}")
    return v


def _fmt_val_plain(v: float) -> str:
    """Write a float with neither exponent notation nor digit loss (1000.0 -> '1000', 0.7071 -> '0.7071')."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def _fmt_args(args: Dict[str, Any], numfmt: Optional[str] = None) -> str:
    """Write a proposal's arguments on one line for the prompt.

    ``numfmt`` (default None = the environment variable; 'legacy' when unset):
        With 'legacy' the output is **byte-for-byte identical to before** (``%.3g``).
        'plain' drops both the exponent notation and the digit loss. See the discussion
        of :data:`PROMPT_NUMFMT_ENV` for details.

    **The output of this function is also used as the signature of a proposal
    (size_sweep_run._sig)**, so pass the same ``numfmt`` for the prompt and the signature
    (a mismatch makes the prompt's REJECTED block and HARD CONSTRAINT block fail to match
    as strings).
    """
    if numfmt is None:
        numfmt = prompt_numfmt_default()
    if numfmt == "legacy":
        return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in args.items())
    return ", ".join(f"{k}={_fmt_val_plain(v)}" if isinstance(v, float)
                     else f"{k}={v}"
                     for k, v in args.items())


async def _qwen_propose(llm, user_prompt: str, tool_specs: List[Dict[str, Any]],
                        system: Optional[str] = None
                        ) -> Tuple[Optional[str], Dict[str, Any], str, Dict[str, Any]]:
    """Make Qwen emit one tool-call and return (name, args, raw_text, meta).

    Returns name=None when tool_calls is empty (prose only / parse failure).
    If multiple tool-calls arrive, only the first is taken (the 1 step = 1 action rule).

    If ``system`` is None, use :data:`_SYSTEM_PROMPT` as before
    (= bit-identical to guidance 'legacy').
    """
    resp = await llm.complete(system=system or _SYSTEM_PROMPT, user=user_prompt,
                              tool_specs=tool_specs)
    meta = {"tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out,
            "finish_reason": resp.meta.get("finish_reason"),
            # The model name the API actually served (when the backend returns meta.model).
            # Used to detect a recurrence of the 2026-06 incident where "a different model
            # was served for a haiku request".
            "model": resp.meta.get("model")}
    if not resp.tool_calls:
        return None, {}, resp.raw_text, meta
    first = resp.tool_calls[0]
    return first.get("name"), dict(first.get("arguments") or {}), resp.raw_text, meta


# ============================================================================
# helpers shared with v2 (per-stem LUFS / scoring / render / stat / save)
# ============================================================================
# The per-stem LUFS for the LLM proposer's context is memoized in a process-shared cache.
# best_state changes only on acceptance, so the recomputation on non-accepting steps
# (acceptance rate 9-20%) can be dropped entirely. The cache identifies the song by the
# identity of state.stems and resets automatically when the song changes (no false hits).
# See the docstring of src/mix_orchestrator/dsp/stem_lufs_cache.py for details and for the
# grounds of bit-identity.
#
# **Note**: this shared instance reads AUTOMIX_STEM_LUFS_CACHE at import time.
# Changing the environment variable after import has no effect (size_sweep_run recreates
# StemLufsCache() per song, so it reads at run time — the timing of the read differs by
# path). To disable it for an isolation experiment, set it **before starting the process**.
_STEM_LUFS_CACHE = StemLufsCache()


def _per_stem_lufs(state: MixState, sr: int,
                   cache: Optional[StemLufsCache] = None
                   ) -> Dict[str, Dict[str, float]]:
    """Solo-render each stem and measure its integrated LUFS (memoized).

    Passing ``cache`` uses that instance (size_sweep_run passes a dedicated instance per
    song). ``None`` uses the module-shared cache.
    ``AUTOMIX_STEM_LUFS_CACHE=0`` kills the memoization. The return value is
    bit-identical to the uncached implementation ``compute_per_stem_lufs``.
    """
    c = _STEM_LUFS_CACHE if cache is None else cache
    return c.per_stem_lufs(state, sr)


def _score_candidate(pq_scorer, sb_ear, audio: np.ndarray, sr: int
                     ) -> Tuple[float, float, float, float]:
    t0 = time.perf_counter()
    pq = float(pq_scorer.score(audio, sr))
    t1 = time.perf_counter()
    sb = float(sb_ear.score(audio, sr))
    t2 = time.perf_counter()
    return pq, sb, (t1 - t0), (t2 - t1)


def _render_and_norm(state: MixState, sr: int,
                     target_lufs: float) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    raw = render(state)
    audio = normalize_for_eval(raw, sr, target_lufs=target_lufs)
    return audio, (time.perf_counter() - t0)


def _stat(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"),
                "total": 0.0, "n": 0}
    a = np.asarray(xs, float)
    return {"mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3), "total": round(float(a.sum()), 3),
            "n": int(a.size)}


def _save_log(log: Dict[str, Any]) -> None:
    tmp = OUT_DIR / "agreement_loop_qwen_log.json.tmp"
    final = OUT_DIR / "agreement_loop_qwen_log.json"
    tmp.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(final)


# ============================================================================
# calibration pool (identical to v2: dry + P2 + KB(eq=False) + simple random candidates)
# ----------------------------------------------------------------------------
# Calibration only builds the z reference from random perturbations and does not use Qwen
# (it can even run before the server is up).
# It keeps a minimal inline version of v2's _propose_action and emits a diverse set of
# candidates including FX.
# ============================================================================
_CALIB_KINDS = ["gain", "pan", "eq", "comp", "reverb", "saturation"]


def _calib_random_action(rng: np.random.Generator, tracks: List[str]
                         ) -> Tuple[str, Dict[str, Any]]:
    kind = str(rng.choice(_CALIB_KINDS))
    t = str(rng.choice(tracks))
    if kind == "gain":
        return "apply_static_gain", {"track": t, "gain_db": float(rng.uniform(-4, 4))}
    if kind == "pan":
        return "apply_static_pan", {"track": t, "pan": float(rng.uniform(-0.6, 0.6))}
    if kind == "eq":
        return "apply_static_eq", {"track": t,
                                   "freq": float(rng.choice([120., 500., 3000., 8000.])),
                                   "gain_db": float(rng.uniform(-4, 4)),
                                   "q": float(rng.uniform(0.7, 1.8))}
    if kind == "comp":
        return "apply_static_compressor", {"track": t,
                                           "threshold_db": float(rng.uniform(-28, -12)),
                                           "ratio": float(rng.uniform(1.5, 3.5)),
                                           "attack_ms": 15.0, "release_ms": 150.0,
                                           "knee_db": 6.0}
    if kind == "reverb":
        return "apply_reverb", {"track": t, "room_size": 0.5, "damping": 0.5,
                                "wet_level": float(rng.uniform(0.1, 0.3)),
                                "dry_level": 0.85, "width": 0.9, "highpass_hz": 150.0}
    return "apply_saturation", {"track": t, "drive_db": float(rng.uniform(3, 8))}


# ============================================================================
# main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musdb18")
    ap.add_argument("--split", default="train")
    ap.add_argument("--song-index", type=int, default=0)
    ap.add_argument("--song", default=None)
    ap.add_argument("--duration-sec", type=float, default=0.0,
                    help="<=0 means the full song (default)")
    ap.add_argument("--n-random-calib", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=12,
                    help="number of Qwen proposal steps (10-15 recommended)")
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    ap.add_argument("--deesser-min-sibilance", type=float, default=0.04,
                    help="lower bound on the sibilance (5-9kHz) energy ratio for allowing apply_deesser")
    ap.add_argument("--max-invalid-retries", type=int, default=2,
                    help="how many times Qwen is asked to re-propose within one step on schema/gate failure")
    ap.add_argument("--prompt-show-budget", dest="prompt_show_budget",
                    action="store_true", default=prompt_show_budget_default(),
                    help="make the head of the prompt 'Step k/K' (default, legacy behavior).")
    ap.add_argument("--no-prompt-show-budget", dest="prompt_show_budget",
                    action="store_false",
                    help="make the head of the prompt 'Step k' and hide the total budget "
                         "(**recommended**). The prefix property of greedy is restored, so "
                         "the result for any budget k can be extracted from one long run "
                         "at zero extra cost. The LLM's output changes, so do not mix with "
                         "existing runs.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    wall_t0 = time.perf_counter()

    # ---- resolve the song id ----
    song_id = args.song
    if song_id is None and args.dataset != "synthetic":
        from _common import enumerate_songs
        songs = enumerate_songs(args.dataset, limit=args.song_index + 1,
                                split=args.split)
        songs = [s for s in songs if s is not None]
        if not songs:
            from mix_orchestrator.data.loaders import list_songs
            songs = list_songs(args.dataset, split=args.split)
        if args.song_index >= len(songs):
            raise IndexError(f"song-index {args.song_index} >= {len(songs)} songs")
        song_id = songs[args.song_index]
    print(f"[loop-qwen] dataset={args.dataset} split={args.split} song={song_id} "
          f"duration_sec={args.duration_sec}")

    # ---- load the stems (full song) ----
    t0 = time.perf_counter()
    stems, sr = load_one(args.dataset, song_id=song_id,
                         duration_sec=args.duration_sec, seed=args.seed,
                         split=args.split)
    load_sec = time.perf_counter() - t0
    main_stems = {k: v for k, v in stems.items() if k != "mixture"}
    dur = max(v.shape[-1] for v in main_stems.values()) / sr
    tracks = list(main_stems.keys())
    print(f"[loop-qwen] loaded {len(main_stems)} stems sr={sr} "
          f"duration={dur:.1f}s in {load_sec:.1f}s: {tracks}")

    # ---- Qwen backend (LocalLLMBackend) + connectivity check ----
    llm = build_llm("local", tracks)
    base_url = getattr(llm, "base_url", os.environ.get("LOCAL_LLM_URL", "?"))
    print(f"[loop-qwen] LLM backend=local model={getattr(llm, 'model', '?')} "
          f"base_url={base_url}")
    _check_llm_alive(base_url)

    # ---- scorers (the real models; both loaded in one process) ----
    from mix_orchestrator.eval.audiobox_pq import AudioboxPQScorer
    from mix_orchestrator.ears.tier6_music_reward import SongBenchMixingEar
    from mix_orchestrator.eval.agreement_reward import (
        fit_calibration, compute_agreement, SongCalibration)
    pq_scorer = AudioboxPQScorer()       # use_proxy_if_missing=False (no proxies allowed)
    sb_ear = SongBenchMixingEar()
    print("[loop-qwen] scorers constructed (models lazy-load on first score)")

    # ========================================================================
    # 1. Initial mix = KB mix eq=False (LUFS balance only)
    # ========================================================================
    t0 = time.perf_counter()
    init_raw, init_meta = build_knowledge_base_mix(main_stems, sr, eq=False)
    kb_build_sec = time.perf_counter() - t0
    init_state = MixState.initial_from_stems(main_stems, sr)
    for name, dec in init_meta["per_stem"].items():
        g = float(dec.get("gain_db", 0.0))
        if abs(g) > 1e-9:
            init_state = A.apply_static_gain(init_state, {"track": name, "gain_db": g})
    init_audio, _ = _render_and_norm(init_state, sr, args.target_lufs)
    init_lufs_per_stem = _per_stem_lufs(init_state, sr)
    print(f"[loop-qwen] initial KB mix (eq=False) built in {kb_build_sec:.1f}s; "
          f"per-stem LUFS:")
    for nm, info in init_lufs_per_stem.items():
        print(f"        {nm:12s} role={info['role']:7s} LUFS={info['lufs']:.2f}")

    # ========================================================================
    # 2. per-song calibration pool (the same composition as v2)
    # ========================================================================
    calib_audios: List[np.ndarray] = []
    calib_labels: List[str] = []

    dry_state = MixState.initial_from_stems(main_stems, sr)
    dry_audio, _ = _render_and_norm(dry_state, sr, args.target_lufs)
    calib_audios.append(dry_audio); calib_labels.append("dry_sum")

    try:
        from mix_orchestrator.strategies.analysis_driven import build_analysis_driven_mix
        p2_raw, _ = asyncio.run(build_analysis_driven_mix(main_stems, sr))
        p2_audio = normalize_for_eval(p2_raw, sr, target_lufs=args.target_lufs)
        calib_audios.append(p2_audio); calib_labels.append("p2_analysis_driven")
    except Exception as ex:                                       # noqa: BLE001
        print(f"[loop-qwen] WARN: P2 baseline skipped: {ex!r}")

    calib_audios.append(init_audio); calib_labels.append("kb_initial_eq_false")

    for i in range(args.n_random_calib):
        nm, aargs = _calib_random_action(rng, tracks)
        try:
            st = A.ACTION_HANDLERS[nm](init_state, aargs)
            au, _ = _render_and_norm(st, sr, args.target_lufs)
            calib_audios.append(au); calib_labels.append(f"rand_calib_{i}_{nm}")
        except Exception as ex:                                   # noqa: BLE001
            print(f"[loop-qwen] WARN: calib rand {i} ({nm}) skipped: {ex!r}")

    print(f"[loop-qwen] calibrating on {len(calib_audios)} pool members ...")
    calib_pq: List[float] = []
    calib_sb: List[float] = []
    calib_rows: List[Dict[str, Any]] = []
    calib_t0 = time.perf_counter()
    for label, au in zip(calib_labels, calib_audios):
        pq, sb, pq_s, sb_s = _score_candidate(pq_scorer, sb_ear, au, sr)
        calib_pq.append(pq); calib_sb.append(sb)
        calib_rows.append({"label": label, "pq": round(pq, 4), "sb": round(sb, 4),
                           "pq_sec": round(pq_s, 3), "sb_sec": round(sb_s, 3)})
        print(f"        {label:30s} PQ={pq:.4f} SB={sb:.4f} "
              f"(pq={pq_s:.2f}s sb={sb_s:.2f}s)")
    calib_total_sec = time.perf_counter() - calib_t0
    calib: SongCalibration = fit_calibration(calib_pq, calib_sb)
    print(f"[loop-qwen] calibration: PQ mean={calib.pq_mean:.4f} std={calib.pq_std:.4f} | "
          f"SB mean={calib.sb_mean:.4f} std={calib.sb_std:.4f} | "
          f"n={calib.n_samples} ({calib_total_sec:.1f}s)")
    # Observe std_floor firing (finding 4 in the note: sb_std tends to get clamped to
    # floor=0.05)
    sb_raw_std = float(np.std(np.asarray(calib_sb), ddof=0))
    pq_raw_std = float(np.std(np.asarray(calib_pq), ddof=0))
    floor_hit = {"pq_raw_std": round(pq_raw_std, 5), "pq_eff_std": round(calib.pq_std, 5),
                 "sb_raw_std": round(sb_raw_std, 5), "sb_eff_std": round(calib.sb_std, 5),
                 "pq_floor_clamped": bool(pq_raw_std < calib.pq_std - 1e-9),
                 "sb_floor_clamped": bool(sb_raw_std < calib.sb_std - 1e-9)}
    print(f"[loop-qwen] std_floor: pq_raw={pq_raw_std:.4f}->{calib.pq_std:.4f}"
          f"{' (CLAMPED)' if floor_hit['pq_floor_clamped'] else ''}  "
          f"sb_raw={sb_raw_std:.4f}->{calib.sb_std:.4f}"
          f"{' (CLAMPED)' if floor_hit['sb_floor_clamped'] else ''}")

    init_pq = calib_pq[calib_labels.index("kb_initial_eq_false")]
    init_sb = calib_sb[calib_labels.index("kb_initial_eq_false")]
    init_reward = compute_agreement(init_pq, init_sb, calib)
    print(f"[loop-qwen] INITIAL reward = min(z_PQ={calib.z_pq(init_pq):.3f}, "
          f"z_SB={calib.z_sb(init_sb):.3f}) = {init_reward:.4f}")

    # ========================================================================
    # 3-4. Qwen-driven eval-gated greedy loop
    # ========================================================================
    best_state = init_state
    best_reward = init_reward
    best_pq, best_sb = init_pq, init_sb
    accepted: List[Dict[str, Any]] = []
    rejected_ctx: List[Dict[str, Any]] = []     # rejection history for the Qwen context
    step_rows: List[Dict[str, Any]] = []
    render_secs: List[float] = []
    pq_secs: List[float] = []
    sb_secs: List[float] = []
    qwen_secs: List[float] = []
    qwen_tokens_in: List[float] = []
    qwen_tokens_out: List[float] = []
    by_kind: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"proposed": 0, "accepted": 0, "rejected": 0,
                 "invalid": 0, "gated": 0})
    render_by_kind: Dict[str, List[float]] = defaultdict(list)

    tool_specs = _allowed_tool_specs()

    log: Dict[str, Any] = {
        "schema": "agreement_loop_qwen", "status": "running",
        "dataset": args.dataset, "split": args.split, "song_id": song_id,
        "duration_sec_arg": args.duration_sec, "actual_duration_sec": round(dur, 2),
        "sr": sr, "target_lufs": args.target_lufs, "seed": args.seed,
        "n_steps": args.n_steps, "n_random_calib": args.n_random_calib,
        "llm": {"backend": "local", "model": getattr(llm, "model", "?"),
                "base_url": base_url},
        "allowed_actions": allowed_actions(),
        "action_set": action_set_default(),
        # whether the total budget was shown in the prompt (presence of the prefix
        # property = essential for telling whether runs are compatible)
        "prompt_show_budget": bool(args.prompt_show_budget),
        "deesser_min_sibilance": args.deesser_min_sibilance,
        "initial_mix": {
            "method": init_meta["method"], "eq": init_meta["eq"],
            "per_stem_decision": init_meta["per_stem"],
            "per_stem_lufs": init_lufs_per_stem,
            "pq": round(init_pq, 4), "sb": round(init_sb, 4),
            "reward": round(init_reward, 4),
        },
        "calibration": {
            "pq_mean": round(calib.pq_mean, 4), "pq_std": round(calib.pq_std, 4),
            "sb_mean": round(calib.sb_mean, 4), "sb_std": round(calib.sb_std, 4),
            "n_samples": calib.n_samples, "std_floor": floor_hit, "pool": calib_rows,
        },
        "steps": step_rows,
    }
    _save_log(log)

    search_t0 = time.perf_counter()
    for step in range(1, args.n_steps + 1):
        cur_lufs = _per_stem_lufs(best_state, sr)
        z_pq, z_sb = calib.z_pq(best_pq), calib.z_sb(best_sb)
        user_prompt = _build_user_prompt(
            cur_lufs, best_pq, best_sb, z_pq, z_sb, best_reward,
            accepted, rejected_ctx, step, args.n_steps,
            show_budget=args.prompt_show_budget)

        # --- Qwen proposal (on schema/gate failure, re-propose up to max_invalid_retries times) ---
        name: Optional[str] = None
        aargs: Dict[str, Any] = {}
        raw_text = ""
        qmeta: Dict[str, Any] = {}
        gate_reason = ""
        qwen_step_sec = 0.0
        attempts: List[Dict[str, Any]] = []
        validated_args: Optional[Dict[str, Any]] = None
        sib_ratio: Optional[float] = None
        for attempt in range(args.max_invalid_retries + 1):
            qt0 = time.perf_counter()
            name, aargs, raw_text, qmeta = asyncio.run(
                _qwen_propose(llm, user_prompt, tool_specs))
            qwen_step_sec += time.perf_counter() - qt0
            if name is None:
                attempts.append({"attempt": attempt, "error": "no tool_call",
                                 "raw_text": raw_text[:200]})
                # retry, prompting the context to "call a tool"
                user_prompt += ("\n\n[system] Your previous reply contained no tool "
                                "call. You MUST call exactly one tool now.")
                continue
            ok, validated_args, reason = _validate_action(name, aargs, tracks)
            if not ok:
                by_kind[name if name in allowed_actions() else "unknown"]["invalid"] += 1
                attempts.append({"attempt": attempt, "action": name, "args": aargs,
                                 "error": reason})
                user_prompt += (f"\n\n[system] Your proposal {name}({aargs}) was "
                                f"invalid: {reason}. Propose a valid action.")
                name = None
                continue
            # de-esser gate (cheap sibilance pre-check; avoids the ~11s render)
            if name == "apply_deesser":
                allow, sib_ratio, gate_reason = _deesser_gate(
                    best_state, validated_args, sr, args.deesser_min_sibilance)
                if not allow:
                    by_kind[name]["gated"] += 1
                    attempts.append({"attempt": attempt, "action": name,
                                     "args": validated_args, "gated": True,
                                     "sibilance_ratio": round(float(sib_ratio), 4),
                                     "error": gate_reason})
                    rejected_ctx.append({"action": name, "args": validated_args,
                                         "reward": best_reward,
                                         "note": "gated: no sibilance"})
                    user_prompt += (f"\n\n[system] {gate_reason}. Propose a "
                                    f"different action.")
                    name = None
                    continue
            break  # valid (and gate-passed) proposal

        qwen_secs.append(qwen_step_sec)
        qwen_tokens_in.append(float(qmeta.get("tokens_in", 0) or 0))
        qwen_tokens_out.append(float(qmeta.get("tokens_out", 0) or 0))

        if name is None or validated_args is None:
            print(f"[step {step:02d}] no valid Qwen proposal after "
                  f"{args.max_invalid_retries + 1} attempts; skipping")
            step_rows.append({"step": step, "skipped": True, "attempts": attempts,
                              "qwen_sec": round(qwen_step_sec, 3)})
            _save_log(log)
            continue

        by_kind[name]["proposed"] += 1
        # --- apply + render + score + eval-gate ---
        try:
            cand_state = A.ACTION_HANDLERS[name](best_state, validated_args)
        except Exception as ex:                                   # noqa: BLE001
            by_kind[name]["invalid"] += 1
            print(f"[step {step:02d}] apply {name} failed: {ex!r}")
            step_rows.append({"step": step, "action": name, "args": validated_args,
                              "invalid": True, "error": repr(ex),
                              "qwen_sec": round(qwen_step_sec, 3),
                              "attempts": attempts})
            _save_log(log)
            continue

        audio, render_sec = _render_and_norm(cand_state, sr, args.target_lufs)
        pq, sb, pq_s, sb_s = _score_candidate(pq_scorer, sb_ear, audio, sr)
        reward = compute_agreement(pq, sb, calib)
        render_secs.append(render_sec); pq_secs.append(pq_s); sb_secs.append(sb_s)
        render_by_kind[name].append(render_sec)
        improved = reward > best_reward
        delta = reward - best_reward

        row = {
            "step": step, "action": name, "args": validated_args,
            "raw_text": raw_text[:500],
            "pq": round(pq, 4), "sb": round(sb, 4),
            "z_pq": round(calib.z_pq(pq), 4), "z_sb": round(calib.z_sb(sb), 4),
            "reward": round(reward, 4), "delta": round(delta, 4),
            "improved": bool(improved),
            "render_sec": round(render_sec, 3),
            "pq_sec": round(pq_s, 3), "sb_sec": round(sb_s, 3),
            "qwen_sec": round(qwen_step_sec, 3),
            "qwen_tokens_in": qmeta.get("tokens_in", 0),
            "qwen_tokens_out": qmeta.get("tokens_out", 0),
            "attempts": attempts,
        }
        if sib_ratio is not None:
            row["sibilance_ratio"] = round(float(sib_ratio), 4)
        step_rows.append(row)
        tag = "ACCEPT" if improved else "reject"
        print(f"[step {step:02d}] {name:22s} {_fmt_args(validated_args)} -> "
              f"PQ={pq:.3f} SB={sb:.3f} R={reward:.4f} (best={best_reward:.4f}) "
              f"[{tag}] qwen={qwen_step_sec:.2f}s render={render_sec:.2f}s "
              f"pq={pq_s:.2f}s sb={sb_s:.2f}s")
        if raw_text.strip():
            print(f"          reason: {raw_text.strip()[:160]}")

        if improved:
            by_kind[name]["accepted"] += 1
            best_state = cand_state
            best_reward = reward
            best_pq, best_sb = pq, sb
            accepted.append({"action": name, "args": validated_args,
                             "reward": round(reward, 4), "delta": round(delta, 4)})
        else:
            by_kind[name]["rejected"] += 1
            rejected_ctx.append({"action": name, "args": validated_args,
                                 "reward": round(reward, 4)})
        _save_log(log)
    search_total_sec = time.perf_counter() - search_t0

    # ========================================================================
    # 5. Render and save the final mix
    # ========================================================================
    final_audio, _ = _render_and_norm(best_state, sr, args.target_lufs)
    final_lufs_per_stem = _per_stem_lufs(best_state, sr)

    import soundfile as sf
    sf.write(str(OUT_DIR / "initial_mix.wav"), init_audio.T, sr, subtype="FLOAT")
    sf.write(str(OUT_DIR / "final_mix.wav"), final_audio.T, sr, subtype="FLOAT")

    per_step_eval_sec = [r["render_sec"] + r["pq_sec"] + r["sb_sec"]
                         for r in step_rows if "render_sec" in r]
    qwen_total = float(np.sum(qwen_secs)) if qwen_secs else 0.0
    cost = {
        "render_sec": _stat(render_secs),
        "pq_sec": _stat(pq_secs),
        "sb_sec": _stat(sb_secs),
        "qwen_sec": _stat(qwen_secs),
        "qwen_tokens_in": _stat(qwen_tokens_in),
        "qwen_tokens_out": _stat(qwen_tokens_out),
        "per_step_eval_sec": _stat(per_step_eval_sec),
        "render_by_action_kind": {k: _stat(v) for k, v in render_by_kind.items()},
        "calibration_total_sec": round(calib_total_sec, 2),
        "search_total_sec": round(search_total_sec, 2),
        "qwen_total_sec": round(qwen_total, 2),
        "stems_load_sec": round(load_sec, 2),
        "wall_total_sec": round(time.perf_counter() - wall_t0, 2),
    }
    per_song_loop_sec = calib_total_sec + search_total_sec
    cost["per_song_loop_sec"] = round(per_song_loop_sec, 2)
    cost["est_150_songs_hours"] = round(per_song_loop_sec * 150 / 3600.0, 2)

    by_kind_out = {k: dict(v) for k, v in by_kind.items()}

    log.update({
        "status": "done",
        "final_mix": {
            "per_stem_lufs": final_lufs_per_stem,
            "pq": round(best_pq, 4), "sb": round(best_sb, 4),
            "reward": round(best_reward, 4),
            "n_accepted": len(accepted),
            "accepted_actions": accepted,
        },
        "reward_delta": round(best_reward - init_reward, 4),
        "by_action_kind": by_kind_out,
        "cost": cost,
    })
    _save_log(log)

    print("\n" + "=" * 72)
    print(f"[RESULT] initial reward = {init_reward:.4f} -> final reward = "
          f"{best_reward:.4f}  (Δ={best_reward - init_reward:+.4f})")
    print(f"[RESULT] initial PQ={init_pq:.3f} SB={init_sb:.3f} -> "
          f"final PQ={best_pq:.3f} SB={best_sb:.3f}")
    n_scored = len([r for r in step_rows if 'render_sec' in r])
    print(f"[RESULT] accepted {len(accepted)}/{n_scored} scored proposals "
          f"({args.n_steps} steps)")
    print("[RESULT] by action (proposed/accepted/rejected/invalid/gated):")
    for k in allowed_actions():
        v = by_kind_out.get(k, {"proposed": 0, "accepted": 0, "rejected": 0,
                                "invalid": 0, "gated": 0})
        print(f"        {k:24s} {v['proposed']}/{v['accepted']}/{v['rejected']}/"
              f"{v['invalid']}/{v['gated']}")
    print(f"[COST] Qwen per-step mean={cost['qwen_sec']['mean']:.2f}s "
          f"(in={cost['qwen_tokens_in']['mean']:.0f} tok, "
          f"out={cost['qwen_tokens_out']['mean']:.0f} tok); "
          f"render+PQ+SB per-step mean={cost['per_step_eval_sec']['mean']:.2f}s")
    print(f"[COST] 1-song loop (calib+search incl. Qwen) = {per_song_loop_sec:.1f}s "
          f"-> 150 songs ~= {cost['est_150_songs_hours']:.1f} h")
    print(f"[COST] wall total = {cost['wall_total_sec']:.1f}s")
    print(f"[loop-qwen] outputs -> {OUT_DIR}")
    print("=" * 72)
    return 0


def _check_llm_alive(base_url: str) -> None:
    """Hit the vLLM server's /v1/models to confirm it is up. Stop with a clear message on failure."""
    import urllib.request
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url + "/models"
    else:
        url = url + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace")
        print(f"[loop-qwen] LLM server alive: {url} -> {body[:200]}")
    except Exception as ex:                                       # noqa: BLE001
        raise RuntimeError(
            f"LLM server not reachable at {url}: {ex!r}. "
            f"Start the vLLM Qwen server first (the local job wrapper ... vllm "
            f"serve --port <P>) and export LOCAL_LLM_URL=http://127.0.0.1:<P>/v1."
        ) from ex


if __name__ == "__main__":
    raise SystemExit(main())
