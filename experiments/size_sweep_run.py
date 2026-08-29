"""Size-sweep runner for dBFS robust-mixing search driven by an LLM proposer.

Purpose (Qwen2.5 size sweep): integrate the **tool-call proposals of a local LLM
(Qwen2.5 behind a vLLM OpenAI-compatible server)**, validated in
agreement_loop_qwen.py, into the production pipeline agreement_loop_all.py and its
dBFS robust-mixing (random mode). Starting from randomized_stems, where each stem is
multiplied by gain_db ~ U(LOW,HIGH) so that the professional balance is destroyed, we
evaluate across model sizes whether the LLM proposer can recover a good mix through
eval-gated greedy search.

Design (follows the finalized design in the note):
  - The heavy machinery of agreement_loop_all.py (build_song_plan / filter_pending /
    per-song JSON / _run_one_song_random / _propose_pipeline / scorers) is **reused via
    import**. This file only holds the LLM proposer (make_llm_search_proposer) and the
    driver (main).
  - LLM tool-call proposals are injected through the ``search_proposer`` hook added to
    the search loop of _propose_pipeline (agreement_loop_all.py, backward compatible).
    The hook is injected by resetting ``args.search_proposer`` for each song.
  - LLM proposals reuse the validated helpers of agreement_loop_qwen.py as-is:
      _allowed_tool_specs / _validate_action / _build_user_prompt / _qwen_propose
      / _per_stem_lufs / _deesser_gate / _SYSTEM_PROMPT.
  - The backend is built by build_llm(args.backend). local=vLLM Qwen, anthropic=Claude,
    mock=scripted MockBackend (for CPU tests, not a proxy).

Hard policy (memory: feedback_no_proxy_no_experiment): no proxies. PQ/SB/LLM are the
real thing. The mock backend supplies scripted actions **for tests only** and is never
used in real runs.

GPU run (vLLM server co-location + size_sweep_run):
  # 1) LLM server (separate job; the model is what the size sweep varies)
  IMAGE_NAME=vllm/vllm-openai:latest NUM_GPUS=1 JOB_TIME=06:00:00 \
    /path/to/the cluster job wrapper python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-7B-Instruct-AWQ --host 0.0.0.0 --port 8000 \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --max-model-len 32768 --served-model-name qwen
  # 2) loop (same node, 127.0.0.1 reachable)
  LOCAL_LLM_URL=http://127.0.0.1:8000/v1 LOCAL_LLM_MODEL=qwen \
  IMAGE_NAME=songbench-image:latest JOB_TIME=06:00:00 \
    /path/to/the cluster job wrapper python experiments/size_sweep_run.py \
      --random-gain-db-range -43 -18 --normalize-corrupted-lufs -14 \
      --splits dev --limit 24 --n-search 12 --random-seeds 1 \
      --backend local --llm-model qwen --run-name sweep_qwen7b_v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# experiments/_common puts ROOT/src on sys.path and configures the HF cache.
from _common import ROOT, build_llm  # noqa: E402

# Reuse the heavy machinery of the production pipeline (the real thing, not a proxy).
import agreement_loop_all as ALL  # noqa: E402

# Reuse the validated LLM proposer helpers.
from agreement_loop_qwen import (  # noqa: E402
    _SYSTEM_PROMPT,           # noqa: F401  (for a future system-prompt override hook)
    GUIDANCE_MODES,
    _allowed_tool_specs,
    _build_user_prompt,
    _deesser_gate,
    _fmt_args,
    _per_stem_lufs,
    _qwen_propose,
    _validate_action,
    ACTION_SETS,
    action_set_default,
    action_set_is_time_absolute,
    allowed_actions,
    PROMPT_NUMFMT_MODES,
    build_system_prompt,
    fewshot_system_suffix,
    guidance_mode_default,
    parse_prompt_history,
    prompt_drop_guidance_default,
    prompt_fewshot_default,
    prompt_history_default,
    prompt_numfmt_default,
    prompt_show_budget_default,
    prompt_system_mode_default,
    SYSTEM_MODES,
)

from mix_orchestrator.dsp.stem_lufs_cache import StemLufsCache  # noqa: E402


# ============================================================================
# Arguments: extend agreement_loop_all.build_arg_parser for size_sweep
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Reuse the agreement_loop_all parser and add LLM-proposer arguments.

    The dBFS / song-window / search arguments (--random-gain-db-range /
    --normalize-corrupted-lufs / --random-seeds / --n-search / --splits / --offset /
    --limit / --run-name) are reused as-is. What is specific to size_sweep is
    backend / model / retry / proposer.
    """
    ap = ALL.build_arg_parser()
    ap.description = ("Size-sweep runner for dBFS robust-mixing search driven by an "
                      "LLM proposer")
    # backend choice (local=vLLM Qwen / anthropic=Claude / mock=scripted, for CPU tests)
    ap.add_argument("--backend", choices=["local", "anthropic", "gemini", "mock"],
                    default="local",
                    help="LLM backend. local=vLLM Qwen (the main path for the size "
                         "sweep), anthropic=Claude, gemini=Gemini API, mock=scripted "
                         "(for CPU tests).")
    ap.add_argument("--llm-model", default=None,
                    help="Served model name for the local backend (overrides env "
                         "LOCAL_LLM_MODEL). This is what the size sweep varies "
                         "(e.g. qwen). Falls back to env when unset.")
    ap.add_argument("--llm-max-invalid-retries", type=int, default=3,
                    help="How many times to re-ask the LLM within one step when the "
                         "proposal fails the schema/gate checks.")
    ap.add_argument("--deesser-min-sibilance", type=float, default=0.04,
                    help="Lower bound on the sibilance (5-9 kHz) energy ratio required "
                         "to allow apply_deesser. If the target vocal stem has low "
                         "sibilance, the proposal is rejected before rendering.")
    # proposer: llm (the main path) / random (apples-to-apples baseline over the same
    # pilot set and the same song_index; reuses the random path of agreement_loop_all
    # as-is).
    # Note: agreement_loop_all alone has no pilot selection, and its song_index is
    # offset-dependent (chunk-local), so a random baseline under conditions identical
    # to the sweep must be run through this runner (no offset = full plan indexing).
    ap.add_argument("--proposer", choices=["llm", "random", "hybrid",
                                           "informed_random"], default="llm",
                    help="Source of search proposals. llm=LLM tool-call proposal at "
                         "every step, random=random proposal at every step (for the "
                         "baseline arm), hybrid=LLM while step<switch and random once "
                         "step>=switch (combines the quality of the LLM's first moves "
                         "with the exploration diversity of continuing at random), "
                         "informed_random=static samples drawn from an empirical prior "
                         "(--proposer-stats) built from the accepted gains of measured "
                         "uniform-random runs (a zero-adaptivity control: the action "
                         "space, acceptance gate and seed machinery are identical to "
                         "random, only the sampling distribution differs).")
    ap.add_argument("--action-set", dest="action_set",
                    choices=list(ACTION_SETS), default=None,
                    help="Action space shown to the LLM (default None = env variable "
                         "AUTOMIX_ACTION_SET, and 'base' = the conventional 9 actions "
                         "when that is unset). 'automation' is the 10 actions obtained "
                         "by adding apply_gain_automation. The breakpoints of "
                         "automation are absolute times, so in excerpt mode "
                         "(--search-excerpt-sec>0) --excerpt-only is mandatory (the "
                         "chain cannot be transferred onto the full song). "
                         "**The random arm never proposes automation** "
                         "(agreement_loop_v2._propose_action still has 9 actions), so "
                         "comparing an 'automation' run directly against the random arm "
                         "makes the action spaces asymmetric. Only LLM runs with the "
                         "same action_set may be compared.")
    # Adaptive switching (added : "switch once the vein of prior knowledge is
    # mined out". If there is no acceptance for stall_n consecutive moves since the last
    # acceptance, that song **switches permanently** to random from then on. The switch
    # races the fixed step (--hybrid-switch-step); whichever fires first wins.
    # Rationale: measurements from the partial k10r100 log show that the only source of
    # argv1's advantage is a 1.3x larger gain per acceptance early on; after it stalls it
    # is consistently worse than random even on acceptance rate.
    ap.add_argument("--hybrid-stall-n", type=int, default=0,
                    help="Adaptive switching for hybrid: permanently switch to random "
                         "after N consecutive moves without acceptance (0=disabled). "
                         "Can be combined with --hybrid-switch-step (whichever fires "
                         "first wins)")
    ap.add_argument("--hybrid-switch-step", type=int, default=4,
                    help="Switch step for --proposer hybrid. step<switch uses LLM "
                         "proposals, step>=switch uses random proposals (continuing "
                         "from best_state). Default 4, taken from the LLM-advantage "
                         "band of the budget curve (k=1..6).")
    # Empirical prior for informed_random (added , in response to the external
    # audit). Default None: unless specified, not a single byte of the behavior of any
    # existing path changes.
    ap.add_argument("--proposer-stats", dest="proposer_stats", default=None,
                    help="Empirical prior JSON for --proposer informed_random "
                         "(e.g. experiments/informed_prior_stats.json). Required for "
                         "informed_random; specifying it with any other proposer is an "
                         "error (fail-fast). The file's sha256 is recorded in "
                         "proposer_provenance of the per-song JSON, and on resume it is "
                         "checked against the existing songs.")
    # pilot: a fixed set of N songs drawn from the 150-song universe with split
    # stratification (shared across all model sizes).
    # song_index (= corruption seed) still follows the full plan, so the
    # randomized_stems of the selected songs match agree_random_dbfs_v1 exactly (only
    # the proposer is swapped).
    ap.add_argument("--pilot-n", type=int, default=None,
                    help="Deterministically sample N songs from the full plan with "
                         "split stratification and restrict execution to them (a fixed "
                         "pilot set shared across all model sizes).")
    ap.add_argument("--pilot-seed", type=int, default=0,
                    help="Seed for the deterministic sampling of --pilot-n (must be "
                         "identical across all model sizes).")
    # Whether to show the total budget (n_search) in the prompt. Default True = legacy
    # behavior. Setting it to False restores the prefix property of greedy search, so a
    # single 5000-move run yields the results for k=28/100/500/1000/5000 all at once at
    # zero extra cost.
    # **This changes the LLM output**, so the default is left alone and it is only
    # toggled by an explicit flag.
    ap.add_argument("--prompt-show-budget", dest="prompt_show_budget",
                    action="store_true", default=prompt_show_budget_default(),
                    help="Start the prompt with 'Step k/K' (default, legacy behavior).")
    ap.add_argument("--no-prompt-show-budget", dest="prompt_show_budget",
                    action="store_false",
                    help="Start the prompt with 'Step k' and hide the total budget "
                         "(**recommended**). Greedy search becomes prefix-exact, so the "
                         "result for any budget k can be cut out of a single long run. "
                         "The LLM output changes, so do not mix with existing runs "
                         "(llm_stats.prompt_show_budget in the run JSON tells them "
                         "apart).")
    # Structural ban on repeated proposals (idea 3). Default OFF = legacy behavior (the
    # prompt is byte-for-byte unchanged).
    # The search algorithm ((1+1) greedy with a strict-improvement gate) is untouched;
    # this only adds context passed to the proposer plus "bounce back duplicate
    # proposals at the same state".
    ap.add_argument("--anti-repeat", dest="anti_repeat", action="store_true",
                    default=False,
                    help="Structurally forbid repeated proposals: show REJECTED as N "
                         "distinct entries, forbid in the prompt any (action,args) "
                         "already evaluated at the current state and bounce it back "
                         "before rendering, enumerate unused tools, and warn on stalls. "
                         "**The LLM output changes**, so do not mix with existing runs "
                         "(llm_stats.anti_repeat tells them apart).")
    ap.add_argument("--anti-repeat-stall-window", type=int, default=20,
                    help="Number of consecutive rejections after which --anti-repeat "
                         "emits a stall warning (default 20).")
    ap.add_argument("--anti-repeat-forbid-n", type=int, default=12,
                    help="Maximum number of forbidden proposals --anti-repeat "
                         "enumerates in the prompt.")
    ap.add_argument("--anti-repeat-rejected-n", type=int, default=10,
                    help="Number of distinct proposals shown in the REJECTED block of "
                         "--anti-repeat.")
    # Prior-knowledge block of the prompt (idea 2). Default legacy = legacy behavior (the
    # prompt is unchanged).
    ap.add_argument("--guidance", choices=list(GUIDANCE_MODES),
                    default=guidance_mode_default(),
                    help="Composition of the prior-knowledge block of the prompt. "
                         "legacy=as before (always writes the reward as "
                         "min(z_PQ,z_SB) and emits the 4-line Guidance that discourages "
                         "reverb/saturation/deesser and recommends EQ/width). "
                         "objfix=make the reward description match the actual "
                         "--reward-form (fixes the bug where it said min even under "
                         "pq_only), while the Guidance itself stays legacy. **objfix "
                         "forgets to fix the 4th Guidance line 'To raise the bottleneck "
                         "metric', so it keeps pointing at a bottleneck that does not "
                         "exist under pq_only/sb_only.** "
                         "objfix2=objfix plus making that 4th line match reward_form "
                         "too (a complete fix of the objective description; the content "
                         "of the Guidance is identical to legacy). "
                         "task=objfix plus dropping all Guidance that ranks actions, "
                         "handing over the rules of the search loop instead (greedy "
                         "acceptance, state unchanged on rejection, deterministic "
                         "render, de-esser gate). "
                         "**The LLM output changes**, so do not mix with existing runs "
                         "(llm_stats.guidance tells them apart).")
    # Output format + action catalogue (idea 4). Default OFF = legacy behavior (the
    # prompt is unchanged).
    ap.add_argument("--prompt-fewshot", dest="prompt_fewshot",
                    action="store_true", default=prompt_fewshot_default(),
                    help="Add to the system prompt an explicit output format plus a "
                         "catalogue of the 9 actions with the legal ranges of their "
                         "arguments. The catalogue is generated mechanically from the "
                         "tool specs, lists the 9 actions on equal footing in the same "
                         "order as the tool specs, and gives ranges only as "
                         "placeholders (e.g. gain_db=<-12..12>) without ever writing a "
                         "concrete value. It does not say which action is effective. "
                         "**The LLM output changes**, so do not mix with existing runs "
                         "(llm_stats.prompt_fewshot tells them apart).")
    ap.add_argument("--no-prompt-fewshot", dest="prompt_fewshot",
                    action="store_false",
                    help="Disable --prompt-fewshot (default, legacy behavior).")
    # Numeric formatting in the history block. Default legacy = legacy behavior (the
    # prompt is byte-for-byte unchanged).
    ap.add_argument("--prompt-numfmt", dest="prompt_numfmt",
                    choices=list(PROMPT_NUMFMT_MODES),
                    default=prompt_numfmt_default(),
                    help="Numeric formatting of the arguments shown in the "
                         "accepted/rejected history. "
                         "legacy=the previous %%.3g (freq=1000 becomes 'freq=1e+03', "
                         "and with 3 significant digits 1230 and 1234 collapse to the "
                         "same string). "
                         "plain=no exponent notation and no loss of digits. "
                         "It is a formatting-only fix that says nothing about which "
                         "action is good, but **the LLM output does change**, so do not "
                         "mix with existing runs (llm_stats.prompt_numfmt tells them "
                         "apart).")
    # How the history is presented (family 1). Default None = legacy behavior (the
    # prompt is byte-for-byte unchanged).
    ap.add_argument("--prompt-history", dest="prompt_history", default=None,
                    help="Specify in JSON **how** the accepted/rejected history is "
                         "presented. Example: "
                         "'{\"accepted_mode\":\"diverse\",\"rejected_mode\":\"distinct\"}'. "
                         "Keys: accepted_mode=recent|diverse|by_kind|none, "
                         "accepted_n, accepted_per_kind, "
                         "rejected_mode=recent|distinct|none, rejected_n, "
                         "forbid_last_n. In stalled runs all 8 entries of the accepted "
                         "history became an arithmetic progression of EQ moves "
                         "(bass 500->600->700->800->900 Hz), so the prompt itself was "
                         "teaching 'next is 1000 Hz' in-context. It says nothing about "
                         "which action is effective (all it passes is the model's own "
                         "action history and the determinism of the pipeline), so this "
                         "is not answer leakage. **The LLM output does change**, so do "
                         "not mix with existing runs (llm_stats.prompt_history tells "
                         "them apart).")
    # Family 4 (removing things). Both defaults are the legacy values = the prompt is
    # byte-for-byte unchanged.
    ap.add_argument("--prompt-system", dest="prompt_system",
                    choices=list(SYSTEM_MODES),
                    default=prompt_system_mode_default(),
                    help="How much of the system prompt to strip. "
                         "full=as before (default). "
                         "nosmall=strip only the single sentence 'Make small, "
                         "musically-motivated moves.' (the output when stalled is a "
                         "repetition of tiny 1 dB EQ moves, so this measures in "
                         "isolation the suspicion that this instruction induces the "
                         "repetition). "
                         "minimal=drop the persona and all advice, keeping only the "
                         "contract with the environment (one tool per move / adopt only "
                         "on improvement / no prose). "
                         "The wording of the reward formula follows --guidance, so this "
                         "is not confounded with the objective-function ablation. "
                         "**The LLM output does change**, so do not mix with existing "
                         "runs (llm_stats.prompt_system tells them apart).")
    ap.add_argument("--drop-guidance", dest="drop_guidance", action="store_true",
                    default=prompt_drop_guidance_default(),
                    help="Omit the prior-knowledge block at the end of the design prompt "
                         "entirely. Under legacy/objfix/objfix2 the 4 hand-written "
                         "lines ranking actions disappear; under task the rules of the "
                         "search loop disappear. It is orthogonal to --guidance (it "
                         "does not touch how the reward formula is written), so the "
                         "effect of 'removing' can be measured in isolation. task "
                         "**added** rules in exchange for dropping the 4 lines, so the "
                         "two were confounded. **The LLM output does change**, so do "
                         "not mix with existing runs (llm_stats.prompt_drop_guidance "
                         "tells them apart).")
    ap.add_argument("--no-drop-guidance", dest="drop_guidance",
                    action="store_false",
                    help="Disable --drop-guidance (default, legacy behavior).")
    return ap


def select_pilot_songs(plan: List[Dict[str, str]], n: int, seed: int) -> List[str]:
    """Deterministically sample n songs with split stratification (largest remainder)
    and return the list of song_ids.

    - The allocation per split is the floor of the proportional share plus one each in
      order of largest remainder (to make up the shortfall against n).
    - Within a split, song_ids sorted ascending are shuffled with a fixed seed and the
      first k are taken (**nested**: increasing n keeps the smaller-n set as a subset,
      pilot 20 subset of 100 subset of 150, so it can be extended on resume). Sampling
      would give a different set for every n and could not be reused.
    - Independent of the order of the plan and of chunking (sorted-based).
    """
    import random as _random

    by_split: Dict[str, List[str]] = {}
    for r in plan:
        by_split.setdefault(r["split"], []).append(r["song_id"])
    total = sum(len(v) for v in by_split.values())
    n = min(n, total)
    raw = {sp: n * len(ids) / total for sp, ids in by_split.items()}
    quota = {sp: int(raw[sp]) for sp in by_split}
    # +1 in order of largest remainder (ties broken deterministically by split name)
    rest = n - sum(quota.values())
    for sp in sorted(by_split, key=lambda s: (-(raw[s] - quota[s]), s))[:rest]:
        quota[sp] += 1
    chosen: List[str] = []
    for sp in sorted(by_split):
        ids = sorted(by_split[sp])
        # Nesting: take a prefix of the shuffle (fixed seed, deterministic). quota grows
        # monotonically with n since it is proportional, so shuffled[:k] keeps its
        # prefix as n increases  fix).
        shuffled = list(ids)
        _random.Random(seed).shuffle(shuffled)
        k = min(quota[sp], len(ids))
        chosen += shuffled[:k]
    return chosen


# ============================================================================
# LLM search proposer: the callable passed to the search_proposer hook of
# _propose_pipeline
# ============================================================================
def make_llm_search_proposer(llm, tool_specs: List[Dict[str, Any]],
                             max_invalid_retries: int,
                             stats: Dict[str, Any],
                             deesser_min_sibilance: float = 0.04,
                             show_budget: Optional[bool] = None,
                             stem_lufs_cache: Optional[StemLufsCache] = None,
                             anti_repeat: bool = False,
                             stall_window: int = 20,
                             forbid_n: int = 12,
                             rejected_distinct_n: int = 10,
                             guidance: Optional[str] = None,
                             reward_form: Optional[str] = None,
                             excerpt_sec: Optional[float] = None,
                             numfmt: Optional[str] = None,
                             fewshot: Optional[bool] = None,
                             history: Optional[Dict[str, Any]] = None,
                             system_mode: Optional[str] = None,
                             drop_guidance: Optional[bool] = None,
                             action_set: Optional[str] = None,
                             beat_grid_fn=None):
    """Return a callable compatible with the search_proposer hook (LLM tool-call
    proposals).

    The signature of the returned callable ``propose`` matches the hook call in
    agreement_loop_all._propose_pipeline:

        propose(state, pq, sb, reward, calib, accepted, rejected,
                step, n_search, tracks, sr) -> (name, validated_args)

    Returns (None, {}) when the proposal is invalid/gated/empty (the hook skips when
    nm is None). Counts of ok/empty/invalid/gated and the latencies are accumulated in
    ``stats`` (a per-song dict).

    Args:
        llm: the backend returned by build_llm (local/anthropic/mock).
        tool_specs: the 9 actions from _allowed_tool_specs().
        max_invalid_retries: how many times to re-ask after a schema/gate failure.
        stats: destination dict for the counters (initialized and passed in by the
            caller at make time).
        deesser_min_sibilance: lower bound on the sibilance ratio for the de-esser gate.
        show_budget: whether the prompt starts with "Step k/K" or with "Step k".
            None uses the environment-variable default (= True, legacy behavior).
            False changes the LLM output (the prefix property of greedy search is
            restored, at the cost of incompatibility with existing runs).
        stem_lufs_cache: memo for the per-stem LUFS. If None, a fresh one is created
            for this song. **Results are bit-identical** (a memo of a pure function of
            the frozen state).
        anti_repeat: structurally forbid repeated proposals (default False = legacy
            behavior). When True, without touching the search algorithm ((1+1) greedy
            with a strict-improvement gate) at all, the following happens **inside the
            proposer**:
              1. The REJECTED block of the prompt becomes "N distinct proposals"
                 (identical proposals are folded into ``xN``).
              2. Any (action,args) **already evaluated and rejected from the current
                 best_state** is explicitly forbidden in the prompt, and if the LLM
                 returns it anyway it is bounced back **before rendering** so that it
                 proposes again (the same path as the existing invalid/gate re-ask).
                 The pipeline is deterministic, so re-evaluating the same proposal at
                 the same state gives a bit-identical reward and is always rejected.
                 Forbidding it therefore costs the search nothing; all that is saved is
                 a wasted render + scoring. The forbidden set is discarded once
                 best_state moves (the same action gives a different result at a
                 different state).
              3. The model's own tool-usage histogram and its unused tools are shown in
                 the prompt.
              4. A stall warning is emitted once ``stall_window`` consecutive moves have
                 been rejected since the last acceptance.
            If every re-ask is a duplicate, the last proposal is returned as-is (i.e.
            the step is not thrown away). Hence **the number of evaluated moves is the
            same as before**.
        stall_window: number of consecutive rejections that triggers the stall warning.
        forbid_n: maximum number of forbidden proposals enumerated in the prompt (newest
            first).
        rejected_distinct_n: number of distinct proposals shown in the REJECTED block.
        guidance: composition of the prior-knowledge block of the prompt
            ('legacy'/'objfix'/'task'). None uses the environment variable
            AUTOMIX_GUIDANCE, and 'legacy' when that is unset (legacy behavior, the
            prompt is byte-for-byte unchanged). See agreement_loop_qwen.GUIDANCE_MODES
            for details.
        reward_form: 'pq_only'/'sb_only'/'min'/'weighted'. When guidance is anything
            other than 'legacy', the reward formula in the prompt is made to match this
            (unused under legacy).
        excerpt_sec: length of the excerpt used for search [s]. Used under
            guidance='task' to state explicitly that "the metrics are measured on the
            excerpt" (unused under legacy/objfix).
        fewshot: idea 4. Add to the system prompt a block with the output format plus a
            catalogue of the 9 actions and their argument ranges (default None =
            environment variable AUTOMIX_PROMPT_FEWSHOT, and False when that is unset =
            legacy behavior with a byte-for-byte unchanged system prompt).
            The catalogue is generated mechanically from the tool specs, lists the 9
            actions on equal footing in the same order as the tool specs, and gives
            ranges only as placeholders without writing concrete values.
            It says nothing about which action is effective. See the discussion at
            agreement_loop_qwen.PROMPT_FEWSHOT_ENV for details.
        action_set: the action space ('base'/'automation'). Default None = the
            environment variable AUTOMIX_ACTION_SET, and 'base' (the conventional 9
            actions) when that is unset.
            It **must be the same value** as the one used to build ``tool_specs``
            (the allowed set of _validate_action is derived from it, so a mismatch means
            the tools shown to the LLM get rejected during validation).
        beat_grid_fn: a callable returning ``() -> (beat_times_rel, tempo_bpm)``, or
            None (default = do not show the beat grid in the prompt = identical to
            before).
            **Why it is a callable**: the proposer is built at the top of the song loop,
            but the excerpt window and the beats are resolved inside
            ``_run_one_song_random``, so the values do not exist yet at construction
            time. They are fetched lazily when propose() runs.
            The beat times passed in must be **seconds relative to the start of the
            excerpt**.
    """
    cache = StemLufsCache() if stem_lufs_cache is None else stem_lufs_cache
    aset = action_set_default() if action_set is None else str(action_set)
    # A mismatch between tool_specs and the allowed set is a silent accident, so kill it
    # at construction time.
    _allow = set(allowed_actions(aset))
    _shown = {s["name"] for s in tool_specs}
    if not _shown <= _allow:
        raise ValueError(
            f"tool_specs contains actions that are not allowed: {sorted(_shown - _allow)} "
            f"(action_set={aset!r}). Pass the same action_set as the one given to "
            f"_allowed_tool_specs.")
    gmode = guidance_mode_default() if guidance is None else guidance
    if gmode not in GUIDANCE_MODES:
        raise ValueError(f"unknown guidance={gmode!r}; valid={GUIDANCE_MODES}")
    # Numeric formatting. **The prompt and the proposal signature (_sig) must always use
    # the same value.** A mismatch makes the strings in the REJECTED block and the HARD
    # CONSTRAINT block disagree, and the anti-repeat ban becomes meaningless from the
    # LLM's point of view.
    nfmt = prompt_numfmt_default() if numfmt is None else numfmt
    if nfmt not in PROMPT_NUMFMT_MODES:
        raise ValueError(f"unknown numfmt={nfmt!r}; valid={PROMPT_NUMFMT_MODES}")
    # How the history is presented (family 1). None uses the environment variable, and
    # the legacy presentation when that is unset.
    hcfg = prompt_history_default() if history is None else history
    if hcfg is not None:
        # Validate keys/values here even when a dict was passed directly (fail-fast).
        hcfg = parse_prompt_history(json.dumps(hcfg))
    # Family 4 (removing things): how much of the system prompt to strip / whether to
    # drop the prior-knowledge block.
    # Both default to the legacy values, so the prompt is byte-for-byte unchanged.
    smode = prompt_system_mode_default() if system_mode is None else str(system_mode)
    if smode not in SYSTEM_MODES:
        raise ValueError(f"unknown system_mode={smode!r}; valid={SYSTEM_MODES}")
    drop_g = (prompt_drop_guidance_default() if drop_guidance is None
              else bool(drop_guidance))
    # The system prompt follows guidance too ('legacy' together with system_mode='full'
    # keeps the None -> use the legacy constant as-is path).
    sys_prompt = (None if (gmode == "legacy" and smode == "full")
                  else build_system_prompt(reward_form, gmode, system_mode=smode))
    # Idea 4: append the format + catalogue block to the system prompt (default OFF).
    use_fewshot = prompt_fewshot_default() if fewshot is None else bool(fewshot)
    if use_fewshot:
        sys_prompt = ((sys_prompt or _SYSTEM_PROMPT)
                      + fewshot_system_suffix())
    # ---- within-song state for anti-repeat (closure) ----
    #   sigs_at_state: signatures of proposals already evaluated and rejected from the
    #                  current best_state -> order of appearance
    #   tried:         per-tool proposal counts over the whole song (to enumerate the
    #                  unused tools)
    #   n_acc / acc_step: the step at which accepted last grew (to compute the stall
    #                  length)
    ar_state: Dict[str, Any] = {
        "sigs_at_state": {}, "tried": {}, "n_acc": None, "acc_step": 0,
    }
    all_tool_names = [s["name"] for s in tool_specs]

    def _sig(name: str, args: Dict[str, Any]) -> str:
        # Using exactly the same formatting as the REJECTED block (_fmt_args, schema
        # order of the validated args) makes the two blocks in the prompt agree as
        # strings.
        return f"{name}({_fmt_args(args, nfmt)})"

    def propose(state, pq: float, sb: float, reward: float, calib,
                accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]],
                step: int, n_search: int, tracks: List[str], sr: int,
                rand_proposal=None  # CRN hook compatibility (unused by the LLM proposer)
                ) -> Tuple[Optional[str], Dict[str, Any]]:
        # ---- build the context (same shape as agreement_loop_qwen) ----
        # state (= best_state) only changes on acceptance, so the memo pays off.
        # The return value is bit-identical to the uncached implementation, so the
        # prompt does not change by a single character.
        stem_lufs = _per_stem_lufs(state, sr, cache=cache)
        z_pq = calib.z_pq(pq)
        z_sb = calib.z_sb(sb)
        # accepted (= best_chain) carries a 'delta' on every entry (added by the hook).
        # _build_user_prompt reads accepted[*]['delta'], so fill it in when missing.
        accepted_ctx: List[Dict[str, Any]] = []
        for r in accepted:
            e = dict(r)
            e.setdefault("delta", float(r.get("reward", 0.0)) - 0.0)
            accepted_ctx.append(e)
        # ---- anti-repeat context (default OFF: ar_ctx=None -> prompt identical to
        # before) ----
        ar_ctx: Optional[Dict[str, Any]] = None
        if anti_repeat:
            # If best_state moved (= accepted grew), the "already rejected at this state"
            # set is invalid and is discarded. The stall baseline is updated here too.
            n_acc = len(accepted)
            if ar_state["n_acc"] is None or n_acc != ar_state["n_acc"]:
                ar_state["n_acc"] = n_acc
                ar_state["acc_step"] = step
                ar_state["sigs_at_state"] = {}
            stall = step - int(ar_state["acc_step"])
            tried: Dict[str, int] = ar_state["tried"]
            untried = [t for t in all_tool_names if t not in tried]
            forb_all = list(ar_state["sigs_at_state"].keys())
            ar_ctx = {
                "rejected_distinct_n": rejected_distinct_n,
                "forbidden": forb_all[-forbid_n:],
                "tried_counts": dict(tried),
                "untried": untried,
                "stall": stall,
                "stall_window": stall_window,
            }
            if stall >= stall_window:
                stats["stall_steps"] = stats.get("stall_steps", 0) + 1

        # Beat grid. With no beat_grid_fn this is equivalent to (None, None) and the
        # prompt stays byte-for-byte identical to before.
        beats, bpm = (beat_grid_fn() if beat_grid_fn is not None else (None, None))

        user = _build_user_prompt(
            stem_lufs, pq, sb, z_pq, z_sb, reward,
            accepted_ctx, rejected, step + 1, n_search,
            show_budget=show_budget, anti_repeat=ar_ctx,
            guidance=gmode, reward_form=reward_form, excerpt_sec=excerpt_sec,
            numfmt=nfmt, history=hcfg, drop_guidance=drop_g,
            beat_times=beats, tempo_bpm=bpm)

        # ---- LLM proposal (re-ask up to max_invalid_retries times on schema/gate
        # failures) ----
        for attempt in range(max_invalid_retries + 1):
            t0 = time.perf_counter()
            # Drive the async backend from this synchronous loop.
            name, args, _raw, _meta = asyncio.run(
                _qwen_propose(llm, user, tool_specs, system=sys_prompt))
            stats.setdefault("latency", []).append(time.perf_counter() - t0)
            served = (_meta or {}).get("model")
            if served:
                sm = stats.setdefault("served_models", {})
                sm[served] = sm.get(served, 0) + 1
            fr = (_meta or {}).get("finish_reason")
            if fr:
                fm = stats.setdefault("finish_reasons", {})
                fm[fr] = fm.get(fr, 0) + 1

            if name is None:
                stats["tool_call_empty"] = stats.get("tool_call_empty", 0) + 1
                user += ("\n\n[system] Your previous reply contained no tool call. "
                         "You MUST call exactly one tool now.")
                continue

            ok, vargs, reason = _validate_action(
                name, args, tracks, action_set=aset, excerpt_sec=excerpt_sec)
            if not ok:
                stats["invalid"] = stats.get("invalid", 0) + 1
                user += (f"\n\n[system] Your proposal {name}({args}) was invalid: "
                         f"{reason}. Propose a valid action.")
                continue

            # de-esser gate: reject before rendering if the target vocal stem has no
            # sibilance.
            if name == "apply_deesser":
                allow, _ratio, gate_reason = _deesser_gate(
                    state, vargs, sr, deesser_min_sibilance)
                if not allow:
                    stats["gated"] = stats.get("gated", 0) + 1
                    user += (f"\n\n[system] {gate_reason}. Propose a different action.")
                    continue

            # ---- duplicate guard: bounce back proposals already evaluated and rejected
            # at the current best_state ----
            # The pipeline is deterministic, so the same state + the same (action,args)
            # always gives the same reward -> always rejected. Re-evaluating has zero
            # information gain, so we ask for another proposal before rendering.
            # The prompt changing **within a step** here is the only way out of the fixed
            # point of greedy search at T=0 (prompt changes across steps alone do not
            # escape it).
            if anti_repeat:
                sig = _sig(name, vargs)
                if sig in ar_state["sigs_at_state"]:
                    if attempt < max_invalid_retries:
                        stats["dup_reask"] = stats.get("dup_reask", 0) + 1
                        user += (
                            f"\n\n[system] You proposed {sig} again. That exact "
                            "action was ALREADY evaluated from the current mix state "
                            "and rejected; the pipeline is deterministic so it will "
                            "be rejected again with the identical reward. It is "
                            "forbidden. Propose a DIFFERENT action now — change the "
                            "tool, the target stem, or the parameter values.")
                        continue
                    # Still a duplicate after exhausting the re-asks -> do not throw the
                    # step away, send it to evaluation as before (this keeps the number
                    # of evaluated moves aligned with legacy runs and leaves the meaning
                    # of the budget unchanged).
                    stats["dup_forced"] = stats.get("dup_forced", 0) + 1
                ar_state["sigs_at_state"][sig] = ar_state["sigs_at_state"].get(sig, 0) + 1
                ar_state["tried"][name] = ar_state["tried"].get(name, 0) + 1

            stats["ok"] = stats.get("ok", 0) + 1
            return name, vargs

        # Invalid on every attempt -> no proposal for this step (the hook skips it).
        return None, {}

    return propose


def make_hybrid_search_proposer(llm, tool_specs: List[Dict[str, Any]],
                                max_invalid_retries: int, stats: Dict[str, Any],
                                switch_step: int,
                                stall_n: int = 0,
                                deesser_min_sibilance: float = 0.04,
                                show_budget: Optional[bool] = None,
                                stem_lufs_cache: Optional[StemLufsCache] = None):
    """hybrid proposer: LLM proposals while step<switch, random proposals once
    step>=switch.

    Combines the "quality of a single shot" of the LLM's opening moves (an advantage at
    k<=6 on the budget curve) with the exploration diversity of continuing at random
    (an advantage at k>=8). The LLM part reuses make_llm_search_proposer, and its stats
    (tool_call success rate, latency, etc.) are recorded in the same dict.

    **CRN (common random numbers)**: the random part uses, as-is, the proposal drawn by
    _propose_pipeline at every step and handed over via `rand_proposal` (= the same rng
    sequence and the same seed_idx dependence as the pure-random run). This cancels the
    variance of the shared random search out of the per-song paired difference
    'hybrid - pure_random' and maximizes statistical power (review .
    The old implementation, which used an independent rng of its own, broke CRN and had
    lower power.
    """
    llm_propose = make_llm_search_proposer(
        llm, tool_specs, max_invalid_retries, stats,
        deesser_min_sibilance=deesser_min_sibilance,
        show_budget=show_budget, stem_lufs_cache=stem_lufs_cache)

    # Permanent-switch flag (one proposer per song, so it lives in the closure)
    _switched = [False]

    def propose(state, pq: float, sb: float, reward: float, calib,
                accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]],
                step: int, n_search: int, tracks: List[str], sr: int,
                rand_proposal=None
                ) -> Tuple[Optional[str], Dict[str, Any]]:
        if not _switched[0]:
            trigger = None
            if step >= switch_step:
                trigger = f"fixed_step_{switch_step}"
            elif stall_n > 0:
                # Measure the no-acceptance length from the step of the last entry of
                # accepted (= best_chain). If there has never been an acceptance, step
                # itself is the no-acceptance length.
                last = int(accepted[-1].get("step", -1)) if accepted else -1
                if step - last > stall_n:
                    trigger = f"stall_{stall_n}"
            if trigger is not None:
                _switched[0] = True
                stats["hybrid_switched_at"] = step
                stats["hybrid_switch_trigger"] = trigger
                print(f"[hybrid] step={step}: permanently switched to random ({trigger})",
                      flush=True)
        if not _switched[0]:
            return llm_propose(state, pq, sb, reward, calib, accepted, rejected,
                               step, n_search, tracks, sr, rand_proposal=rand_proposal)
        # After the switch: use the random proposal drawn by the pipeline (the same
        # sequence as pure-random).
        if rand_proposal is None:
            return None, {}
        return rand_proposal

    return propose


def assert_prompt_budget_consistent(run_dir, args) -> None:
    """Check that ``prompt_show_budget`` of an existing run matches this invocation's
    arguments.

    Same fail-fast idea as ``ALL.assert_corruption_consistent``.
    Whether the total budget is shown in the prompt **changes the LLM's output**, so
    mixing both within one run breaks the prefix assumption that "the prefix at budget k
    = the first k moves of a run with budget K", and the JSON only reveals it quietly.
    This mechanically stops you from forgetting the flag on resume.

    Backward compatibility (fixed :
      * Songs with no ``llm_stats`` at all (e.g. the random arm, songs that use no
        prompt) are out of scope.
      * Songs that have ``llm_stats`` but no ``prompt_show_budget`` key come from a
        **run predating this field**, and back then the total budget was always shown.
        They are therefore treated as ``True``. Previously these were skipped too, which
        let the **most likely accident of all** through: resuming an existing run (with
        no key) while adding ``--no-prompt-show-budget``
        (none of the existing LLM sweeps under outputs/runs carry the key).

    Can be deliberately disabled with the environment variable
    ``ALLOW_PROMPT_BUDGET_MISMATCH=1``.
    """
    import json as _json

    if os.environ.get("ALLOW_PROMPT_BUDGET_MISMATCH") == "1":
        print("[guard] ALLOW_PROMPT_BUDGET_MISMATCH=1: skipping the prompt budget "
              "consistency check", flush=True)
        return
    songs_dir = run_dir / "songs"
    if not songs_dir.is_dir():
        return
    want = bool(args.prompt_show_budget)
    mismatches: List[str] = []
    for p in sorted(songs_dir.glob("*.json")):
        try:
            rec = _json.loads(p.read_text("utf-8"))
        except (OSError, _json.JSONDecodeError):
            continue
        if rec.get("status") != "done":
            continue
        stats = rec.get("llm_stats")
        if not isinstance(stats, dict):
            continue                            # song that uses no prompt (random arm)
        have = stats.get("prompt_show_budget")
        if have is None:
            # Old JSON. _build_user_prompt always wrote "Step k/K" back then, so True is
            # the fact for the existing run. Skipping here would miss a forgotten flag.
            have = True
            shown = "True (old JSON: no key = the default before this was introduced)"
        else:
            shown = repr(have)
        if bool(have) != want:
            mismatches.append(f"  {rec.get('song_id')}: existing={shown} now={want!r}")
    if mismatches:
        head = mismatches[:10]
        more = (f"\n  ... and {len(mismatches)-10} more" if len(mismatches) > 10 else "")
        raise SystemExit(
            f"[guard] prompt_show_budget disagrees with the existing run "
            f"({len(mismatches)} songs). Resuming would mix songs with different "
            "prompts and break the prefix property of greedy search (the result at "
            "budget k = the first k moves of a longer run):\n"
            + "\n".join(head) + more
            + "\n  -> Match --prompt-show-budget / --no-prompt-show-budget to the "
              "existing run, or use a different --run-name."
              "\n  -> Only when mixing them deliberately, set "
              "ALLOW_PROMPT_BUDGET_MISMATCH=1.")


def assert_action_set_consistent(run_dir, args, action_set: str) -> None:
    """Check that ``action_set`` of an existing run matches this invocation.

    Same fail-fast as ``assert_prompt_budget_consistent``. The action space **is the
    search space itself**, so if songs with 9 actions and songs with 10 actions are
    mixed in one run, a comparison such as "84 acceptances vs N" becomes meaningless.
    And you cannot tell without opening the JSON and looking at
    ``llm_stats.action_set``.

    Backward compatibility: songs with no ``llm_stats`` (random arm) are out of scope.
    Songs with no ``action_set`` key come from a **run predating this field**, and back
    then there were always 9 actions, so they are treated as ``'base'``.

    Can be deliberately disabled with the environment variable
    ``ALLOW_ACTION_SET_MISMATCH=1``.
    """
    import json as _json

    if os.environ.get("ALLOW_ACTION_SET_MISMATCH") == "1":
        print("[guard] ALLOW_ACTION_SET_MISMATCH=1: skipping the action space "
              "consistency check", flush=True)
        return
    songs_dir = run_dir / "songs"
    if not songs_dir.is_dir():
        return
    mismatches: List[str] = []
    for p in sorted(songs_dir.glob("*.json")):
        try:
            rec = _json.loads(p.read_text("utf-8"))
        except (OSError, _json.JSONDecodeError):
            continue
        if rec.get("status") != "done":
            continue
        stats = rec.get("llm_stats")
        if not isinstance(stats, dict):
            continue                            # song that uses no prompt (random arm)
        have = stats.get("action_set")
        shown = repr(have) if have is not None else "'base' (old JSON: no key)"
        if (have or "base") != action_set:
            mismatches.append(
                f"  {rec.get('song_id')}: existing={shown} now={action_set!r}")
    if mismatches:
        head = mismatches[:10]
        more = (f"\n  ... and {len(mismatches)-10} more" if len(mismatches) > 10 else "")
        raise SystemExit(
            f"[guard] action_set disagrees with the existing run "
            f"({len(mismatches)} songs). Resuming would mix in **songs with a different "
            f"action space**, and comparisons of acceptance counts and PQ would no "
            f"longer hold:\n"
            + "\n".join(head) + more
            + "\n  -> Match --action-set to the existing run, or use a different "
              "--run-name."
              "\n  -> Only when mixing them deliberately, set "
              "ALLOW_ACTION_SET_MISMATCH=1.")


def _new_stats() -> Dict[str, Any]:
    return {"ok": 0, "invalid": 0, "gated": 0, "tool_call_empty": 0, "latency": []}


def _stats_summary(stats: Dict[str, Any], backend: str, llm_model: str,
                   show_budget: Optional[bool] = None,
                   stem_lufs_cache: Optional[StemLufsCache] = None,
                   llm: Any = None,
                   anti_repeat: Optional[Dict[str, Any]] = None,
                   guidance: Optional[str] = None,
                   reward_form: Optional[str] = None,
                   prompt_fewshot: Optional[bool] = None,
                   prompt_numfmt: Optional[str] = None,
                   prompt_history: Optional[Dict[str, Any]] = None,
                   prompt_system: Optional[str] = None,
                   prompt_drop_guidance: Optional[bool] = None,
                   action_set: Optional[str] = None,
                   allowed_action_names: Optional[List[str]] = None,
                   beat_grid: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    lat = stats.get("latency", [])
    cache = stem_lufs_cache
    # If the backend exposes it, record the sampling configuration actually sent.
    # temperature/seed change the behavior of the search itself, so a run carrying this
    # cannot be mixed with one that does not (same treatment as prompt_show_budget).
    sampling = None
    getter = getattr(llm, "sampling_config", None)
    if callable(getter):
        try:
            sampling = getter()
        except Exception:                                       # noqa: BLE001
            sampling = None
    return {
        "sampling": sampling,
        # Action space . 'base' is the conventional 9 actions. 'automation'
        # is the 10 actions obtained by adding apply_gain_automation, and since **the
        # search space itself differs**, acceptance counts and PQ must not be compared
        # directly against a 'base' run.
        "action_set": action_set,
        "allowed_actions": allowed_action_names,
        # Whether the beat grid was shown in the prompt . None = not shown
        # (= the same prompt as before). Runs that showed it have a different prompt
        # string.
        "beat_grid": beat_grid,
        # Prior-knowledge block of the prompt (idea 2). 'legacy' is the previous prompt.
        # 'objfix'/'task' change the prompt string and cannot be mixed with existing
        # runs.
        "guidance": guidance,
        "guidance_reward_form": reward_form,
        # Numeric formatting of the history block (family 2). 'legacy' is the previous
        # prompt (%.3g). 'plain' drops the exponent notation and the loss of digits, so
        # the prompt string changes.
        "prompt_numfmt": prompt_numfmt,
        # How the accepted/rejected history is presented (family 1). None = the legacy
        # presentation (accepted = the raw time series of the last 8, rejected = the raw
        # time series of the last 10). Runs with a non-None value have a different prompt
        # string and cannot be mixed with None runs.
        "prompt_history": prompt_history,
        # How much of the system prompt was stripped (family 4). 'full' is the previous
        # behavior. 'nosmall'/'minimal' runs have a different system prompt string.
        "prompt_system": prompt_system,
        # Whether the prior-knowledge block at the end of the design prompt was dropped
        # (family 4). False is the previous behavior. True runs carry neither the
        # hand-written action ranking nor (under task) the search rules.
        "prompt_drop_guidance": (None if prompt_drop_guidance is None
                                 else bool(prompt_drop_guidance)),
        # Output format + action catalogue block (idea 4). True runs have a different
        # system prompt and cannot be mixed with False runs.
        "prompt_fewshot": (None if prompt_fewshot is None else bool(prompt_fewshot)),
        # Settings and firing counts of the repeat ban (idea 3). None = disabled (the
        # legacy prompt).
        "anti_repeat": anti_repeat,
        "dup_reask": int(stats.get("dup_reask", 0)) if anti_repeat else None,
        "dup_forced": int(stats.get("dup_forced", 0)) if anti_repeat else None,
        "stall_steps": int(stats.get("stall_steps", 0)) if anti_repeat else None,
        # Breakdown of finish_reason (to detect truncation; a 'length' means max_tokens
        # is too small).
        "finish_reasons": stats.get("finish_reasons") or None,
        "backend": backend,
        "llm_model": llm_model,
        "ok": int(stats.get("ok", 0)),
        "invalid": int(stats.get("invalid", 0)),
        "gated": int(stats.get("gated", 0)),
        "tool_call_empty": int(stats.get("tool_call_empty", 0)),
        "llm_calls": len(lat),
        "latency_mean_sec": (round(float(np.mean(lat)), 3) if lat else None),
        # Breakdown of the model names actually served (to detect divergence from the
        # requested model).
        "served_models": stats.get("served_models") or None,
        # Whether the total budget was shown in the prompt (presence of the prefix
        # property = essential for telling run compatibility apart).
        "prompt_show_budget": (None if show_budget is None else bool(show_budget)),
        # How well the per-stem LUFS memo works (telemetry that has no effect whatsoever
        # on the behavior of the search).
        "stem_lufs_cache": (cache.stats() if cache is not None else None),
    }


# ============================================================================
# main: follow the random path of agreement_loop_all.main and inject search_proposer
# ============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not ALL.is_random_mode(args):
        raise SystemExit(
            "size_sweep_run assumes random mode. Please specify "
            "--random-gain-db-range (or --random-init-gain-range).")

    # ---- argument consistency for informed_random (fail-fast; bail out before creating
    # run_dir) ----
    # Same philosophy as the corruption guard: silently accepting a missing or a
    # superfluous flag means you can no longer tell which prior a run was made with
    # without opening the JSON afterwards.
    if (args.proposer == "informed_random"
            and not getattr(args, "proposer_stats", None)):
        raise SystemExit(
            "--proposer informed_random requires --proposer-stats <json> "
            "(e.g. experiments/informed_prior_stats.json).")
    if (args.proposer != "informed_random"
            and getattr(args, "proposer_stats", None)):
        raise SystemExit(
            f"--proposer-stats is only for --proposer informed_random "
            f"(you specified --proposer {args.proposer}).")

    # ---- consistency of the action space and excerpt mode (fail-fast) --------------
    # The breakpoints of automation are absolute times, so a chain searched on an
    # excerpt cannot be transferred onto the full song (swap_state_stems refuses it).
    # Noticing that **at the very end**, after finishing a 500-move search, would throw
    # away the entire GPU time.
    # This can be decided from the arguments alone, so bail out before creating run_dir.
    aset = (action_set_default() if getattr(args, "action_set", None) is None
            else str(args.action_set))
    _exc_sec = float(getattr(args, "search_excerpt_sec", 0.0) or 0.0)
    _exc_only = getattr(args, "excerpt_only", None)
    if _exc_only is None:
        _exc_only = ALL.excerpt_only_default()
    _exc_only = bool(_exc_only)
    if (args.proposer in ("llm", "hybrid") and action_set_is_time_absolute(aset)
            and _exc_sec > 0 and not _exc_only):
        raise SystemExit(
            f"[sweep] action_set={aset!r} contains an action with absolute times "
            f"(breakpoints). In excerpt mode with --search-excerpt-sec={_exc_sec} the "
            f"chain cannot be transferred onto the full-song stems, so "
            f"--excerpt-only (or AUTOMIX_EXCERPT_ONLY=1) is mandatory.\n"
            f"  -> Either add --excerpt-only and report only the 12-second excerpt "
            f"results,\n"
            f"  -> or go back to --action-set base.\n"
            f"  See the discussion at agreement_loop_all.EXCERPT_ONLY_ENV for details.")
    if bool(getattr(args, "excerpt_beats", False)) and _exc_sec <= 0:
        raise SystemExit(
            "[sweep] --excerpt-beats is for excerpt mode only "
            "(please specify --search-excerpt-sec > 0).")

    # Override the model of the local/gemini backend from the CLI (build_llm reads env).
    if args.backend == "local" and args.llm_model:
        os.environ["LOCAL_LLM_MODEL"] = args.llm_model
    if args.backend == "gemini" and args.llm_model:
        os.environ["GEMINI_MODEL"] = args.llm_model
    if args.backend == "local":
        llm_model = args.llm_model or os.environ.get("LOCAL_LLM_MODEL", "default")
    elif args.backend == "anthropic":
        if args.llm_model:
            os.environ["ANTHROPIC_MODEL"] = args.llm_model
        llm_model = (os.environ.get("ANTHROPIC_VERTEX_MODEL")
                     or os.environ.get("ANTHROPIC_MODEL")
                     or ("claude-haiku-4-5@20251001"
                         if os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
                         else "claude-opus-4-7"))
    elif args.backend == "gemini":
        llm_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    else:
        llm_model = "mock-deterministic-v1"

    splits = ALL.parse_splits(args.splits)
    run_dir = ROOT / "outputs" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    plan = ALL.build_song_plan(splits, dataset=args.dataset,
                               offset=args.offset, limit=args.limit)
    # To keep the random-mode seed stable regardless of resume / chunking
    # (offset/limit) / pilot selection, song_index is built from the full plan before
    # offset/limit are applied (the same review fix as agreement_loop_all, ;
    # runs without offset are unchanged).
    song_index_of = ALL.build_full_plan_song_index(splits, dataset=args.dataset)
    pilot_ids: Optional[List[str]] = None
    if args.pilot_n is not None:
        pilot_ids = select_pilot_songs(plan, args.pilot_n, args.pilot_seed)
        plan_exec = [r for r in plan if r["song_id"] in set(pilot_ids)]
    else:
        plan_exec = plan
    # Check the consistency of the corruption settings before resuming (prevents a
    # recurrence of the 32B/72B accident of 
    ALL.assert_corruption_consistent(run_dir, args)
    # Likewise check the consistency of showing the total budget in the prompt (to
    # protect the prefix property)
    assert_prompt_budget_consistent(run_dir, args)
    # Consistency of the action space. Stops songs with 9 actions and songs with 10
    # actions from being mixed in one run.
    assert_action_set_consistent(run_dir, args, aset)
    # Consistency of the search domain (full song / N-second excerpt). Mandatory because
    # the objective function itself changes.
    ALL.assert_search_domain_consistent(run_dir, args)
    # The effective base seed passed to random_seed_for. --random-base-seed (added to
    # agreement_loop_all on  is an additive shift on --seed, and equals the
    # previous value at its default of 0.
    eff_base_seed = int(args.seed) + int(getattr(args, "random_base_seed", 0) or 0)
    # ---- informed_random: load the empirical prior and check resume consistency ----
    # Lazy import: do not import the new module unless we are in informed_random
    # (running jobs keep importing this file in fresh processes, so we do not add even
    # one module to the import set of the default path).
    informed_stats = None
    informed_sha: Optional[str] = None
    informed_prov: Optional[Dict[str, Any]] = None
    if args.proposer == "informed_random":
        from informed_random_proposer import (
            assert_informed_stats_consistent,
            build_provenance,
            load_action_stats,
            make_informed_random_proposer,
        )
        informed_stats, informed_sha = load_action_stats(args.proposer_stats)
        assert_informed_stats_consistent(run_dir, informed_sha)
        informed_prov = build_provenance(informed_stats, args.proposer_stats,
                                         informed_sha, eff_base_seed)
    pending = ALL.filter_pending(plan_exec, run_dir, resume=args.resume)
    print(f"[sweep] splits={splits} plan={len(plan)} "
          f"pilot={len(plan_exec) if pilot_ids is not None else '-'} "
          f"pending={len(pending)} "
          f"(offset={args.offset} limit={args.limit} resume={args.resume} "
          f"pilot_n={args.pilot_n} pilot_seed={args.pilot_seed}) "
          f"backend={args.backend} model={llm_model}", flush=True)
    _db = getattr(args, "random_gain_db_range", None)
    _gain_desc = (f"gain_db_range={tuple(_db)} (dBFS)" if _db is not None
                  else f"gain_range={tuple(args.random_init_gain_range)} (linear)")
    print(f"[sweep] RANDOM mode: {_gain_desc} seeds={args.random_seeds} "
          f"n_search={args.n_search} run={args.run_name}", flush=True)
    for r in pending:
        print(f"    - {r['split']:5s} {r['song_id']}")

    if args.dry_run:
        print("[sweep] --dry-run: exiting without scoring (no GPU needed)")
        ALL._write_json_atomic(
            run_dir / "plan.json",
            {"splits": splits, "plan": plan, "pending": pending,
             "pilot_n": args.pilot_n, "pilot_seed": args.pilot_seed,
             "pilot_ids": pilot_ids,
             "random_mode": True, "backend": args.backend, "llm_model": llm_model,
             "prompt_show_budget": bool(args.prompt_show_budget)})
        return 0

    # ---- LLM backend (no proxies; the real backend). Not built for the random arm ----
    if args.proposer in ("llm", "hybrid"):
        # tracks is used when validating per-stem proposals but is not needed to build
        # the backend (empty is fine).
        llm = build_llm(args.backend, tracks=[])
        tool_specs = _allowed_tool_specs(action_set=aset)
        extra = (f" hybrid_switch_step={args.hybrid_switch_step}"
                 if args.proposer == "hybrid" else "")
        print(f"[sweep] proposer={args.proposer} backend={args.backend} "
              f"model={getattr(llm, 'model', '?')} "
              f"action_set={aset} "
              f"allowed_actions={len(tool_specs)}{extra} "
              f"prompt_show_budget={bool(args.prompt_show_budget)} "
              f"prompt_fewshot={bool(getattr(args, 'prompt_fewshot', False))} "
              f"guidance={args.guidance} "
              f"prompt_numfmt={getattr(args, 'prompt_numfmt', None)} "
              f"prompt_system={getattr(args, 'prompt_system', None)} "
              f"drop_guidance={bool(getattr(args, 'drop_guidance', False))}",
              flush=True)
    elif args.proposer == "informed_random":
        llm = None
        tool_specs = []
        print(f"[sweep] proposer=informed_random (no LLM; static empirical prior "
              f"stats={args.proposer_stats} sha256={informed_sha[:12]}... "
              f"base_seed={eff_base_seed})", flush=True)
    else:
        llm = None
        tool_specs = []
        print("[sweep] proposer=random (no LLM, the random path of "
              "agreement_loop_all)", flush=True)

    # ---- scorers (default local = in-process as before, no proxies) ----
    # With --scorer remote the work is sent to a shared scoring server and the search
    # process never imports torch, so it uses no GPU memory. Construction is centralized
    # in ALL.build_scorers (we go through the same function as agreement_loop_all.py,
    # because **fixing only one of them would make the behavior depend on the search
    # path**).
    scorers = ALL.build_scorers(getattr(args, "scorer", "local"),
                                getattr(args, "scorer_addr", None),
                                int(getattr(args, "scorer_client_index", 0) or 0))

    # Settings for idea 3 (the structural ban on repetition). Records None when disabled.
    ar_cfg: Optional[Dict[str, Any]] = None
    if getattr(args, "anti_repeat", False) and args.proposer in ("llm", "hybrid"):
        ar_cfg = {"enabled": True,
                  "stall_window": int(args.anti_repeat_stall_window),
                  "forbid_n": int(args.anti_repeat_forbid_n),
                  "rejected_distinct_n": int(args.anti_repeat_rejected_n)}
        print(f"[sweep] anti_repeat=ON {ar_cfg}", flush=True)

    # Settings for family 1 (how the history is presented). Records None when unspecified
    # (= the legacy presentation).
    hist_cfg: Optional[Dict[str, Any]] = None
    if getattr(args, "prompt_history", None) and args.proposer in ("llm", "hybrid"):
        hist_cfg = parse_prompt_history(args.prompt_history)
        print(f"[sweep] prompt_history={hist_cfg}", flush=True)

    index_path = run_dir / "index.json"
    done_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(pending):
        print(f"[sweep] ({i+1}/{len(pending)}) {row['split']} {row['song_id']}",
              flush=True)
        # llm/hybrid proposer: create stats per song and inject a proposer that captures
        # those stats. random proposer: leave search_proposer=None (the default path).
        stats = _new_stats()
        si = song_index_of[row["song_id"]]
        # The per-stem LUFS memo is **created fresh for every song** (so it does not keep
        # holding the stem ndarrays of the previous song). Within a song, best_state only
        # changes on acceptance, so the per-stem renders of the non-accepted steps
        # disappear entirely. Bit-identical.
        # With --render-cache, the render of the solo stem is made incremental too
        # (removing the O(K^2) residue of re-applying the full chain of one stem on every
        # accepted step). The output is bit-identical. By default (without the flag) it
        # stays uncached as before.
        lufs_cache = StemLufsCache(
            render_cache=bool(getattr(args, "render_cache", False)))
        if args.proposer == "llm":
            args.search_proposer = make_llm_search_proposer(
                llm, tool_specs, args.llm_max_invalid_retries, stats,
                deesser_min_sibilance=args.deesser_min_sibilance,
                show_budget=args.prompt_show_budget,
                stem_lufs_cache=lufs_cache,
                anti_repeat=bool(ar_cfg),
                stall_window=int(args.anti_repeat_stall_window),
                forbid_n=int(args.anti_repeat_forbid_n),
                rejected_distinct_n=int(args.anti_repeat_rejected_n),
                guidance=args.guidance,
                reward_form=args.reward_form,
                excerpt_sec=getattr(args, "search_excerpt_sec", None),
                fewshot=bool(getattr(args, "prompt_fewshot", False)),
                numfmt=getattr(args, "prompt_numfmt", None),
                history=hist_cfg,
                system_mode=getattr(args, "prompt_system", None),
                drop_guidance=bool(getattr(args, "drop_guidance", False)),
                action_set=aset,
                # The beat grid is put on args by _run_one_song_random (after resolving
                # the window for each song). Here we only pass a lazy reference.
                # Without --excerpt-beats it stays None, so the prompt is unchanged.
                beat_grid_fn=((lambda: (getattr(args, "excerpt_beat_times_rel", None),
                                        getattr(args, "excerpt_tempo_bpm", None)))
                              if bool(getattr(args, "excerpt_beats", False))
                              else None))
        elif args.proposer == "hybrid":
            # CRN: the random tail receives the rng of _propose_pipeline (the same
            # sequence as pure-random) through rand_proposal. The proposer holds no
            # independent rng of its own.
            args.search_proposer = make_hybrid_search_proposer(
                llm, tool_specs, args.llm_max_invalid_retries, stats,
                switch_step=args.hybrid_switch_step,
                stall_n=int(getattr(args, "hybrid_stall_n", 0) or 0),
                deesser_min_sibilance=args.deesser_min_sibilance,
                show_budget=args.prompt_show_budget,
                stem_lufs_cache=lufs_cache)
        elif args.proposer == "informed_random":
            # Static informed-random. A proposer with its own dedicated stream,
            # independent of the pipeline rng, is created per song (seeded
            # deterministically from song_index). The pipeline keeps drawing a uniform
            # proposal at every step as before (preserving CRN), so the calibration pool
            # matches the uniform run at the same seed exactly, and the informed proposal
            # takes the form of "discarding and replacing" the uniform proposal.
            args.search_proposer = make_informed_random_proposer(
                informed_stats, eff_base_seed, si)
        try:
            res = ALL._run_one_song_random(row, si, run_dir, args, scorers)
            if args.proposer in ("llm", "hybrid"):
                res["llm_stats"] = _stats_summary(
                    stats, args.backend, llm_model,
                    show_budget=args.prompt_show_budget,
                    stem_lufs_cache=lufs_cache, llm=llm,
                    anti_repeat=ar_cfg,
                    guidance=args.guidance, reward_form=args.reward_form,
                    prompt_fewshot=bool(getattr(args, "prompt_fewshot", False)),
                    prompt_numfmt=getattr(args, "prompt_numfmt", None),
                    prompt_history=hist_cfg,
                    prompt_system=getattr(args, "prompt_system", None),
                    prompt_drop_guidance=bool(getattr(args, "drop_guidance", False)),
                    action_set=aset,
                    allowed_action_names=[s["name"] for s in tool_specs],
                    beat_grid=({"n_beats": len(getattr(args, "excerpt_beat_times_rel", None) or []),
                                "tempo_bpm": getattr(args, "excerpt_tempo_bpm", None),
                                "times_rel_sec": getattr(args, "excerpt_beat_times_rel", None)}
                               if bool(getattr(args, "excerpt_beats", False)) else None))
            res["proposer"] = args.proposer
            if args.proposer == "hybrid":
                res["hybrid_switch_step"] = args.hybrid_switch_step
                res["hybrid_stall_n"] = int(getattr(args, "hybrid_stall_n", 0) or 0)
            if informed_prov is not None:
                # Write the key only under informed_random (the same "do not write the
                # key at all outside the relevant mode" policy as SCORING_PROVENANCE;
                # the JSON of a default run does not change by a single byte).
                res["proposer_provenance"] = dict(informed_prov)
        except Exception as ex:                                 # noqa: BLE001
            res = {"status": "error", "song_id": row["song_id"],
                   "split": row["split"], "error": repr(ex)}
            if args.proposer in ("llm", "hybrid"):
                res["llm_stats"] = _stats_summary(
                    stats, args.backend, llm_model,
                    show_budget=args.prompt_show_budget,
                    stem_lufs_cache=lufs_cache, llm=llm,
                    anti_repeat=ar_cfg,
                    guidance=args.guidance, reward_form=args.reward_form,
                    prompt_fewshot=bool(getattr(args, "prompt_fewshot", False)),
                    prompt_numfmt=getattr(args, "prompt_numfmt", None),
                    prompt_history=hist_cfg,
                    prompt_system=getattr(args, "prompt_system", None),
                    prompt_drop_guidance=bool(getattr(args, "drop_guidance", False)),
                    action_set=aset,
                    allowed_action_names=[s["name"] for s in tool_specs],
                    beat_grid=({"n_beats": len(getattr(args, "excerpt_beat_times_rel", None) or []),
                                "tempo_bpm": getattr(args, "excerpt_tempo_bpm", None),
                                "times_rel_sec": getattr(args, "excerpt_beat_times_rel", None)}
                               if bool(getattr(args, "excerpt_beats", False)) else None))
            print(f"[sweep] ERROR {row['song_id']}: {ex!r}", flush=True)
        finally:
            # Do not leave search_proposer on args (cleared explicitly since it is reset
            # for the next song).
            args.search_proposer = None
        ALL._write_json_atomic(ALL.song_result_path(run_dir, row["song_id"]), res)
        done_rows.append({"song_id": row["song_id"], "split": row["split"],
                          "status": res["status"]})
        ALL._write_json_atomic(
            index_path, {"run_name": args.run_name, "splits": splits,
                         "rows": done_rows, "random_mode": True,
                         "backend": args.backend, "llm_model": llm_model,
                         "prompt_show_budget": bool(args.prompt_show_budget),
                         # Write the key only under informed_random (the default is
                         # unchanged). Resume works per song, so the per-song record is
                         # the authoritative one.
                         **({"proposer_stats_path": str(args.proposer_stats),
                             "proposer_stats_sha256": informed_sha}
                            if informed_sha else {})})
    print(f"[sweep] done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
