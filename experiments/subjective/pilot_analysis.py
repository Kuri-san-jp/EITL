#!/usr/bin/env python3
"""pilot_analysis.py -- analysis pipeline for the Prolific pilot listening test (CPU only).

master plan: internal notes §6 (session structure) / §7 (exclusion and analysis).
Real-data analyser that replaces the existing analyze_mushra.py (hidden-reference
design, retired) and prolific_power_sim.py / master_power.py (power). There is no
hidden reference here.

Input (the JSON records produced by the SPA's incremental POSTs)
----------------------------------------------------------------
1 screen = 1 record. --records accepts any of:
  * a JSONL file (one record per line)
  * a JSON array file / a {"records": [...]} wrapper
  * a directory holding record files (reads every *.json / *.jsonl in it)

Default schema (every field name can be swapped out via the --config JSON. If the
schema does not line up with what the SPA side sends, only the field map needs
fixing. --dump-config prints a template):

  {"type":"stage1",  "pid":"...", "hp_correct":6, "ap_correct":5}
  {"type":"mushra",  "pid":"...", "screen_index":3, "song":"...",
   "ratings":{"proposed":78,...}, "duration_sec":112.4, "is_training":false}
  {"type":"afc",     "pid":"...", "pair_index":5, "song":"...",
   "choice":"proposed", "is_repeat":true, "repeat_of":1, "duration_sec":24.0}
  {"type":"abx",     "pid":"...", "trial_index":2, "correct":true}
  {"type":"session", "pid":"...", "completed":true, "total_sec":1712.0,
   "order_group":"afc_first"}

A field map value may be a dot-separated nested path such as "payload.ratings".
The real schema of pilot_v2/app (logic.js buildRecord) -- envelope {type,pid,stage,
demo,payload:{...}}, type names mushra_trial/afc_trial/abx_trial/hpscreen/session_end,
ratings keys = "audio/<hmac16>.opus", 2AFC choice = a blind choice_file,
completion = payload.code_logical -- is readable with **--preset spa_v1**
(= SPA_V1_OVERRIDES; a copy lives in pilot_analysis_config.spa_v1.json).

If the ratings keys are blind ids / blind file paths, de-blind them with --manifest.
--manifest accepts both pilot_v2/stimuli/manifest.json (unreleased, trials[].clips
format) and the flat format (blind_id -> {song, condition}). The same manifest is
also used to resolve the 2AFC choice_file and trial_id -> song.
For a resend with the same (pid, type, index), the later record wins (retry support).
Only session records are kept per stage, and completion in any stage counts as
completed (so that Stage1's session_end does not overwrite Stage2's completion record).

Exclusion pipeline (the 6 criteria of master plan §7.2, applied in this order)
------------------------------------------------------------------------------
  1. Stage1 HP+AP not passed (both HP >=5/6 and AP >=5/6 are required)
  2. Stage2 incomplete (session.completed / screen-count fallback)
  3. Basic check failed on more than 20% of the MUSHRA screens:
     (i) random_dry > median of the other conditions on the same screen,
     (ii) all conditions other than the anchor tied
  4. Total time < 0.5 x the median of the survivors (speeder)
  5. 2AFC agreement over the 4 repeated pairs < 25%
  6. IQR rule (listener mean score outside the Tukey fences)

Analysis
--------
MUSHRA (§7.3):  score ~ condition + (1|listener) + (1|song)
                + (1|listener:condition) + (1|song:condition) + presentation_order
is solved by reducing it, per contrast, to the within-screen difference
d_ls = score_A - score_B. Since the difference is taken within one screen, the
listener / song main effects and the presentation_order main effect cancel exactly,
and the remaining model is

    d_ls = mu + u_l + v_s + e_ls,
    u_l ~ N(0, tau_LC^2), v_s ~ N(0, tau_SC^2), e_ls ~ N(0, sigma_d^2)

(tau_LC, tau_SC, sigma_d follow the same "difference scale" definition as
master_power.py. The lme4-style per-condition variances are 1/sqrt(2) times these.)
This is solved with an in-house dense REML, and the t test on mu_hat +- SE with
df = min(L,S)-1 is the primary one. The statsmodels MixedLM variance-components
specification (crossed) is kept behind --full-lmm as a cross-check (statsmodels is
weak on crossed designs and unstable with many VCs, so it is not run by default).

H1 (proposed vs professional) = a **one-sided non-inferiority test on the ratings**
(design A; the authoritative text is pilot_v2/internal notes §3 and §5.4b). On the
within-screen difference d = score_proposed - score_professional,

    H0: mu <= -Delta    vs    H1: mu > -Delta        (Delta > 0 = non-inferiority margin)

as a one-sided t test (df = min(L,S)-1). **This is not an equivalence test (TOST).**
The claim is not "no different from the professional" but "not worse than the
professional by Delta or more".
**Delta must not be chosen from power**, so the implementation fixes the order:
decide Delta from external grounds first (config analysis.noninf_margin_points and
its _source), and only afterwards let noninferiority_report report the achievable
power.

2AFC / ABX are **not run** under design A. The implementations are kept
(afc_analysis / abx_analysis); with 0 records they simply return "not applicable"
and do not enter the primary family. Restoring the blocks in app/config.js brings
them straight back. The old H1 (2AFC binomial GLMM  choice ~ 1 + (1|listener)
+ (1|song) (statsmodels BinomialBayesMixedGLM, VB approximation) + a one-sided
Wilcoxon on the per-listener preference rate + TOST (equivalent if the 95% CI of the
preference rate ⊂ [0.44, 0.56]), with the 4 repeated pairs used only as a
reliability check and only the first presentations used for estimation) survives
intact inside afc_analysis, and is reported as an **auxiliary** result whenever
2AFC data exist (the primary family stays at 3 tests).

Main purpose of the pilot: measure tau_LC / tau_SC / sigma_d and plug them into the
master_power.py MDE formula
    Var(mean d) = tau_LC^2/L + tau_SC^2/S + sigma_d^2/(L*S_per)
to recompute L for the main study (recompute_design). The ABX block demotes H2 out
of the primary family when the group mean accuracy is <= 0.60 (master plan §3).

Usage
-----
  python3 experiments/subjective/pilot_analysis.py --records <path> [--manifest m.json]
  python3 experiments/subjective/pilot_analysis.py --preset spa_v1 \
      --records <SPA POST logs> --manifest experiments/subjective/pilot_v2/stimuli/manifest.json
  python3 experiments/subjective/pilot_analysis.py --dump-config [--preset spa_v1]
  python3 experiments/subjective/pilot_analysis.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------- #
# Configuration (every field name can be swapped out here)
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    # Record field names (override with --config to match the SPA-side schema)
    "fields": {
        "type": "type",
        "pid": "pid",
        "stage": "stage",         # used for session dedupe (may be absent)
        "demo": "demo",           # if truthy, drop the whole record (SPA demo/preview)
        "song": "song",
        "ratings": "ratings",
        "screen_index": "screen_index",
        "duration": "duration_sec",
        "is_training": "is_training",
        "pair_index": "pair_index",
        "choice": "choice",
        "is_repeat": "is_repeat",
        "repeat_of": "repeat_of",
        "trial_index": "trial_index",
        "correct": "correct",
        "response": "response",
        "truth": "truth",
        "hp_correct": "hp_correct",
        "ap_correct": "ap_correct",
        "completed": "completed",
        "total_sec": "total_sec",
        "order_group": "order_group",
    },
    # record type values (edit here if the SPA uses different strings)
    "type_values": {
        "stage1": "stage1",
        "mushra": "mushra",
        "afc": "afc",
        "abx": "abx",
        "session": "session",
        "questionnaire": "questionnaire",
        # Single-study design : the record of the screening
        # questionnaire itself, and the record of the eligibility verdict the
        # server issued on the spot.
        #   screen_q : the participant's answers (q_device / q_hearing / q_prod / q_know*)
        #   elig     : the /api/elig_score response (eligible / provisional / note)
        # If the SPA cannot reach the endpoint it lets the participant through on a
        # provisional pass, so **the analysis must always re-adjudicate**
        # (step 1a of exclusion_pipeline).
        "screen_q": "screen_questionnaire",
        "elig": "elig_score",
    },
    # Condition-name normalisation (SPA-side raw value -> canonical). Canonical
    # names come from master plan §4.
    "condition_map": {
        "professional": "professional",
        "proposed": "proposed",
        "pq_only": "pq_only",
        "mean_reward": "mean_reward",
        "fxnorm": "fxnorm",
        "random_dry": "random_dry",
    },
    # : 6 conditions -> 8 conditions (sb_only and megami added; master plan §15)
    "conditions": ["professional", "proposed", "pq_only", "sb_only",
                   "mean_reward", "fxnorm", "megami", "random_dry"],
    "anchor_condition": "random_dry",
    # 2AFC choice value -> canonical condition (map here if the SPA sends "A"/"B")
    "choice_map": {"proposed": "proposed", "professional": "professional"},
    "afc_pair": ["proposed", "professional"],   # [condition of interest, comparison condition]
    # Exclusion criteria (in the order of master plan §7.2)
    "screening": {
        "hp_min": 5, "ap_min": 5,               # Stage1: HP >=5/6 and AP >=5/6
        "require_stage1": False,                # treat a missing stage1 record as a fail?
        # Fallback when the completed flag is missing. Design A assigns 9 songs per
        # participant, so 8 is a deliberately loose threshold that "tolerates one
        # missing screen POST". The backend does not issue a completion code unless
        # all 9 screens are present (backend_app._expected_counts), so even without
        # setting this to 9 there is a separate gate on "were 9 screens presented?".
        # This intent is also written down in internal notes §5.4b.
        "required_mushra_screens": 8,
        # Required number of 2AFC trials. **Design A does not present 2AFC.**
        # Whether it is required at all is decided by afc_block_expected (below).
        "required_afc_trials": 12,
        # Who the value above is required of. None = auto / True = everyone /
        # False = nobody. Auto means: if the share of participants that have 2AFC
        # rows is >= afc_block_min_share, assume the block was presented and require
        # it of everyone; otherwise require it only of the participants that do have
        # afc rows. The old implementation simply looked at "is there even one afc
        # row in the data", so a single row from an older end-to-end run that
        # included 2AFC (TESTRUN and the like) was enough to drag design A finishers
        # into the exclusion (fixed in the  integration check).
        "afc_block_expected": None,
        "afc_block_min_share": 0.5,
        "screen_fail_frac": 0.20,               # exclude above 20% failed-check screens
        "flat_tol": 1e-9,                       # tolerance for calling scores "tied"
        "speeder_factor": 0.5,                  # exclude below 0.5 x the median
        "repeat_agree_min": 0.25,               # exclude if repeat agreement < 25%
        "iqr_k": 1.5,
        # Values that count as "completed" when the completed field is a string
        # (completion code). None = interpret as bool (for the boolean schema).
        "completed_values": None,
        # Re-adjudication of the screening questionnaire at analysis time
        # (single-study design, .
        #   recheck_eligibility        : re-adjudicate with screen_eligibility.py and exclude
        #   require_eligibility_record : also exclude people with no adjudicable
        #        questionnaire record (default False. Set True if you want to drop
        #        participants who suppressed only the questionnaire POST by tampering
        #        with the network. It can also drop honest participants by mistake,
        #        so look at the audit table (eligibility_audit) before deciding.)
        "recheck_eligibility": True,
        "require_eligibility_record": False,
    },
    # Test settings (master plan §3 / §7.3 / §7.4)
    "analysis": {
        "alpha_family": 0.05,
        "n_primary": 3,                          # H1, H2, H3 (Holm)
        # ------------------------------------------------------------------ #
        # Non-inferiority margin Delta for H1 (0-100 points). The core of design A.
        # Authoritative text: pilot_v2/internal notes §3.
        #
        # **Delta must not be chosen from power.** The procedure is fixed in this order:
        #   1. Decide Delta from external grounds (below)
        #   2. Report the power achievable for that Delta
        #      (noninferiority_report computes and prints it)
        #   3. "Power is insufficient, so widen Delta" is forbidden. A back-solved
        #      Delta is post hoc dressed up as pre-registration, and reviewers will
        #      always catch it.
        #
        # Grounds for the default of 10.0 points (= external grounds, not back-solved
        # from power):
        #   (i) The 0-100 continuous scale of ITU-R BS.1534-3 allots 20 points to
        #       each step of the 5-step verbal scale
        #       (bad / poor / fair / good / excellent).
        #       10 points = half of one verbal step. A practical line: a difference
        #       larger than this can change the verbal grade = a difference that
        #       matters in production.
        #   (ii) The a priori effect-size scenarios (the H1_topline row of internal
        #       notes §2.1) are professional - proposed = +2 / +6 / +12 points.
        #       **The numbers are taken as-is from the source, but the labels are
        #       read the other way round.** The source is written in a "detection"
        #       framing, where +2 is pessimistic (a small difference, hard to detect)
        #       and +12 optimistic. Under a non-inferiority framing the favourable
        #       and unfavourable directions flip (the larger the difference, the
        #       harder it is to claim non-inferiority), so here +12 is the
        #       unfavourable side and +2 the favourable one. This re-reading is
        #       spelled out so that anyone checking against the source does not see
        #       it as a misquote (raised in the  integration check).
        #       Delta = 10 is wider than the central +6 and narrower than the most
        #       unfavourable +12, i.e. it is set so that "if a difference of a full
        #       +12 points really exists, non-inferiority cannot be claimed".
        #   (iii) Zieliński et al. (AES) report a range-equalizing bias of up to 22%,
        #       so widening Delta beyond that makes it indistinguishable from
        #       measurement bias. Hence Delta <= 20 points.
        #
        # **Be honest about a Delta that cannot be achieved.** With a target of 28
        # analysed participants and 9 songs each, the minimum detectable effect is
        # H1 = 6.27 points (design_tradeoff.py --plan, power 80%, one-sided
        # alpha=0.05/3 after Holm).
        # **A Delta smaller than 6.27 points is not achievable at this scale.**
        # If a smaller Delta is chosen deliberately, state explicitly that "the power
        # for that Delta does not reach 80%" and give the achieved value
        # (noninferiority_report does this automatically).
        # Do not re-derive Delta to fit 6.27 (that is back-solving after the fact).
        "noninf_margin_points": 10.0,
        "noninf_margin_source": (
            "Half of one verbal-scale step (20 points) of ITU-R BS.1534-3. "
            "Between the central +6 and the most unfavourable +12 of the a priori "
            "effect-size scenarios (internal notes §2.1: "
            "professional - proposed = +2 / +6 / +12 points; the numbers are taken "
            "as-is from the source, and since its pessimistic/optimistic labels "
            "belong to a detection framing they are read in reverse under "
            "non-inferiority). "
            "Not back-solved from power (design_a_spec §3)"),
        # Minimum detectable effect at planning time (the 28-participant row of
        # design_tradeoff.py --plan). A **reporting-only** constant meaning "if Delta
        # falls below this, power does not reach 80%".
        # Do not rewrite it to justify a Delta.
        "noninf_planned_mde_points": 6.27,
        "noninf_planned_L": 28,                  # target analysed N (design_a_spec §4)
        # "True difference" scenarios used when reporting power (points; positive =
        # proposed is worse). 0 = truly equivalent, 6.0 = the a priori central scenario.
        "noninf_report_gaps": [0.0, 6.0],
        # ------------------------------------------------------------------ #
        # Equivalence-test settings for the old H1 (2AFC). Design A does not run
        # 2AFC, so the default contrasts contain no aux_tost and these two are unused.
        # **Do not delete them**: adding a contrast with family="aux_tost" in the
        # config brings them back, and afc_analysis will emit the TOST as-is when
        # 2AFC data exist.
        "tost_margin_points": 5.0,               # equivalence margin in MUSHRA points
        "tost_pref_lo": 0.44, "tost_pref_hi": 0.56,   # equivalence band for the preference rate
        "abx_demote_threshold": 0.60,            # demote H2 when group mean accuracy <= 0.60
        "power_target": 0.80,
        # MUSHRA songs per participant. Design A uses **9 songs** (a cyclic block of
        # 22 songs over 44 slots; fixed on  by tools/reassign_songs.py).
        "S_per": 9,
        # Song pool of the main study (total number of songs). The design A
        # assignment is 22 songs x 44 slots, giving 9 songs per participant and
        # exactly 18 slots per song (tools/reassign_songs.py).
        # The old comment "rotation 8-7-7" described the previous assignment of 7-8
        # songs per participant, so it was removed on  (S_per=9 is correct).
        "S_main": 22,
        "mde_targets": [4.0, 5.0, 6.0],          # MDE targets to recompute (points)
        "L_grid": [12, 24, 32, 48, 60, 80],
    },
    # Definition of the MUSHRA contrasts. family: primary / primary_noninf /
    #                            secondary / check
    #                            (aux_tost is unused under design A; see the note above)
    # side: greater = one-sided a>b, noninf = one-sided a >= b - Delta, two = two-sided
    "contrasts": [
        # H1 comes from a **one-sided non-inferiority test on the ratings**, not from
        # the 2AFC win rate (design A §3). Since 2AFC is no longer presented, this is
        # the only primary test for H1.
        # [Fixed , before data collection of the main experiment] The
        # proposed method is **min**.
        # Background: it was switched to mean once on 08-26, but in the pilot (n=6)
        # min scored 46.2 > mean 43.9 (professional 51.9), i.e. against the objective
        # trade-off (min leans towards PQ / mean towards SB) perception leaned towards
        # min. Reverting to min.
        # The stimuli do not change at all: the 8 conditions of the main study already
        # contain both min (called "proposed" in the manifest) and mean
        # ("mean_reward"); this only specifies which of them is the primary arm.
        # mean is kept as the secondary comparison (S1).
        {"name": "H1_topline_noninf", "a": "proposed", "b": "professional",
         "side": "noninf", "family": "primary_noninf"},
        {"name": "H2_hack", "a": "proposed", "b": "pq_only",
         "side": "greater", "family": "primary"},
        {"name": "H3_sota", "a": "proposed", "b": "fxnorm",
         "side": "greater", "family": "primary"},
        {"name": "S1_minvsmean", "a": "proposed", "b": "mean_reward",
         "side": "two", "family": "secondary"},
        {"name": "check_dry", "a": "proposed", "b": "random_dry",
         "side": "none", "family": "check"},
    ],
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def canon(s: str) -> str:
    """Normalisation for song_id matching (master plan §1.4)."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# Preset for the real schema of pilot_v2/app (logic.js buildRecord + app.js log()).
# envelope = {v, set, app, type, seq, stage, pid, ..., demo, payload:{...}}.
# The ratings keys and the 2AFC choice_file are blind file paths
# ("audio/<hmac16>.opus"), so --manifest (pilot_v2/stimuli/manifest.json) is required.
SPA_V1_OVERRIDES: Dict[str, Any] = {
    "fields": {
        "type": "type",
        "pid": "pid",
        "stage": "stage",
        "demo": "demo",
        "song": "payload.trial_id",             # trial_id -> song resolved via the manifest
        "ratings": "payload.ratings",
        "screen_index": "payload.trial_index",
        "duration": "payload.duration_sec",
        "is_training": "payload.is_training",
        "pair_index": "payload.afc_index",
        "choice": "payload.choice_file",        # blind path -> condition resolved via the manifest
        "is_repeat": "payload.is_repeat",
        "repeat_of": "payload.repeat_of",       # the SPA does not send it -> match by song
        "trial_index": "payload.abx_index",
        "correct": "payload.correct",
        "response": "payload.response",
        "truth": "payload.truth",
        "hp_correct": "payload.score.hp_correct",
        "ap_correct": "payload.score.ap_correct",
        "completed": "payload.code_logical",
        "total_sec": "payload.total_sec",
        "order_group": "afc_first",             # a bool in the envelope
    },
    "type_values": {
        "stage1": "hpscreen",
        "mushra": "mushra_trial",
        "afc": "afc_trial",
        "abx": "abx_trial",
        "session": "session_end",
        "questionnaire": "post_questionnaire",
        # The SPA sends the **pre-**screening questionnaire as type="questionnaire"
        # and the post-session survey as type="post_questionnaire" (see log() in
        # app.js). Years of experience, number of finished tracks and the knowledge
        # questions all live only in the former.
        "screen_q": "questionnaire",
        "elig": "elig_score",
    },
    # MAIN_FAILCHECK means "finished but failed the inline check" = counts as Stage2
    # completed. Any exclusion beyond that is decided (by recomputation) by criteria 3-6.
    "screening": {"completed_values": ["MAIN_COMPLETE", "MAIN_FAILCHECK"]},
}
PRESETS: Dict[str, Dict[str, Any]] = {"spa_v1": SPA_V1_OVERRIDES}


def _get(rec: Any, path: Optional[str], default: Any = None) -> Any:
    """Read a field from a dict. Supports "a.b.c" nested paths (flat keys win)."""
    if path is None or not isinstance(rec, dict):
        return default
    if path in rec:
        return rec[path]
    cur: Any = rec
    for part in str(path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# Question IDs of the screening questionnaire (selectQ / numberQ of
# SCREENS.questionnaire in app.js). Used only to decide "does this record contain
# enough to adjudicate eligibility". **The correct answers and thresholds are not
# here** (the authoritative copy is pilot_v2/tools/screen_eligibility.py, which the
# analysis imports and uses).
# Updated for the  questionnaire reduction (12 questions -> 8). q_device
# was promoted to an eligibility criterion, and q_mixes / q_recent / q_instr /
# q_paid were dropped as questions entirely.
SCREEN_QIDS = ("q_device", "q_hearing", "q_prod", "q_prod_years",
               "q_know1", "q_know2", "q_know3")


_ELIG_MOD: Any = None
_ELIG_ERR: Optional[str] = None


def elig_rules() -> Any:
    """Load pilot_v2/tools/screen_eligibility.py, the authoritative eligibility rules
    (returns None if it is not there).

    Do not reimplement the rules in this file. Having all three places -- SPA,
    backend and analysis -- use the same module is what keeps "the criterion applied
    in production" and "the criterion re-adjudicated in the analysis" from drifting
    apart.
    """
    global _ELIG_MOD, _ELIG_ERR
    if _ELIG_MOD is not None or _ELIG_ERR is not None:
        return _ELIG_MOD
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "pilot_v2", "tools")
    try:
        if d not in sys.path:
            sys.path.insert(0, d)
        import screen_eligibility as _m       # type: ignore  # noqa: PLC0415
        if not hasattr(_m, "classify_app_questionnaire"):
            raise ImportError("classify_app_questionnaire is missing (old version)")
        _ELIG_MOD = _m
    except Exception as e:                                     # noqa: BLE001
        _ELIG_ERR = str(e)
        print(f"[warn] cannot read screen_eligibility.py ({e}). "
              f"Skipping re-adjudication of the screening questionnaire", file=sys.stderr)
    return _ELIG_MOD


def _screen_summary(ans: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one questionnaire to flat reporting fields + the re-adjudication result.

    Only screen_eligibility.py holds the correct answers to the knowledge questions
    and the eligibility criteria. If it cannot be read, set eligible=None
    (unadjudicable) and surface the case in the audit table rather than excluding it.
    """
    m = elig_rules()
    out: Dict[str, Any] = {
        "hearing": ans.get("q_hearing"),
        # Playback device. Promoted to an eligibility criterion on .
        # Eligible participants use one of over_ear / in_ear / tws, and **the size of
        # the tws group must be reported as a covariate** (on-device signal
        # processing can affect the ratings).
        "device": ans.get("q_device"),
        "exp_band": ans.get("q_prod"),
        "exp_years": _float_or_nan(ans.get("q_prod_years")),
        # Flattening of the questions dropped on  (q_mixes / q_paid /
        # q_recent) was removed. The questions no longer exist, so keeping the
        # columns would make them None for everyone and they would show up in the
        # Methods table as "not answered".
        "prev_tests": ans.get("q_prev"),
        "adjudicable": bool(ans) and any(q in ans for q in SCREEN_QIDS),
        "eligible": None, "reasons": [], "n_know_ok": float("nan"),
        "know_pass": None, "know_bogus": None, "criteria_version": None,
    }
    if m is None or not out["adjudicable"]:
        return out
    try:
        v = m.classify_app_questionnaire(ans)
    except Exception as e:                                     # noqa: BLE001
        out["reasons"] = [f"classify_error:{e}"]
        return out
    reasons = [str(x) for x in (v.get("reasons") or [])]
    out.update({"eligible": bool(v.get("eligible")), "reasons": reasons,
                "n_know_ok": _float_or_nan(v.get("n_know_correct")),
                "know_pass": "fail_knowledge" not in reasons,
                "know_bogus": "fail_bogus" in reasons,
                "criteria_version": v.get("criteria_version")})
    return out


def _screen_answers(rec: Any) -> Dict[str, Any]:
    """Extract the questionnaire answer dict from a record; {} if there is none.

    The SPA calls log("questionnaire", ans), so envelope.payload is the answer dict
    itself. In the flat schema (via the old Google Forms route) it sits either
    directly on the record or under record["answers"].
    To pick it up wherever it lives, walk the candidates in order and take **the one
    that contains question IDs**.
    """
    for cand in (_get(rec, "payload.answers"), _get(rec, "answers"),
                 _get(rec, "payload"), rec):
        if isinstance(cand, dict) and any(q in cand for q in SCREEN_QIDS):
            return cand
    return {}


_STEM_RE = re.compile(r"([^/\\]+?)(\.[A-Za-z0-9]+)?$")


def _blind_lookup(blind: Dict[str, Tuple[str, str]], key: str
                  ) -> Optional[Tuple[str, str]]:
    """Look up the blind map: exact match first, then the extension-stripped stem of
    the path tail (equivalent to clipKey in logic.js)."""
    hit = blind.get(key)
    if hit is not None:
        return hit
    m = _STEM_RE.search(str(key))
    return blind.get(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _parse_text(text: str) -> List[dict]:
    text = text.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            if isinstance(obj.get("records"), list):
                return [r for r in obj["records"] if isinstance(r, dict)]
            return [obj]
    except json.JSONDecodeError:
        pass
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def read_raw_records(path: str) -> List[dict]:
    paths: List[str]
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.json"))
                       + glob.glob(os.path.join(path, "*.jsonl")))
    else:
        paths = [path]
    recs: List[dict] = []
    for p in paths:
        with open(p) as fh:
            recs.extend(_parse_text(fh.read()))
    return recs


def load_blind_map(path: Optional[str]
                   ) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    """manifest -> (blind_id/file -> (song, condition), trial_id -> song_id).

    Supported formats:
      * pilot_v2/stimuli/manifest.json (build_stimuli_v2, unreleased):
        {"trials":[{"trial_id","song_id","clips":{cond:{"file","blind_id",...}}}],
         "practice": {...}}
      * old build_stimuli format: {"songs":[{"song_id","files":{cond:{"blind_id"}}}]}
      * flat: {bid: {"song","condition"}} / {bid: [song, condition]}
    On the blind side, register both the blind_id and the file path as keys
    (the SPA's ratings keys / choice_file are "audio/<bid>.opus").
    """
    if not path:
        return {}, {}
    man = json.load(open(path))
    out: Dict[str, Tuple[str, str]] = {}
    t2s: Dict[str, str] = {}

    def reg(key: Any, song: str, cond: str) -> None:
        if key:
            out[str(key)] = (song, cond)

    if isinstance(man, dict) and isinstance(man.get("trials"), list):
        trs = list(man["trials"])
        if isinstance(man.get("practice"), dict):
            trs.append(man["practice"])
        for t in trs:
            song = str(t.get("song_id", ""))
            if t.get("trial_id"):
                t2s[str(t["trial_id"])] = song
            for cond, c in (t.get("clips") or {}).items():
                if isinstance(c, dict):
                    reg(c.get("blind_id"), song, cond)
                    reg(c.get("file"), song, cond)
                else:
                    reg(c, song, cond)
    elif isinstance(man, dict) and "songs" in man:          # old build_stimuli format
        for m in man["songs"]:
            for lab, f in (m.get("files") or {}).items():
                bid = f["blind_id"] if isinstance(f, dict) else f
                reg(bid, m["song_id"], lab)
                if isinstance(f, dict):
                    reg(f.get("file"), m["song_id"], lab)
    elif isinstance(man, dict):                              # flat: bid -> {...}
        src = man.get("blind_map", man)
        for bid, v in src.items():
            if isinstance(v, dict) and "condition" in v:
                reg(bid, v.get("song", ""), v["condition"])
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                reg(bid, str(v[0]), str(v[1]))
    return out, t2s


def normalise(raw: List[dict], cfg: dict,
              blind: Dict[str, Tuple[str, str]],
              trial2song: Optional[Dict[str, str]] = None
              ) -> Dict[str, pd.DataFrame]:
    """Turn the raw records into per-type DataFrames. For resends (same
    pid/type/index) the last one wins.

    Field map values may be dot-separated nested paths (_get). When the song field
    holds a trial_id, resolve it with trial2song (derived from the manifest). Only
    session records are kept per stage (so that Stage1's session_end does not
    clobber Stage2's completion record).
    """
    trial2song = trial2song or {}
    F = cfg["fields"]
    TV = {v: k for k, v in cfg["type_values"].items()}      # raw type -> canonical
    cmap = cfg["condition_map"]
    chmap = cfg["choice_map"]
    completed_vals = cfg["screening"].get("completed_values")

    def resolve_song(v: Any) -> str:
        s = str(v or "")
        return trial2song.get(s, s)

    dedup: Dict[Tuple, dict] = {}
    for r in raw:
        t = TV.get(str(_get(r, F["type"], "")))
        if t is None:
            continue
        if bool(_get(r, F.get("demo", "demo"), False)):
            continue                                       # drop demo / preview
        pid = str(_get(r, F["pid"], "") or "") or None
        if not pid:
            continue
        if t == "mushra":
            key = (pid, t, _get(r, F["screen_index"]), _get(r, F["song"]))
        elif t == "afc":
            key = (pid, t, _get(r, F["pair_index"]))
        elif t == "abx":
            key = (pid, t, _get(r, F["trial_index"]))
        elif t == "session":
            key = (pid, t, _get(r, F.get("stage", "stage")))
        else:
            key = (pid, t)
        dedup[key] = r
    n_dupe = len(raw) - len(dedup)

    mushra, afc, abx, stage1, sess, quest = [], [], [], [], [], []
    screenq: List[dict] = []
    eligrec: List[dict] = []
    for (pid, t, *_), r in dedup.items():
        if t == "mushra":
            if bool(_get(r, F["is_training"], False)):
                continue
            ratings = _get(r, F["ratings"]) or {}
            song = resolve_song(_get(r, F["song"], ""))
            for key, val in ratings.items():
                key = str(key)
                hit = _blind_lookup(blind, key)
                if hit is not None:
                    b_song, cond_raw = hit
                    song_eff = b_song or song
                else:
                    cond_raw, song_eff = key, song
                cond = cmap.get(cond_raw)
                if cond is None:
                    continue
                try:
                    score = float(val)
                except (TypeError, ValueError):
                    continue
                mushra.append({
                    "listener": pid, "song": canon(song_eff), "condition": cond,
                    "score": score,
                    "screen": _get(r, F["screen_index"]),
                    "dur": float(_get(r, F["duration"]) or np.nan),
                })
        elif t == "afc":
            song_eff = resolve_song(_get(r, F["song"], ""))
            ch_raw = _get(r, F["choice"])
            ch = None
            if ch_raw is not None:
                ch = chmap.get(str(ch_raw))
                if ch is None:                     # resolve a blind choice_file
                    hit = _blind_lookup(blind, str(ch_raw))
                    if hit is not None:
                        b_song, cond_raw = hit
                        ch = chmap.get(cond_raw, cmap.get(cond_raw))
                        if not song_eff:
                            song_eff = b_song
            rep_of = _get(r, F["repeat_of"])
            afc.append({
                "listener": pid, "song": canon(song_eff),
                "pair_index": _get(r, F["pair_index"]),
                "choice": ch,
                "is_repeat": bool(_get(r, F["is_repeat"], rep_of is not None)),
                "repeat_of": rep_of,
                "dur": float(_get(r, F["duration"]) or np.nan),
            })
        elif t == "abx":
            corr = _get(r, F["correct"])
            if corr is None:
                resp, truth = _get(r, F["response"]), _get(r, F["truth"])
                if resp is not None and truth is not None:
                    corr = resp == truth
            if corr is None:
                continue
            abx.append({"listener": pid, "trial": _get(r, F["trial_index"]),
                        "correct": bool(corr)})
        elif t == "stage1":
            stage1.append({"listener": pid,
                           "hp": _int_or_nan(_get(r, F["hp_correct"])),
                           "ap": _int_or_nan(_get(r, F["ap_correct"]))})
        elif t == "session":
            comp_raw = _get(r, F["completed"])
            if comp_raw is None:
                completed = None
            elif completed_vals is not None:
                completed = str(comp_raw) in completed_vals
            elif isinstance(comp_raw, str):
                completed = comp_raw.strip().lower() in (
                    "true", "1", "yes", "completed", "complete")
            else:
                completed = bool(comp_raw)
            og = _get(r, F["order_group"])
            if isinstance(og, bool):               # the afc_first bool of the SPA envelope
                og = "afc_first" if og else "mushra_first"
            sess.append({"listener": pid,
                         "completed": completed,
                         "total_sec": _float_or_nan(_get(r, F["total_sec"])),
                         "order_group": og})
        elif t in ("questionnaire", "screen_q"):
            # Mixing experience is a mandatory item to report in the paper's Methods
            # . Keep the raw record and additionally flatten out the
            # fields used for reporting.
            # The question IDs correspond to selectQ in app.js (8 questions since
            # : q_device/q_hearing/q_prod/q_prod_years/q_know1-3/q_prev).
            #
            # In the real schema (spa_v1) the **pre-**questionnaire is
            # type="questionnaire" and the post-session survey is
            # type="post_questionnaire". The experience and knowledge questions live
            # only in the former, so it is the post-session survey whose flattened
            # fields come out empty. Use the screen_q table for adjudication and for
            # the Methods report.
            row = {"listener": pid}
            row.update(_screen_summary(_screen_answers(r)))
            row["raw"] = {k: v for k, v in r.items()}
            (screenq if t == "screen_q" else quest).append(row)
        elif t == "elig":
            # The verdict the SPA received on the spot (provisional=True marks a pass
            # let through because the server was unreachable).
            p = _get(r, "payload") if isinstance(_get(r, "payload"), dict) else r
            eligrec.append({"listener": pid,
                            "client_eligible": p.get("eligible"),
                            "provisional": bool(p.get("provisional")),
                            "note": p.get("note")})

    data = {
        "mushra": pd.DataFrame(mushra),
        "afc": pd.DataFrame(afc),
        "abx": pd.DataFrame(abx),
        "stage1": pd.DataFrame(stage1),
        "session": pd.DataFrame(sess),
        "questionnaire": pd.DataFrame(quest),
        "screen_q": pd.DataFrame(screenq),
        "elig": pd.DataFrame(eligrec),
    }
    data["_n_raw"] = len(raw)
    data["_n_dupe"] = n_dupe
    return data


def _int_or_nan(v) -> float:
    try:
        return int(v)
    except (TypeError, ValueError):
        return float("nan")


def _float_or_nan(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def all_listeners(data: Dict[str, pd.DataFrame]) -> Set[str]:
    out: Set[str] = set()
    for k in ("mushra", "afc", "abx", "stage1", "session"):
        df = data[k]
        if len(df):
            out |= set(df["listener"])
    return out


# --------------------------------------------------------------------------- #
# Exclusion pipeline (master plan §7.2, applied in this order)
# --------------------------------------------------------------------------- #
def screen_check_frac(g: pd.DataFrame, cfg: dict) -> float:
    """Per-screen basic-check failure rate for a single listener's MUSHRA screens."""
    anchor = cfg["anchor_condition"]
    tol = cfg["screening"]["flat_tol"]
    n_fail = 0
    screens = list(g.groupby(["song", "screen"], dropna=False))
    for _, sg in screens:
        s = sg.set_index("condition")["score"]
        others = s.drop(index=[anchor], errors="ignore")
        fail = False
        if anchor in s.index and len(others) >= 2:
            if s[anchor] > float(np.median(others.values)):      # (i)
                fail = True
        if len(others) >= 2 and float(np.std(others.values)) <= tol:  # (ii)
            fail = True
        n_fail += int(fail)
    return n_fail / len(screens) if screens else 1.0


def repeat_agreement(g: pd.DataFrame) -> float:
    """Choice agreement rate over the repeated pairs. Match on repeat_of when it is
    present, otherwise on the duplicated song."""
    g = g.dropna(subset=["choice"])
    reps = g[g["is_repeat"].astype(bool)]
    orig = g[~g["is_repeat"].astype(bool)]
    if not len(reps):
        return float("nan")
    n, agree = 0, 0
    by_idx = orig.set_index("pair_index")["choice"] if len(orig) else pd.Series(dtype=object)
    by_song: Dict[str, str] = {}
    for _, r in orig.iterrows():
        by_song.setdefault(r["song"], r["choice"])
    for _, r in reps.iterrows():
        ref = None
        if r["repeat_of"] is not None and r["repeat_of"] in by_idx.index:
            ref = by_idx.loc[r["repeat_of"]]
        elif r["song"] in by_song:
            ref = by_song[r["song"]]
        if ref is None:
            continue
        n += 1
        agree += int(ref == r["choice"])
    return agree / n if n else float("nan")


def total_time(lid: str, data: Dict[str, pd.DataFrame]) -> float:
    sess = data["session"]
    if len(sess):
        vals = sess.loc[sess.listener == lid, "total_sec"].values
        vals = [v for v in vals if np.isfinite(v)]
        if vals:                      # multiple rows per stage are possible -> take the max (= the Stage2 session)
            return float(max(vals))
    tot = 0.0
    for k in ("mushra", "afc"):
        df = data[k]
        if not len(df):
            continue
        g = df[df.listener == lid]
        if k == "mushra" and len(g):
            tot += float(g.groupby(["song", "screen"], dropna=False)["dur"]
                         .first().fillna(0.0).sum())
        elif len(g):
            tot += float(g["dur"].fillna(0.0).sum())
    return tot


def exclusion_pipeline(data: Dict[str, pd.DataFrame], cfg: dict
                       ) -> Tuple[Set[str], List[dict]]:
    sc = cfg["screening"]
    table: List[dict] = []
    alive = sorted(all_listeners(data))

    def log_stage(stage: str, desc: str, excluded: List[str]) -> None:
        nonlocal alive
        excluded = sorted(set(excluded) & set(alive))
        alive = [l for l in alive if l not in excluded]
        table.append({"stage": stage, "desc": desc, "n_excluded": len(excluded),
                      "n_remaining": len(alive), "excluded": excluded})

    # 1a. Re-adjudication of the screening questionnaire (single-study design, 
    #
    # In production the SPA decides who proceeds from the response to
    # POST /api/elig_score, but an ineligible participant can reach the main session
    # either way:
    #   - server unreachable (an outage, or the participant cutting the network)
    #     -> provisional pass
    #   - forging the response itself (swapping out fetch in devtools)
    # So, **independently of payment**, the analysis re-adjudicates from the stored
    # questionnaire answers using screen_eligibility.py.
    # Unadjudicable cases (no stored answers) are not excluded by default; they are
    # surfaced in the audit table.
    elig_bad: List[str] = []
    n_adj = n_unknown = 0
    if sc.get("recheck_eligibility", True):
        sq = data.get("screen_q")
        seen: Dict[str, Optional[bool]] = {}
        if sq is not None and len(sq):
            for _, r in sq.iterrows():
                if not r.get("adjudicable"):
                    continue
                seen[r["listener"]] = r["eligible"]
        n_adj = sum(1 for v in seen.values() if v is not None)
        elig_bad = [l for l, ok in seen.items() if ok is False]
        unknown = [l for l in alive if seen.get(l) is None]
        n_unknown = len(unknown)
        if sc.get("require_eligibility_record", False):
            elig_bad += unknown
    log_stage("1a_ineligible",
              "questionnaire re-adjudicated by screen_eligibility.py "
              f"(judged {n_adj}, unjudgeable {n_unknown})", elig_bad)

    # 1. Stage1 HP+AP
    s1 = data["stage1"]
    bad = []
    if len(s1):
        for _, r in s1.iterrows():
            hp, ap = r["hp"], r["ap"]
            if (np.isfinite(hp) and hp < sc["hp_min"]) or \
               (np.isfinite(ap) and ap < sc["ap_min"]):
                bad.append(r["listener"])
    if sc["require_stage1"]:
        have = set(s1["listener"]) if len(s1) else set()
        bad += [l for l in alive if l not in have]
    log_stage("1_stage1_hp_ap", f"HP<{sc['hp_min']}/6 or AP<{sc['ap_min']}/6", bad)

    # 2. Stage2 incomplete (session can have several rows, one per stage: completed
    #    in any of them counts as completed)
    sess = data["session"]
    completed_flag: Dict[str, bool] = {}
    if len(sess):
        for _, r in sess.iterrows():
            c = r["completed"]
            if c is None or (isinstance(c, float) and not np.isfinite(c)):
                continue
            lid0 = r["listener"]
            completed_flag[lid0] = bool(completed_flag.get(lid0, False) or bool(c))
    mus, afc = data["mushra"], data["afc"]
    # Fallback for participants without a completed flag. Design A  does
    # not present 2AFC, so a 2AFC trial count must not be required. Requiring it
    # would exclude every design A finisher as "incomplete" merely for dropping the
    # session_end POST.
    #
    # **The decision is made per participant (fixed in the  integration
    # check).** Previously it looked at the whole dataset via `len(afc)`, so a single
    # row from a past end-to-end run that included 2AFC (TESTRUN and the like,
    # without the demo flag) was enough to reinstate the requirement and reproduce
    # exactly the accident this was meant to prevent.
    #   afc_block_expected = None  : auto. (a) If the share of participants with 2AFC
    #                                rows is at least afc_block_min_share, assume the
    #                                block was presented and require it of
    #                                **everyone**; (b) otherwise require it only of
    #                                **participants that do have afc rows**.
    #                                (a) is needed so that, in a the reported runs with
    #                                the blocks restored, someone who dropped the afc
    #                                POSTs entirely does not slip through.
    #   afc_block_expected = True  : require it of everyone
    #   afc_block_expected = False : require it of nobody
    afc_expected = sc.get("afc_block_expected")
    need_afc_all = int(sc["required_afc_trials"])
    afc_listeners = set(afc["listener"]) if len(afc) else set()
    n_alive = len(alive)
    share = (len(afc_listeners & set(alive)) / n_alive) if n_alive else 0.0
    min_share = float(sc.get("afc_block_min_share", 0.5))
    if afc_expected is None:
        afc_block_on = bool(n_alive) and share >= min_share
        how = (f"auto: share of participants with 2AFC rows {share:.2f} "
               f"{'>=' if afc_block_on else '<'} {min_share:.2f} "
               f"-> require 2AFC of {'everyone' if afc_block_on else 'only those with afc rows'}")
    else:
        afc_block_on = bool(afc_expected)
        how = f"config: afc_block_expected={afc_expected}"
    bad = []
    for lid in alive:
        flag = completed_flag.get(lid)
        if flag is not None:
            ok = bool(flag)
        else:
            n_scr = (mus[mus.listener == lid].groupby(["song", "screen"],
                                                      dropna=False).ngroups
                     if len(mus) else 0)
            n_afc = int((afc.listener == lid).sum()) if len(afc) else 0
            need_afc = need_afc_all if (afc_block_on or n_afc) else 0
            ok = (n_scr >= sc["required_mushra_screens"]
                  and n_afc >= need_afc)
        if not ok:
            bad.append(lid)
    log_stage("2_incomplete",
              "Stage2 incomplete (completed flag / screen count; " + how + ")",
              bad)

    # 3. MUSHRA basic check failed on > 20% of screens
    bad = []
    if len(mus):
        for lid in alive:
            g = mus[mus.listener == lid]
            if not len(g):
                continue
            if screen_check_frac(g, cfg) > sc["screen_fail_frac"]:
                bad.append(lid)
    log_stage("3_screen_checks",
              "dry>median(others) or zero-variance on >20% of screens", bad)

    # 4. speeder (below 0.5 x the median of the survivors)
    times = {lid: total_time(lid, data) for lid in alive}
    finite = [t for t in times.values() if np.isfinite(t) and t > 0]
    bad = []
    med = float(np.median(finite)) if finite else float("nan")
    if np.isfinite(med):
        thr = sc["speeder_factor"] * med
        bad = [l for l, t in times.items() if np.isfinite(t) and t < thr]
    log_stage("4_speeder",
              f"total time < {sc['speeder_factor']} x median ({med:.0f}s)", bad)

    # 5. 2AFC repeat agreement < 25%
    bad = []
    if len(afc):
        for lid in alive:
            agree = repeat_agreement(afc[afc.listener == lid])
            if np.isfinite(agree) and agree < sc["repeat_agree_min"]:
                bad.append(lid)
    log_stage("5_afc_repeat", f"repeat agreement < {sc['repeat_agree_min']:.0%}", bad)

    # 6. IQR rule (listener mean score)
    bad = []
    if len(mus):
        means = (mus[mus.listener.isin(alive)]
                 .groupby("listener")["score"].mean())
        if len(means) >= 4:
            q1, q3 = np.percentile(means.values, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - sc["iqr_k"] * iqr, q3 + sc["iqr_k"] * iqr
            bad = [l for l, m in means.items() if m < lo or m > hi]
    log_stage("6_iqr_outlier", f"listener mean score outside Tukey fences "
                               f"(k={sc['iqr_k']})", bad)
    return set(alive), table


def eligibility_audit(data: Dict[str, pd.DataFrame]) -> dict:
    """Audit of whether the questionnaire screening actually took effect
    (single-study design, .

    When /api/elig_score is unreachable the SPA lets the participant into the main
    session on a **provisional pass**. The analysis therefore cross-checks three
    things.

      A. Re-adjudication from the stored answers  (screen_eligibility.py)
      B. The verdict the SPA received             (record type "elig_score")
      C. The verdict the server issued            (record type "elig_score_server")

    A disagreeing with B, B being provisional, or B passing with no C at all are all
    signs of "reached the main session without being adjudicated". Exclusion is done
    from A only (so there is a single rule), while B and C are used to report how
    many participants were in each state.
    """
    sq = data.get("screen_q")
    el = data.get("elig")
    recheck: Dict[str, Optional[bool]] = {}
    versions: Set[str] = set()
    if sq is not None and len(sq):
        for _, r in sq.iterrows():
            if r.get("adjudicable"):
                recheck[r["listener"]] = r["eligible"]
                if r.get("criteria_version"):
                    versions.add(str(r["criteria_version"]))
    prov: List[str] = []
    client_ok: Dict[str, Any] = {}
    if el is not None and len(el):
        for _, r in el.iterrows():
            client_ok[r["listener"]] = r["client_eligible"]
            if r["provisional"]:
                prov.append(r["listener"])
    disagree = sorted(l for l, ok in recheck.items()
                      if ok is False and client_ok.get(l) is True)
    return {"n_judged": sum(1 for v in recheck.values() if v is not None),
            "n_ineligible_on_recheck": sum(1 for v in recheck.values()
                                           if v is False),
            "provisional_pass": sorted(set(prov)),
            "client_passed_but_ineligible": disagree,
            "no_elig_record": sorted(set(recheck) - set(client_ok)),
            "criteria_versions": sorted(versions)}


def print_eligibility_audit(a: dict) -> None:
    if not a["n_judged"] and not a["provisional_pass"]:
        return                     # no questionnaire records (old data / synthetic data)
    print("\n## Screening questionnaire audit (single-study design)\n")
    print(f"  Adjudicable participants        : {a['n_judged']} "
          f"(criteria {', '.join(a['criteria_versions']) or 'n/a'})")
    print(f"  Ineligible on re-adjudication   : {a['n_ineligible_on_recheck']}")
    print(f"  provisional pass (server unreachable): {len(a['provisional_pass'])} "
          f"{a['provisional_pass'][:8]}")
    print(f"  SPA passed but re-adjudged ineligible: "
          f"{len(a['client_passed_but_ineligible'])} "
          f"{a['client_passed_but_ineligible'][:8]}")
    print(f"  No verdict record (check this)  : {len(a['no_elig_record'])} "
          f"{a['no_elig_record'][:8]}")
    if a["provisional_pass"] or a["client_passed_but_ineligible"]:
        print("  Note: payment is already done. Exclusion here is only from the analysis.")


def print_exclusion_table(table: List[dict], n_start: int) -> None:
    print("\n## Exclusion pipeline (in the order of master plan §7.2)\n")
    print(f"| stage | criterion | excluded | remaining | excluded pid |")
    print("|---|---|---|---|---|")
    print(f"| 0_start | all listeners seen | - | {n_start} | |")
    for row in table:
        ids = ", ".join(row["excluded"][:8]) + \
              (" ..." if len(row["excluded"]) > 8 else "")
        print(f"| {row['stage']} | {row['desc']} | {row['n_excluded']} "
              f"| {row['n_remaining']} | {ids} |")


# --------------------------------------------------------------------------- #
# General-purpose variance-components REML (Woodbury; handles crossed effects like lme4)
# --------------------------------------------------------------------------- #
def reml_vc(y: np.ndarray, X: np.ndarray, Zs: List[np.ndarray]) -> dict:
    """Solve y = X beta + sum_i Z_i b_i + e,  b_i ~ N(0, v_i I),  e ~ N(0, ve I) by REML.

    Applies Woodbury to V = ve I + U D U^T (U = [Z_1 .. Z_k]) so the evaluation uses a
    q x q Cholesky (q = total number of random effects) instead of an n x n one. Fast
    even for crossed designs.
    Returns: beta, cov_beta, sd = [sqrt(v_1..v_k), sqrt(ve)], converged.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    sizes = [Z.shape[1] for Z in Zs]
    U = np.hstack(Zs) if Zs else np.zeros((n, 0))
    q = U.shape[1]
    A = U.T @ U                       # q x q
    AX = U.T @ X                      # q x p
    ay = U.T @ y                      # q
    XtX, Xty, yty = X.T @ X, X.T @ y, float(y @ y)
    vy = max(float(np.var(y, ddof=1)) if n > 1 else 1.0, 1e-6)
    blk = np.repeat(np.arange(len(sizes)), sizes)     # block id per column
    Iq = np.eye(q)

    def core(theta: np.ndarray):
        v = np.exp(theta[:-1])
        ve = float(np.exp(theta[-1]))
        w = np.sqrt(v)[blk] if q else np.zeros(0)
        M = Iq + (w[:, None] * A * w[None, :]) / ve
        c = np.linalg.cholesky(M)
        logdetV = n * math.log(ve) + 2.0 * float(np.sum(np.log(np.diag(c))))

        def vinv_quad(Ba, ua: np.ndarray, ub: np.ndarray):
            """Compute a^T V^-1 b from (a^T b, U^T a, U^T b) (Woodbury)."""
            ua2 = ua.reshape(q, -1)
            ub2 = ub.reshape(q, -1)
            ta = np.linalg.solve(c, w[:, None] * ua2)
            tb = np.linalg.solve(c, w[:, None] * ub2)
            return (Ba - np.squeeze(ta.T @ tb) / ve) / ve

        XtViX = vinv_quad(XtX, AX, AX).reshape(p, p)
        XtViy = vinv_quad(Xty, AX, ay).reshape(p)
        ytViy = float(vinv_quad(yty, ay, ay))
        return logdetV, XtViX, XtViy, ytViy

    def neg_reml(theta: np.ndarray) -> float:
        try:
            logdetV, XtViX, XtViy, ytViy = core(theta)
            sign, logdetX = np.linalg.slogdet(XtViX)
            if sign <= 0:
                return 1e12
            beta = np.linalg.solve(XtViX, XtViy)
            quad = float(ytViy - XtViy @ beta)
            return 0.5 * (logdetV + logdetX + quad)
        except np.linalg.LinAlgError:
            return 1e12

    k = len(Zs)
    bounds = [(math.log(vy * 1e-8), math.log(vy * 50))] * (k + 1)
    best = None
    inits = [[vy / (2 * k + 2)] * k + [vy / 2],
             [vy / (10 * k)] * k + [vy * 0.8]]
    for init in inits:
        res = optimize.minimize(
            neg_reml, np.log(np.maximum(init, 1e-9)), method="L-BFGS-B",
            bounds=bounds, options={"ftol": 1e-11, "gtol": 1e-8, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    logdetV, XtViX, XtViy, ytViy = core(best.x)
    cov = np.linalg.inv(XtViX)
    beta = cov @ XtViy
    v = np.exp(best.x)
    return {"beta": beta, "cov_beta": cov,
            "sd_components": [float(math.sqrt(x)) for x in v[:-1]],
            "sigma_resid": float(math.sqrt(v[-1])),
            "converged": bool(best.success), "neg_reml": float(best.fun),
            "n": n}


def _indicator(codes: np.ndarray) -> np.ndarray:
    _, inv = np.unique(codes, return_inverse=True)
    Z = np.zeros((len(codes), inv.max() + 1))
    Z[np.arange(len(codes)), inv] = 1.0
    return Z


def reml_crossed_diff(y: np.ndarray, lid: np.ndarray, sid: np.ndarray) -> dict:
    """Within-screen difference model of a contrast, d_ls = mu + u_l + v_s + e_ls
    (see the module docstring)."""
    y = np.asarray(y, dtype=float)
    L, S = len(np.unique(lid)), len(np.unique(sid))
    r = reml_vc(y, np.ones((len(y), 1)), [_indicator(lid), _indicator(sid)])
    return {"mu": float(r["beta"][0]), "se": float(math.sqrt(r["cov_beta"][0, 0])),
            "df": max(min(L, S) - 1, 2), "n": len(y), "L": L, "S": S,
            "tau_LC": r["sd_components"][0], "tau_SC": r["sd_components"][1],
            "sigma_d": r["sigma_resid"], "converged": r["converged"],
            "neg_reml": r["neg_reml"]}


def reml_joint(mus: pd.DataFrame, cfg: dict) -> dict:
    """Solve the full model of master plan §7.3 jointly over all 6 conditions:

        score ~ condition + presentation_order
              + (1|listener) + (1|song) + (1|listener:cond) + (1|song:cond)

    This is the preferred route for measuring the variance components (the main
    purpose of the pilot): the per-contrast difference model only has S song x cond
    cells per contrast, giving tau_SC a sampling error of ~30%, whereas the joint
    model uses all S x C cells and is far more precise.
    To convert a per-condition component tau to the difference scale of
    master_power.py, multiply by sqrt(2) (d is a difference of 2 conditions, so the
    variance doubles).
    """
    conds = [c for c in cfg["conditions"] if c in set(mus.condition)]
    d = mus[mus.condition.isin(conds)].copy()
    d["order_c"] = d.groupby("listener")["screen"].transform(
        lambda s: s.astype(float) - s.astype(float).mean()).fillna(0.0)
    y = d["score"].values.astype(float)
    cat = pd.Categorical(d["condition"], categories=conds)
    dummies = pd.get_dummies(cat, drop_first=False)
    ref = "proposed" if "proposed" in conds else conds[0]
    fe_conds = [c for c in conds if c != ref]
    X = np.column_stack([np.ones(len(d))]
                        + [dummies[c].values.astype(float) for c in fe_conds]
                        + [d["order_c"].values])
    lid = d["listener"].values
    sid = d["song"].values
    lc = np.char.add(np.char.add(lid.astype(str), "|"),
                     d["condition"].values.astype(str))
    sc = np.char.add(np.char.add(sid.astype(str), "|"),
                     d["condition"].values.astype(str))
    r = reml_vc(y, X, [_indicator(lid), _indicator(sid),
                       _indicator(lc), _indicator(sc)])
    sd_l, sd_s, sd_lc, sd_sc = r["sd_components"]
    s2 = math.sqrt(2.0)
    out = {
        "ref_condition": ref,
        "fixed_effects": {"intercept": float(r["beta"][0]),
                          **{c: float(b) for c, b in zip(fe_conds, r["beta"][1:-1])},
                          "order_slope": float(r["beta"][-1])},
        "fe_se": {c: float(math.sqrt(r["cov_beta"][i + 1, i + 1]))
                  for i, c in enumerate(fe_conds)},
        "sd_listener": sd_l, "sd_song": sd_s,
        "sd_listener_cond": sd_lc, "sd_song_cond": sd_sc,
        "sigma_rating": r["sigma_resid"],
        # Difference scale of master_power.py (plugs straight into the SE formula)
        "tau_LC_diff": sd_lc * s2, "tau_SC_diff": sd_sc * s2,
        "sigma_d_diff": r["sigma_resid"] * s2,
        "converged": r["converged"], "n": r["n"],
    }
    return out


def t_pvalues(mu: float, se: float, df: float) -> dict:
    tval = mu / se if se > 0 else float("inf")
    return {
        "t": tval,
        "p_greater": float(1 - stats.t.cdf(tval, df)),
        "p_less": float(stats.t.cdf(tval, df)),
        "p_two": float(2 * (1 - stats.t.cdf(abs(tval), df))),
    }


def holm(p: Sequence[float]) -> List[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


# --------------------------------------------------------------------------- #
# H1: one-sided non-inferiority test on the ratings (design A; internal notes §3 / §5.4b)
# --------------------------------------------------------------------------- #
def noninf_stats(r: dict, an: dict) -> dict:
    """Build the non-inferiority test statistics from the within-screen difference
    estimate r.

    On d = score_a - score_b (a = proposed, b = professional), a one-sided t test of
        H0: mu <= -Delta     H1: mu > -Delta
    with t = (mu_hat + Delta) / SE and the same df = min(L,S)-1 as the contrast.
    Non-inferiority can be claimed when the one-sided 95% lower bound exceeds -Delta
    (equivalent to p < 0.05).

    **Delta is not decided here.** It is taken as-is from
    analysis.noninf_margin_points in the config (the grounds are in the
    DEFAULT_CONFIG comments). Power is reported separately by
    noninferiority_report; this function never looks at power.
    """
    delta = float(an["noninf_margin_points"])
    mu, se, df = float(r["mu"]), float(r["se"]), float(r["df"])
    if se > 0:
        t_ni = (mu + delta) / se
        p_ni = float(1 - stats.t.cdf(t_ni, df))
    else:                     # degenerate SE=0 case (does not occur with real data)
        t_ni = math.inf if (mu + delta) > 0 else -math.inf
        p_ni = 0.0 if t_ni > 0 else 1.0
    lo_one = mu - float(stats.t.ppf(0.95, df)) * se
    return {"noninf_margin": delta,
            "noninf_margin_source": str(an.get("noninf_margin_source", "")),
            "t_noninf": float(t_ni), "p_noninf": p_ni,
            "ci95_lower_one_sided": float(lo_one),
            "noninferior": bool(lo_one > -delta)}


def noninf_contrast_name(cfg: dict) -> Optional[str]:
    """Name of the contrast defined as the non-inferiority test (H1_topline_noninf
    under design A)."""
    for con in cfg["contrasts"]:
        if con.get("side") == "noninf":
            return str(con["name"])
    return None


def noninferiority_report(mushra_out: dict, cfg: dict) -> dict:
    """Present Delta for H1 first, and report the achievable power **afterwards**.

    What design_a_spec §3 forbids is the reverse order (deciding Delta from power).
    The order of the printout and of the return value fixes that procedure:
      1. Delta and its external grounds
      2. Its relation to the minimum detectable effect at planning time (6.27
         points). If Delta falls below it, state explicitly that "power does not
         reach 80%"
      3. The power for this Delta, computed from the variance components measured in
         the pilot
    """
    an = cfg["analysis"]
    name = noninf_contrast_name(cfg)
    rec = (mushra_out.get("contrasts") or {}).get(name or "")
    delta = float(an["noninf_margin_points"])
    floor = float(an["noninf_planned_mde_points"])
    L = int(an["noninf_planned_L"])
    S, S_per = int(an["S_main"]), int(an["S_per"])
    alpha = an["alpha_family"] / an["n_primary"]
    target = float(an["power_target"])

    out: Dict[str, Any] = {
        "contrast": name, "margin_points": delta,
        "margin_source": str(an.get("noninf_margin_source", "")),
        "planned_mde_points": floor, "planned_L": L, "S": S, "S_per": S_per,
        "alpha_one_sided": alpha, "power_target": target,
        "margin_below_planned_mde": bool(delta < floor),
    }
    print("\n## H1 non-inferiority margin and power (read in this order)\n")
    print(f"  1. Non-inferiority margin Delta = {delta:.2f} points (0-100 scale)")
    print(f"     Grounds: {out['margin_source']}")
    print("     Delta was decided from external grounds first. It is not back-solved "
          "from power (design_a_spec §3).")
    print(f"  2. Minimum detectable effect at planning time (L={L} participants, "
          f"{S_per} songs each, one-sided "
          f"alpha={alpha:.4f}, power {target:.0%}) = {floor:.2f} points")
    if out["margin_below_planned_mde"]:
        print(f"     ** Delta = {delta:.2f} points falls below {floor:.2f} points. "
              f"Power {target:.0%} is not reachable at this scale.** The achieved value is reported below.")
    else:
        print(f"     Delta >= {floor:.2f} points, so **under the planning-time variance "
              f"assumptions** the power at a true difference of 0 is at least {target:.0%}. "
              f"See 3 below for the value achieved with the measured components (it drops if the variance is larger than planned).")

    # 3. Power from the measured variance components. Components come from the joint
    #    model when available (same as recompute_design).
    jm = mushra_out.get("joint_model")
    comp: Dict[str, float] = {}
    if jm and jm.get("converged", True):
        comp = {"tau_LC": float(jm["tau_LC_diff"]), "tau_SC": float(jm["tau_SC_diff"]),
                "sigma_d": float(jm["sigma_d_diff"])}
        comp_src = "joint_model"
    elif rec:
        comp = {k: float(rec[k]) for k in ("tau_LC", "tau_SC", "sigma_d")}
        comp_src = f"contrast {name}"
    if not comp:
        print("  3. Variance components unavailable, so power is not computed (not enough MUSHRA data)")
        return out
    se_plan = math.sqrt(comp["tau_LC"] ** 2 / L + comp["tau_SC"] ** 2 / S
                        + comp["sigma_d"] ** 2 / (L * min(S_per, S)))
    z_a = float(stats.norm.ppf(1 - alpha))
    mde_obs = (z_a + float(stats.norm.ppf(target))) * se_plan
    powers = {}
    print(f"  3. Variance components measured in the pilot ({comp_src}: tau_LC={comp['tau_LC']:.2f} "
          f"tau_SC={comp['tau_SC']:.2f} sigma_d={comp['sigma_d']:.2f}) "
          f"-> SE={se_plan:.2f} points")
    print(f"     Minimum detectable effect with the measured components = {mde_obs:.2f} points "
          f"(look honestly at the gap against the planned {floor:.2f} points here)")
    print("     | true difference (points by which proposed is worse) | power for Delta |")
    print("     |---|---|")
    for gap in an["noninf_report_gaps"]:
        pw = float(stats.norm.cdf((delta - float(gap)) / se_plan - z_a))
        powers[str(float(gap))] = pw
        flag = "" if pw >= target else f"  <- below {target:.0%} (not hidden)"
        print(f"     | {float(gap):+.1f} | {pw:.3f}{flag} |")
    out.update({"components": comp, "components_source": comp_src,
                "se_planned": se_plan, "mde_observed_components": mde_obs,
                "power_by_true_gap": powers})
    if rec:
        out.update({k: rec[k] for k in
                    ("mu", "se", "df", "p_noninf", "ci95_lower_one_sided",
                     "noninferior") if k in rec})
    return out


# --------------------------------------------------------------------------- #
# MUSHRA analysis
# --------------------------------------------------------------------------- #
def mushra_analysis(mus: pd.DataFrame, cfg: dict) -> dict:
    an = cfg["analysis"]
    out: Dict[str, Any] = {"condition_means": {}, "contrasts": {}}
    conds = [c for c in cfg["conditions"] if c in set(mus.condition)]

    print("\n## MUSHRA-lite: condition means (aggregated per listener)\n")
    piv_l = mus.pivot_table(index="listener", columns="condition", values="score")
    print(f"| condition | mean | sd | median | n listener |")
    print("|---|---|---|---|---|")
    for c in conds:
        v = piv_l[c].dropna().values
        out["condition_means"][c] = {
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "median": float(np.median(v)), "n": int(len(v))}
        print(f"| {c} | {v.mean():.1f} | {v.std(ddof=1):.1f} "
              f"| {np.median(v):.1f} | {len(v)} |")

    # Per-contrast within-screen difference -> crossed REML
    cell = mus.pivot_table(index=["listener", "song"], columns="condition",
                           values="score")
    scr_idx = mus.pivot_table(index=["listener", "song"], values="screen",
                              aggfunc="first")
    print("\n## Contrasts (crossed REML on the within-screen differences, df=min(L,S)-1)\n")
    print("| contrast | family | n | mu +- SE | 95% CI | t(df) | p(one-sided) | p(two-sided) "
          "| tau_LC | tau_SC | sigma_d |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for con in cfg["contrasts"]:
        a, b = con["a"], con["b"]
        if a not in cell.columns or b not in cell.columns:
            continue
        sub = cell[[a, b]].dropna()
        if len(sub) < 8:
            continue
        d = (sub[a] - sub[b]).values
        lid = sub.index.get_level_values("listener").values
        sid = sub.index.get_level_values("song").values
        r = reml_crossed_diff(d, lid, sid)
        pv = t_pvalues(r["mu"], r["se"], r["df"])
        tcrit95 = stats.t.ppf(0.975, r["df"])
        tcrit90 = stats.t.ppf(0.95, r["df"])
        ci95 = (r["mu"] - tcrit95 * r["se"], r["mu"] + tcrit95 * r["se"])
        ci90 = (r["mu"] - tcrit90 * r["se"], r["mu"] + tcrit90 * r["se"])
        # Exploratory: effect of presentation order (its main effect does not show up
        # in the within-screen difference)
        order_slope = float("nan")
        try:
            oi = scr_idx.loc[sub.index, "screen"].astype(float).values
            if np.nanstd(oi) > 0:
                order_slope = float(np.polyfit(oi - np.nanmean(oi), d, 1)[0])
        except Exception:
            pass
        rec = {**con, **r, **pv,
               "ci95": [float(ci95[0]), float(ci95[1])],
               "ci90": [float(ci90[0]), float(ci90[1])],
               "order_slope_per_screen": order_slope}
        m = an["tost_margin_points"]
        if con["family"] == "aux_tost":
            # The default contrasts of design A contain no aux_tost (H1 is the
            # non-inferiority test). Kept here in case one is added via the config.
            rec["tost_margin"] = m
            rec["tost_equivalent_ci95"] = bool(-m < ci95[0] and ci95[1] < m)
            rec["tost_equivalent_ci90"] = bool(-m < ci90[0] and ci90[1] < m)
        if con.get("side") == "noninf":
            rec.update(noninf_stats(r, an))
        out["contrasts"][con["name"]] = rec
        p_one = pv["p_greater"] if con["side"] == "greater" else pv["p_two"]
        print(f"| {con['name']} ({a}-{b}) | {con['family']} | {r['n']} "
              f"| {r['mu']:+.2f} +- {r['se']:.2f} "
              f"| [{ci95[0]:+.2f},{ci95[1]:+.2f}] | {pv['t']:.2f}({r['df']}) "
              f"| {pv['p_greater']:.3g} | {pv['p_two']:.3g} "
              f"| {r['tau_LC']:.1f} | {r['tau_SC']:.1f} | {r['sigma_d']:.1f} |")
        if con["family"] == "aux_tost":
            print(f"|   TOST +-{m:.0f}pt: inside 95%CI={rec['tost_equivalent_ci95']} "
                  f"/ inside 90%CI={rec['tost_equivalent_ci90']} |||||||||||")
        if con.get("side") == "noninf":
            print(f"|   non-inferiority (H0: mu <= -{rec['noninf_margin']:.1f}pt): "
                  f"one-sided p = {rec['p_noninf']:.3g}, one-sided 95% lower bound = "
                  f"{rec['ci95_lower_one_sided']:+.2f} points -> "
                  f"{'non-inferiority can be claimed' if rec['noninferior'] else 'non-inferiority cannot be claimed'}"
                  f" |||||||||||")

    # joint model (the full model of §7.3): the preferred route for measuring the
    # variance components
    try:
        jm = reml_joint(mus, cfg)
        out["joint_model"] = jm
        print("\n## joint crossed model (score ~ condition + order + 4 VC + resid)\n")
        print(f"  fixed effects (ref={jm['ref_condition']}): "
              + ", ".join(f"{c}={v:+.2f}" for c, v in jm["fixed_effects"].items()
                          if c not in ("intercept",)))
        print(f"  sd_listener={jm['sd_listener']:.2f}  sd_song={jm['sd_song']:.2f}  "
              f"sd_listener:cond={jm['sd_listener_cond']:.2f}  "
              f"sd_song:cond={jm['sd_song_cond']:.2f}  "
              f"sigma_rating={jm['sigma_rating']:.2f}")
        print(f"  Difference scale (master_power.py definition): "
              f"tau_LC={jm['tau_LC_diff']:.2f}  tau_SC={jm['tau_SC_diff']:.2f}  "
              f"sigma_d={jm['sigma_d_diff']:.2f}  (converged={jm['converged']})")
    except Exception as e:                                       # noqa: BLE001
        print(f"[warn] joint model failed: {e}")
    return out


def full_lmm_check(mus: pd.DataFrame, base: str = "proposed") -> Optional[dict]:
    """Cross-check with statsmodels MixedLM (crossed, via a VC specification).
    Slow and unstable, so it is optional."""
    try:
        import statsmodels.formula.api as smf
    except Exception as e:                                       # noqa: BLE001
        print(f"[warn] statsmodels not available: {e}")
        return None
    d = mus.copy()
    d["grp"] = 1
    d["order_c"] = d.groupby("listener")["screen"].transform(
        lambda s: (s - s.mean()))
    try:
        md = smf.mixedlm(
            f"score ~ C(condition, Treatment(reference='{base}')) + order_c",
            d, groups="grp",
            vc_formula={"listener": "0+C(listener)", "song": "0+C(song)",
                        "lc": "0+C(listener):C(condition)",
                        "sc": "0+C(song):C(condition)"},
            re_formula="0")
        res = md.fit(reml=True, method="lbfgs", maxiter=200)
        print("\n## statsmodels MixedLM (crossed VC) cross-check\n")
        print(res.summary())
        return {"converged": bool(res.converged),
                "vcomp": {k: float(v) for k, v in
                          zip(res.model.exog_vc.names, np.sqrt(res.vcomp))}}
    except Exception as e:                                       # noqa: BLE001
        print(f"[warn] full crossed MixedLM failed (expected): {e}")
        return None


# --------------------------------------------------------------------------- #
# 2AFC analysis (the old H1. **Not run** under design A, but the implementation stays)
# --------------------------------------------------------------------------- #
# Design A (approved  does not present the 2AFC block, so this function
# receives 0 rows. In that case it simply returns "not applicable"
# (applicable=False): it does not crash and it does not enter the primary family.
# Setting blocks.afc back to true in app/config.js makes the SPA present 2AFC again,
# and this code works **as-is** (do not delete it).
# The primary test for H1 moved to the non-inferiority test on the ratings
# (noninf_stats / noninferiority_report).
def _not_applicable(block: str, reason: str) -> Dict[str, Any]:
    print(f"\n## {block}: not applicable ({reason})")
    return {"applicable": False, "block": block, "reason": reason,
            "n_trials": 0}


def afc_analysis(afc: pd.DataFrame, cfg: dict) -> dict:
    an = cfg["analysis"]
    target, other = cfg["afc_pair"]
    out: Dict[str, Any] = {}
    if afc is None or not len(afc):
        return _not_applicable(
            "2AFC (old H1)",
            "0 2AFC records. Not presented under design A "
            "(app/config.js blocks.afc = false)")
    g = afc.dropna(subset=["choice"]).copy()
    firsts = g[~g["is_repeat"].astype(bool)]
    if not len(firsts):
        return _not_applicable("2AFC (old H1)",
                               "0 first-presentation 2AFC records")
    firsts = firsts.assign(y=(firsts["choice"] == target).astype(int))

    # Per-listener preference rate
    rates = firsts.groupby("listener")["y"].mean()
    Ln = len(rates)
    mean_rate = float(rates.mean())
    sd_rate = float(rates.std(ddof=1)) if Ln > 1 else float("nan")
    se_rate = sd_rate / math.sqrt(Ln) if Ln > 1 else float("nan")
    tcrit = stats.t.ppf(0.975, Ln - 1) if Ln > 1 else float("nan")
    ci95 = (mean_rate - tcrit * se_rate, mean_rate + tcrit * se_rate)
    # One-sided Wilcoxon (p_pref > 0.5)
    diffs = rates.values - 0.5
    if np.allclose(diffs, 0) or Ln < 5:
        p_wil = float("nan")
    else:
        try:
            p_wil = float(stats.wilcoxon(diffs, alternative="greater",
                                         zero_method="wilcox").pvalue)
        except ValueError:
            p_wil = float("nan")
    # TOST: 95% CI ⊂ [0.44, 0.56]
    lo, hi = an["tost_pref_lo"], an["tost_pref_hi"]
    tost_ok = bool(np.isfinite(ci95[0]) and lo < ci95[0] and ci95[1] < hi)

    # binomial GLMM: choice ~ 1 + (1|listener) + (1|song)
    glmm: Dict[str, Any] = {"ok": False}
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        dd = firsts.rename(columns={"listener": "lst"})[["y", "lst", "song"]].copy()
        md = BinomialBayesMixedGLM.from_formula(
            "y ~ 1", {"lst": "0 + C(lst)", "song": "0 + C(song)"}, dd)
        fit = md.fit_vb()
        b0, s0 = float(fit.fe_mean[0]), float(fit.fe_sd[0])
        glmm = {"ok": True, "intercept_logit": b0, "intercept_sd": s0,
                "pref_rate_glmm": float(1 / (1 + math.exp(-b0))),
                "p_one_sided": float(1 - stats.norm.cdf(b0 / s0)),
                "note": "VB approximation (the posterior SD tends to be too small). "
                        "For robustness see the Wilcoxon / listener-level t"}
    except Exception as e:                                       # noqa: BLE001
        glmm["error"] = str(e)

    rep_agree = {lid: repeat_agreement(g[g.listener == lid])
                 for lid in g.listener.unique()}
    rep_vals = [v for v in rep_agree.values() if np.isfinite(v)]

    per_song = firsts.groupby("song")["y"].agg(["mean", "count"])

    out = {
        "applicable": True,
        # Even if 2AFC comes back, the primary test for H1 stays the non-inferiority
        # test on the ratings. This is auxiliary.
        "role": "auxiliary (the primary family stays at 3 tests; design_a_spec §3)",
        "target": target, "other": other,
        "n_listeners": Ln, "n_trials_first": int(len(firsts)),
        "pref_rate_mean": mean_rate, "pref_rate_sd": sd_rate,
        "pref_rate_ci95": [float(ci95[0]), float(ci95[1])],
        "p_wilcoxon_one_sided": p_wil,
        "tost_band": [lo, hi], "tost_equivalent": tost_ok,
        "glmm": glmm,
        "repeat_agreement_mean": float(np.mean(rep_vals)) if rep_vals else float("nan"),
        "per_song_rate": {s: {"rate": float(r["mean"]), "n": int(r["count"])}
                          for s, r in per_song.iterrows()},
    }
    print(f"\n## 2AFC (old H1: {target} vs {other}, first presentations only, {len(firsts)} trials. "
          f"**Auxiliary**. The primary test is the non-inferiority test on the ratings)\n")
    print(f"  Preference rate P({target}) = {mean_rate:.3f} +- {sd_rate:.3f} (per listener, "
          f"L={Ln}), 95% CI [{ci95[0]:.3f}, {ci95[1]:.3f}]")
    if glmm.get("ok"):
        print(f"  GLMM intercept (logit) = {glmm['intercept_logit']:+.3f} "
              f"(sd {glmm['intercept_sd']:.3f}) -> rate {glmm['pref_rate_glmm']:.3f}, "
              f"one-sided p = {glmm['p_one_sided']:.3g}")
    else:
        print(f"  [warn] GLMM failed: {glmm.get('error')}")
    print(f"  One-sided Wilcoxon (p>0.5): p = {p_wil:.3g}")
    print(f"  TOST equivalence (95% CI ⊂ [{lo},{hi}]): {tost_ok}")
    print(f"  Repeat agreement (reliability): {out['repeat_agreement_mean']:.2f}")
    return out


# --------------------------------------------------------------------------- #
# ABX aggregation (pilot only; H2 demotion decision)
# --------------------------------------------------------------------------- #
# Design A does not present ABX either (app/config.js blocks.abx = false). With 0
# rows it returns "not applicable" and no H2 demotion decision happens. Do not delete
# the implementation (restore the setting and it works).
# As stated in design_a_spec §3, ABX has no direction, so it was never evidence for
# H2 (reward hacking) in the first place. The demotion rule itself is kept.
def abx_analysis(abx: pd.DataFrame, cfg: dict) -> dict:
    if abx is None or not len(abx):
        return _not_applicable(
            "ABX", "0 ABX records. Not presented under design A "
                   "(app/config.js blocks.abx = false) -> no H2 demotion decision is made")
    thr = cfg["analysis"]["abx_demote_threshold"]
    acc = abx.groupby("listener")["correct"].mean()
    grand = float(acc.mean())
    n = len(acc)
    se = float(acc.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    ci = (grand - tcrit * se, grand + tcrit * se)
    k = int(abx["correct"].sum())
    m = int(len(abx))
    p_binom = float(stats.binomtest(k, m, 0.5, alternative="greater").pvalue)
    demote = bool(grand <= thr)
    out = {"applicable": True, "n_listeners": n, "n_trials": m,
           "grand_accuracy": grand, "acc_ci95_listener": [float(ci[0]), float(ci[1])],
           "p_binom_pooled_greater_chance": p_binom,
           "demote_threshold": thr, "demote_H2": demote,
           "per_listener": {l: float(a) for l, a in acc.items()}}
    print(f"\n## ABX block (H2 discriminability)\n")
    print(f"  Group mean accuracy = {grand:.3f} (per-listener 95% CI "
          f"[{ci[0]:.3f}, {ci[1]:.3f}], L={n}, {m} trials)")
    print(f"  pooled binomial test (>0.5): p = {p_binom:.3g}")
    print(f"  H2 demotion decision (group mean <= {thr:.2f}): {'demote' if demote else 'keep'}")
    return out


# --------------------------------------------------------------------------- #
# Measured variance components -> recomputing L for the main study
# (plugged into the master_power.py formula)
# --------------------------------------------------------------------------- #
def mde_from_components(L: int, S: int, S_per: int, tau_LC: float, tau_SC: float,
                        sigma_d: float, alpha: float, power: float) -> float:
    k = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return k * math.sqrt(tau_LC ** 2 / L + tau_SC ** 2 / S
                         + sigma_d ** 2 / (L * min(S_per, S)))


def required_L(target: float, S: int, S_per: int, tau_LC: float, tau_SC: float,
               sigma_d: float, alpha: float, power: float) -> float:
    k = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    denom = (target / k) ** 2 - tau_SC ** 2 / S
    if denom <= 0:
        return float("inf")
    return (tau_LC ** 2 + sigma_d ** 2 / min(S_per, S)) / denom


def recompute_design(mushra_out: dict, cfg: dict) -> dict:
    """Plug the variance components measured in the pilot into the master_power.py
    MDE formula and recompute L."""
    an = cfg["analysis"]
    alpha = an["alpha_family"] / an["n_primary"]      # Holm worst case
    power = an["power_target"]
    S, S_per = an["S_main"], an["S_per"]

    comps = {name: r for name, r in mushra_out.get("contrasts", {}).items()
             if r.get("family") in ("primary", "primary_noninf", "aux_tost",
                                    "secondary")}
    if not comps and "joint_model" not in mushra_out:
        return {}
    per_contrast_med = {k: float(np.median([r[k] for r in comps.values()]))
                        for k in ("tau_LC", "tau_SC", "sigma_d")} if comps else {}
    jm = mushra_out.get("joint_model")
    if jm and jm.get("converged", True):
        # The joint model (which uses all S x C cells) has the smaller sampling
        # error, so prefer it.
        pooled = {"tau_LC": jm["tau_LC_diff"], "tau_SC": jm["tau_SC_diff"],
                  "sigma_d": jm["sigma_d_diff"], "source": "joint_model"}
    else:
        pooled = {**per_contrast_med, "source": "per_contrast_median"}

    print("\n## Measured variance components (main purpose of the pilot; difference scale = the master_power.py definition)\n")
    print("| source | tau_LC | tau_SC | sigma_d |")
    print("|---|---|---|---|")
    for name, r in comps.items():
        print(f"| contrast {name} | {r['tau_LC']:.2f} | {r['tau_SC']:.2f} "
              f"| {r['sigma_d']:.2f} |")
    if per_contrast_med:
        print(f"| per-contrast median | {per_contrast_med['tau_LC']:.2f} "
              f"| {per_contrast_med['tau_SC']:.2f} "
              f"| {per_contrast_med['sigma_d']:.2f} |")
    print(f"| **adopted ({pooled['source']})** | **{pooled['tau_LC']:.2f}** "
          f"| **{pooled['tau_SC']:.2f}** | **{pooled['sigma_d']:.2f}** |")
    scen = {"optimistic": (4.0, 17.0), "central": (6.0, 25.0),
            "pessimistic": (9.0, 37.0)}
    print("\nScenarios of master plan §1.2 (tau_LC, sigma_d): "
          + ", ".join(f"{k}=({a},{b})" for k, (a, b) in scen.items()))

    print(f"\n## Recomputed MDE for the main study (S={S}, S_per={S_per}, one-sided alpha={alpha:.4f}, "
          f"power {power})\n")
    print("| L | MDE (points) |")
    print("|---|---|")
    grid = {}
    for L in an["L_grid"]:
        m = mde_from_components(L, S, S_per, pooled["tau_LC"], pooled["tau_SC"],
                                pooled["sigma_d"], alpha, power)
        grid[L] = m
        print(f"| {L} | {m:.2f} |")
    print("\n| MDE target (points) | required L |")
    print("|---|---|")
    reqs = {}
    for tgt in an["mde_targets"]:
        Lr = required_L(tgt, S, S_per, pooled["tau_LC"], pooled["tau_SC"],
                        pooled["sigma_d"], alpha, power)
        reqs[tgt] = Lr
        print(f"| {tgt:.0f} | {'inf (song floor)' if math.isinf(Lr) else math.ceil(Lr)} |")
    k = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    s_floor = (k * pooled["tau_SC"]) ** 2
    print(f"\nsong floor: S > (k*tau_SC/MDE)^2 -> for an MDE of 5 points, S > "
          f"{s_floor / 25.0:.1f} songs")
    return {"per_contrast": {n: {k2: r[k2] for k2 in ("tau_LC", "tau_SC", "sigma_d")}
                             for n, r in comps.items()},
            "pooled": pooled, "alpha_one_sided": alpha, "power": power,
            "S": S, "S_per": S_per,
            "mde_by_L": {str(k2): v for k2, v in grid.items()},
            "required_L_by_target": {str(k2): v for k2, v in reqs.items()}}


# --------------------------------------------------------------------------- #
# Holm (primary family: H1 = non-inferiority on the ratings, H2/H3 = one-sided
# superiority on MUSHRA)
# --------------------------------------------------------------------------- #
# Design A (design_a_spec §3 / §5.4b): **the structure of 3 primary tests does not
# change.**
#   H1 = H1_topline_noninf  one-sided non-inferiority on the ratings
#                           (previously: the 2AFC binomial GLMM)
#   H2 = H2_hack            proposed (min) > pq_only   (fixed 
#   H3 = H3_sota            proposed (min) > fxnorm    (same)
# Even when 2AFC data exist, the primary family stays at 3 tests and 2AFC is placed
# as auxiliary (making the family 4 tests would change the Holm family and break the
# pre-registered alpha allocation).
def primary_family(mushra_out: dict, afc_out: dict, abx_out: dict,
                   cfg: dict) -> dict:
    ps: List[Tuple[str, float]] = []
    h1_name = noninf_contrast_name(cfg)
    cons = mushra_out.get("contrasts", {})
    if h1_name:
        r1 = cons.get(h1_name)
        if r1 and np.isfinite(r1.get("p_noninf", float("nan"))):
            ps.append((h1_name, float(r1["p_noninf"])))
        else:
            print(f"\n[warn] cannot produce the non-inferiority test for H1 ({h1_name}) "
                  f"(not enough same-screen data for proposed / professional). "
                  f"The primary tests fall short of 3, so check the Holm family")
    for name in ("H2_hack", "H3_sota"):
        r = cons.get(name)
        if r:
            ps.append((name, r["p_greater"]))
    if not ps:
        return {}
    adj = holm([p for _, p in ps])
    fam = {name: {"p_raw": p, "p_holm": a}
           for (name, p), a in zip(ps, adj)}
    if h1_name in fam:
        r1 = cons[h1_name]
        fam[h1_name]["test"] = "one-sided non-inferiority (rating)"
        fam[h1_name]["noninf_margin"] = r1["noninf_margin"]
        fam[h1_name]["noninferior"] = r1["noninferior"]
    if abx_out.get("demote_H2") and "H2_hack" in fam:
        fam["H2_hack"]["note"] = ("ABX demotion condition met: H2 is demoted out of "
                                  "primary (reported as metric oversensitivity)")
    n_exp = int(cfg["analysis"]["n_primary"])
    print(f"\n## primary family (Holm, {len(fam)}/{n_exp} tests)\n")
    print("| hypothesis | p raw | p Holm | verdict (alpha=0.05) |")
    print("|---|---|---|---|")
    for name, r in fam.items():
        sig = "significant" if r["p_holm"] < cfg["analysis"]["alpha_family"] else "n.s."
        note = " " + r.get("note", "")
        if name == h1_name:
            note += (f" (one-sided test with a non-inferiority margin of {r['noninf_margin']:.1f} points. "
                     f"significant = not worse than the professional by {r['noninf_margin']:.1f} points or more)")
        print(f"| {name} | {r['p_raw']:.3g} | {r['p_holm']:.3g} | {sig}{note} |")
    if len(fam) != n_exp:
        print(f"  ** only {len(fam)} primary tests are present (pre-registration says {n_exp}). "
              f"The Holm correction is being applied over a different family than pre-registered. **")
    if afc_out.get("applicable"):
        print("  2AFC is listed separately as **auxiliary** (not part of the primary family). "
              "design_a_spec §3")
    return fam


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def flow_only(records_path: str, cfg: dict, manifest: Optional[str]) -> dict:
    """Print only the participant-flow counts (no ratings, contrasts or tests at all).

    Used for the report that decides, after the wave-1 deadline, whether to run a
    second wave. Changing the sample size after looking at between-condition
    differences or p values is optional stopping, and pushes the false-positive rate
    above the nominal alpha. To prevent that structurally, this mode restricts the
    decision material to "how many participants remain".
    See internal notes 0.5.4.2.
    """
    raw = read_raw_records(records_path)
    blind, trial2song = load_blind_map(manifest)
    data = normalise(raw, cfg, blind, trial2song)
    n_start = len(all_listeners(data))
    if not n_start:
        raise SystemExit("Cannot read any records. Check the field map in --config")

    kept, table = exclusion_pipeline(data, cfg)

    # Breakdown by completion code (HP failure, unsupported device and quality
    # violations are counts independent of the ratings)
    verdicts: Dict[str, int] = {}
    sess = data.get("session")
    if sess is not None and len(sess) and "verdict" in getattr(sess, "columns", []):
        for v in sess["verdict"].dropna():
            verdicts[str(v)] = verdicts.get(str(v), 0) + 1

    # "Number of people who answered the questionnaire" = the screening
    # questionnaire (screen_q). The post-session survey (questionnaire) is only
    # submitted by people who made it to the end of the main session, so counting
    # that here would underestimate the number of entrants.
    sq = data.get("screen_q")
    quest = data.get("questionnaire")
    n_quest = len(sq) if sq is not None and len(sq) else (
        len(quest) if quest is not None else 0)

    print("# pilot_analysis --flow-only  (participant-flow counts only; no ratings are printed)")
    print(f"#   input: {records_path}")
    print()
    print(f"  Started the study                : {n_start}")
    print(f"  Answered the questionnaire       : {n_quest}")
    for k in sorted(verdicts):
        print(f"  completion code {k:<18}: {verdicts[k]}")
    print()
    print_exclusion_table(table, n_start)
    print_eligibility_audit(eligibility_audit(data))
    print()
    print(f"  ** Participants entering the analysis (= N of the main analysis) : {len(kept)} **")
    print()
    # Single-study design : there is no longer a "second wave". Adding
    # participants is now a decision about adding places to the same study
    # (internal notes §3.3).
    print("  Decide on adding participants (extra places on the same study) from this number")
    print("  and the Prolific-side eligibility rate, participation rate and spend alone. Per-condition")
    print("  mean ratings, between-condition differences, p values and effect sizes come after that decision is recorded in internal notes.")
    return {"mode": "flow_only", "n_started": n_start,
            "n_questionnaire": n_quest, "verdicts": verdicts,
            "n_analysed": len(kept), "exclusion_table": table}


def analyse(records_path: str, cfg: dict, manifest: Optional[str],
            out_path: Optional[str], full_lmm: bool = False) -> dict:
    raw = read_raw_records(records_path)
    blind, trial2song = load_blind_map(manifest)
    data = normalise(raw, cfg, blind, trial2song)
    n_start = len(all_listeners(data))
    print(f"# pilot_analysis: {records_path}")
    print(f"raw records {data['_n_raw']} (dedupe -{data['_n_dupe']}), "
          f"listeners seen {n_start}, mushra rows {len(data['mushra'])}, "
          f"afc {len(data['afc'])}, abx {len(data['abx'])}")
    if not n_start:
        raise SystemExit("Cannot read any records. Check the field map in --config")

    kept, table = exclusion_pipeline(data, cfg)
    print_exclusion_table(table, n_start)
    audit = eligibility_audit(data)
    print_eligibility_audit(audit)
    if not kept:
        raise SystemExit("The exclusion pipeline excluded everyone")

    mus = data["mushra"][data["mushra"].listener.isin(kept)]
    afc = data["afc"][data["afc"].listener.isin(kept)] if len(data["afc"]) else \
        data["afc"]
    abx = data["abx"][data["abx"].listener.isin(kept)] if len(data["abx"]) else \
        data["abx"]

    mushra_out = mushra_analysis(mus, cfg) if len(mus) else {}
    if full_lmm and len(mus):
        full_lmm_check(mus)
    # 2AFC / ABX come out as 0 rows under design A. **Always call the functions**:
    # letting them return "not applicable" internally makes it visible from the
    # output whether the block is switched off or actually broken.
    afc_out = afc_analysis(afc, cfg)
    abx_out = abx_analysis(abx, cfg)
    h1 = noninferiority_report(mushra_out, cfg) if mushra_out else {}
    fam = primary_family(mushra_out, afc_out, abx_out, cfg)
    design = recompute_design(mushra_out, cfg) if mushra_out else {}

    results = {
        "records_path": records_path,
        "n_listeners_seen": n_start,
        "n_listeners_kept": len(kept),
        "kept": sorted(kept),
        "exclusion_table": table,
        "eligibility_audit": audit,
        "mushra": mushra_out,
        "h1_noninferiority": h1,
        "afc": afc_out,
        "abx": abx_out,
        "primary_family_holm": fam,
        "design_recompute": design,
    }
    if out_path:
        if os.path.isdir(out_path):          # if --out is a directory, write inside it
            out_path = os.path.join(out_path, "pilot_analysis.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(_jsonable(results), fh, indent=1, ensure_ascii=False)
        print(f"\nwrote {out_path}")
    return results


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


# --------------------------------------------------------------------------- #
# self-test: verification on synthetic data with known parameters
# --------------------------------------------------------------------------- #
SELFTEST_TRUTH = {
    # 8 conditions . sb_only is "hacking SB at the expense of PQ" and its
    # objective loss (ΔPQ −0.21) is larger than pq_only's, so its true value is set
    # lower than pq_only as well.
    # megami is objectively worse than FxNorm (ΔSB +0.46 vs +0.28), hence below fxnorm.
    "cond_means": {"professional": 72.0, "proposed": 70.0, "pq_only": 62.0,
                   "sb_only": 60.0, "mean_reward": 66.0, "fxnorm": 64.0,
                   "megami": 58.0, "random_dry": 30.0},
    "tau_LC": 6.0,     # difference scale (per-condition this is 6/sqrt2)
    "tau_SC": 8.0,
    "sigma_d": 16.0,   # difference scale (per-rating this is 16/sqrt2)
    "sd_listener": 8.0, "sd_song": 5.0,
    "afc_flip_repeat": 0.15,
    "abx_accuracy": 0.58,
}


def generate_synthetic(seed: int, out_path: str, cfg: dict) -> dict:
    """40 listeners / 3 x S_per songs / 8 conditions. Synthetic records with missing
    data and bad respondents mixed in.

    The song count is 3 rotations x S_per (27 songs when S_per=9 as in design A), so
    that every rotation has the same number of screens.
    """
    rng = np.random.default_rng(seed)
    T = SELFTEST_TRUTH
    conds = cfg["conditions"]
    anchor = cfg["anchor_condition"]
    S_per = int(cfg["analysis"]["S_per"])
    songs = [f"song{q:02d}" for q in range(3 * S_per)]
    rotations = [songs[i * S_per:(i + 1) * S_per] for i in range(3)]

    planted = {
        "1_stage1_hp_ap": [f"S1FAIL{i}" for i in range(5)],
        "2_incomplete": [f"DROP{i}" for i in range(2)],
        "3_screen_checks": ["RAND0", "RAND1", "FLAT0"],
        "4_speeder": ["SPEED0"],
        "5_afc_repeat": ["REV0"],
        "6_iqr_outlier": ["LOW0"],
    }
    good = [f"P{i:03d}" for i in range(27)]
    order = good + sum(planted.values(), [])

    g_song = {s: rng.normal(0, T["sd_song"]) for s in songs}
    q_sc = {(s, c): rng.normal(0, T["tau_SC"] / math.sqrt(2))
            for s in songs for c in conds}
    d_pp_song = {s: (q_sc[(s, "professional")] - q_sc[(s, "proposed")])
                 for s in songs}   # share the 2AFC song effect with MUSHRA

    lines: List[str] = []

    def emit(rec: dict) -> None:
        lines.append(json.dumps(rec))

    for i, pid in enumerate(order):
        rot = rotations[i % 3]
        kind = "good"
        for stage, ids in planted.items():
            if pid in ids:
                kind = stage
        # ---- stage1 ----
        if kind == "1_stage1_hp_ap":
            emit({"type": "stage1", "pid": pid,
                  "hp_correct": int(rng.integers(1, 4)), "ap_correct": 6})
            continue                       # does not proceed to Stage2
        emit({"type": "stage1", "pid": pid, "hp_correct": 6,
              "ap_correct": int(rng.integers(5, 7))})

        b_l = rng.normal(0, T["sd_listener"])
        if kind == "6_iqr_outlier":
            b_l = -45.0                    # listener with a compressed scale
        h_lc = {c: rng.normal(0, T["tau_LC"] / math.sqrt(2)) for c in conds}
        # The experiment assumes the anchor (random_dry, degraded by up to 25 dB) is
        # last for every listener. Allowing a listener-side preference for dry would
        # make exclusion criterion 3(i) fire correctly and drop a good respondent,
        # which would leave the self-test's false-positive verdict up to the RNG.
        h_lc[anchor] = 0.0
        sigma_rating = T["sigma_d"] / math.sqrt(2)

        # ---- MUSHRA screens ----
        song_order = list(rng.permutation(rot))
        n_screens = 3 if kind == "2_incomplete" else len(song_order)
        for scr, song in enumerate(song_order[:n_screens]):
            if kind == "good" and rng.random() < 0.03:
                continue                   # dropped incremental POST (missing data)
            ratings = {}
            for c in conds:
                if kind == "3_screen_checks" and pid.startswith("RAND"):
                    v = rng.uniform(0, 100)
                elif kind == "3_screen_checks" and pid.startswith("FLAT"):
                    v = 62.0
                else:
                    v = (T["cond_means"][c] + b_l + g_song[song] + h_lc[c]
                         + q_sc[(song, c)] + rng.normal(0, sigma_rating))
                ratings[c] = float(np.clip(v, 0, 100))
            dur = 18.0 if kind == "4_speeder" else float(rng.normal(110, 15))
            emit({"type": "mushra", "pid": pid, "screen_index": scr, "song": song,
                  "ratings": ratings, "duration_sec": max(dur, 10.0),
                  "is_training": False})
        if kind == "2_incomplete":
            continue                       # no session record = incomplete

        # ---- 2AFC (proposed vs professional): first presentation of S_per songs +
        #      repeats of the first 4 songs ----
        #      (with S_per=9 as in design A: 9 first + 4 repeat = 13 rows. The old
        #       comment's "8 first" dated from when S_per was 8, and was corrected on
        #       
        afc_songs = rot
        latent_first: Dict[str, float] = {}
        for j, song in enumerate(afc_songs):
            lat = ((T["cond_means"]["professional"] - T["cond_means"]["proposed"])
                   + (h_lc["professional"] - h_lc["proposed"])
                   + d_pp_song[song] + rng.normal(0, T["sigma_d"]))
            latent_first[song] = lat
            if pid.startswith("RAND") or kind == "5_afc_repeat":
                choice = str(rng.choice(["proposed", "professional"]))
            else:
                choice = "professional" if lat > 0 else "proposed"
            latent_first[song + "__choice"] = choice
            emit({"type": "afc", "pid": pid, "pair_index": j, "song": song,
                  "choice": choice, "is_repeat": False,
                  "duration_sec": 6.0 if kind == "4_speeder"
                  else float(rng.normal(26, 5))})
        for j, song in enumerate(afc_songs[:4]):
            first_choice = latent_first[song + "__choice"]
            if kind == "5_afc_repeat":     # always flips on the repeat -> agreement 0
                choice = ("proposed" if first_choice == "professional"
                          else "professional")
            elif rng.random() < T["afc_flip_repeat"]:
                choice = ("proposed" if first_choice == "professional"
                          else "professional")
            else:
                choice = first_choice
            # Lay out the pair_index of the repeated pairs after the first
            # presentations (starting at len(afc_songs)). A hard-coded 8 would clash
            # with a first presentation when S_per=9 and be swallowed by the dedupe.
            emit({"type": "afc", "pid": pid, "pair_index": len(afc_songs) + j,
                  "song": song,
                  "choice": choice, "is_repeat": True, "repeat_of": j,
                  "duration_sec": 6.0 if kind == "4_speeder"
                  else float(rng.normal(26, 5))})

        # ---- ABX, 8 trials ----
        for j in range(8):
            emit({"type": "abx", "pid": pid, "trial_index": j,
                  "correct": bool(rng.random() < T["abx_accuracy"])})

        # ---- session ----
        total = (280.0 if kind == "4_speeder"
                 else float(rng.normal(1450, 120)))
        emit({"type": "session", "pid": pid, "completed": True,
              "total_sec": total,
              "order_group": "afc_first" if i % 2 == 0 else "mushra_first"})
        emit({"type": "questionnaire", "pid": pid, "cue": "balance"})

    # Resends (dedupe test): duplicate the first 5 records
    lines += lines[:5]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"planted": planted, "good": good, "n_records": len(lines),
            "true_pref_professional": None}


def _wrap_spa_v1(rec_path: str, outdir: str, cfg: dict) -> Tuple[str, str]:
    """Re-wrap the flat synthetic records into the real pilot_v2 SPA schema
    (envelope + payload + blind file names) and write them out together with a
    manifest in build_stimuli_v2 format.
    This is the input that exercises the real-data path of --preset spa_v1 +
    --manifest as-is."""
    import hashlib

    def bid(song: str, cond: str) -> str:
        return hashlib.sha256(f"{song}|{cond}".encode()).hexdigest()[:16]

    def tid(song: str) -> str:
        return hashlib.sha256(f"{song}|__trial__".encode()).hexdigest()[:16]

    flat = [json.loads(ln) for ln in open(rec_path) if ln.strip()]
    songs = sorted({r["song"] for r in flat
                    if r.get("type") in ("mushra", "afc") and r.get("song")})
    manifest = {"set_name": "selftest_spa", "trials": [
        {"trial_id": tid(s), "song_id": s,
         "clips": {c: {"file": f"audio/{bid(s, c)}.opus", "blind_id": bid(s, c)}
                   for c in cfg["conditions"]}}
        for s in songs]}
    man_path = os.path.join(outdir, "manifest_spa.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh)

    lines: List[str] = []
    for r in flat:
        t, pid = r.get("type"), r.get("pid")
        env: Dict[str, Any] = {"v": 1, "set": "selftest_spa", "app": "wrap-1.0",
                               "pid": pid, "stage": 1 if t == "stage1" else 2,
                               "demo": False}
        if t == "stage1":
            env["type"] = "hpscreen"
            env["payload"] = {"score": {"hp_correct": r["hp_correct"],
                                        "ap_correct": r["ap_correct"],
                                        "hp_n": 6, "ap_n": 6}}
        elif t == "mushra":
            s = r["song"]
            env["type"] = "mushra_trial"
            env["payload"] = {
                "trial_index": r["screen_index"], "trial_id": tid(s),
                "ratings": {f"audio/{bid(s, c)}.opus": v
                            for c, v in r["ratings"].items()},
                "duration_sec": r["duration_sec"],
                "is_training": bool(r.get("is_training", False))}
        elif t == "afc":
            s = r["song"]
            env["type"] = "afc_trial"
            env["payload"] = {
                "afc_index": r["pair_index"], "trial_id": tid(s),
                "is_repeat": bool(r.get("is_repeat", False)),
                "first_file": f"audio/{bid(s, 'proposed')}.opus",
                "second_file": f"audio/{bid(s, 'professional')}.opus",
                "choice_position": "n/a",
                "choice_file": f"audio/{bid(s, r['choice'])}.opus",
                "duration_sec": r["duration_sec"]}
        elif t == "abx":
            env["type"] = "abx_trial"
            env["payload"] = {"abx_index": r["trial_index"],
                              "correct": bool(r["correct"])}
        elif t == "session":
            env["type"] = "session_end"
            env["afc_first"] = r.get("order_group") == "afc_first"
            env["payload"] = {
                "code_logical": "MAIN_COMPLETE" if r.get("completed")
                                else "RETURNED",
                "total_sec": r["total_sec"], "pending_uploads": 0}
        elif t == "questionnaire":
            env["type"] = "post_questionnaire"
            env["payload"] = {k: v for k, v in r.items()
                              if k not in ("type", "pid")}
        else:
            continue
        lines.append(json.dumps(env))
    spa_path = os.path.join(outdir, "records_spa_v1.jsonl")
    with open(spa_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return spa_path, man_path


def self_test(seed: int) -> int:
    """Verify (a) exclusion, (b) variance components, (c) contrast recovery,
    (d) the field map and (e) the SPA v1 real-schema round-trip on synthetic data
    with known parameters. The verdicts are defined so that they deterministically
    PASS with the default seed (20260813).
    With other seeds, borderline FAILs can occur because of the probabilistically
    planted bad respondents (e.g. a ~3.5% chance that a uniform-random rater slips
    past the basic check) or the luck of an honest respondent (a ~0.1% chance of
    placing dry above the median on 2/8 screens). That is a property of the exclusion
    rules, not an implementation bug (the rules are pre-registered in master plan
    §7.2)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    T = SELFTEST_TRUTH
    # The output directory defaults to a **fixed shared directory**. The self-test
    # writes synthetic records and reads them back, so **running two or more of them
    # concurrently in the same directory makes them stamp on each other's files and
    # can produce false FAILs / false PASSes**
    # (measured with 3 concurrent runs on . It is not merely slower).
    # The rule is "only one at a time". If you really must run them in parallel, give
    # **a separate output directory per process** via the PILOT_SELFTEST_OUTDIR
    # environment variable.
    outdir = (os.environ.get("PILOT_SELFTEST_OUTDIR")
              or os.path.join(ROOT, "outputs", "analysis", "pilot_selftest"))
    rec_path = os.path.join(outdir, "records.jsonl")
    meta = generate_synthetic(seed, rec_path, cfg)
    print(f"[self-test] wrote {meta['n_records']} records "
          f"({len(meta['good'])} good / {sum(len(v) for v in meta['planted'].values())} "
          f"planted-bad listeners) to {rec_path}\n")

    res = analyse(rec_path, cfg, manifest=None,
                  out_path=os.path.join(outdir, "pilot_analysis_selftest.json"))

    checks: List[Tuple[str, bool, str]] = []

    # (a) Exclusion pipeline: the planted bad respondents must be caught at the
    # intended stage.
    # Zero false exclusions of good respondents are required for stages 1-5. Stage 6
    # (IQR / Tukey fence) is a rule with a ~0.7%/listener false-positive rate even on
    # normal data, so up to 2 are allowed (a property of the pre-registered rule
    # itself, not an implementation bug).
    table = {row["stage"]: row for row in res["exclusion_table"]}
    good = set(meta["good"])
    for stage, ids in meta["planted"].items():
        got = set(table[stage]["excluded"])
        ok = set(ids) <= got
        checks.append((f"(a) {stage} catches {ids}", ok,
                       f"excluded at stage = {sorted(got)}"))
    fp_15 = sorted(good & set().union(
        *[set(table[s]["excluded"]) for s in list(table)[:5]]))
    fp_iqr = sorted(good & set(table["6_iqr_outlier"]["excluded"]))
    checks.append(("(a) no false exclusion of good respondents in stages 1-5", len(fp_15) == 0,
                   f"false positives = {fp_15}"))
    checks.append(("(a) good respondents excluded by stage6 IQR <= 2 (known Tukey fence false positives)",
                   len(fp_iqr) <= 2, f"iqr false positives = {fp_iqr}"))
    exp_kept = len(good) - len(fp_iqr) - len(fp_15)
    checks.append((f"(a) kept = {exp_kept}",
                   res["n_listeners_kept"] == exp_kept,
                   f"kept = {res['n_listeners_kept']}"))

    # (b) Variance components within +-30% of the true values
    pooled = res.get("design_recompute", {}).get("pooled", {})
    for key, true in (("tau_LC", T["tau_LC"]), ("tau_SC", T["tau_SC"]),
                      ("sigma_d", T["sigma_d"])):
        est = pooled.get(key, float("nan"))
        ok = np.isfinite(est) and abs(est - true) / true <= 0.30
        checks.append((f"(b) {key} within +-30% of {true}", bool(ok),
                       f"est = {est:.2f}"))

    # (c) Contrast estimates recover the true values
    tm = T["cond_means"]
    # : the primary arm is proposed (min). Keep this in step with the
    # contrasts above.
    for name, a, b in (("H2_hack", "proposed", "pq_only"),
                       ("H3_sota", "proposed", "fxnorm"),
                       # The old H1_topline_aux (two-sided TOST) was replaced by the
                       # non-inferiority test of design A. The quantity being
                       # recovered (mu) is the same, so the check stays.
                       ("H1_topline_noninf", "proposed", "professional"),
                       ("S1_minvsmean", "proposed", "mean_reward")):
        r = res["mushra"]["contrasts"].get(name)
        if r is None:
            checks.append((f"(c) contrast {name} present", False, "missing"))
            continue
        true_mu = tm[a] - tm[b]
        tol = max(2.5 * r["se"], 2.0)
        ok = abs(r["mu"] - true_mu) <= tol
        checks.append((f"(c) {name} mu recovers {true_mu:+.1f}", bool(ok),
                       f"mu = {r['mu']:+.2f} +- {r['se']:.2f} (tol {tol:.1f})"))
    r = res["mushra"]["contrasts"]["H2_hack"]
    checks.append(("(c) H2 (true +8) one-sided p < 0.05",
                   r["p_greater"] < 0.05, f"p = {r['p_greater']:.3g}"))
    r = res["mushra"]["contrasts"]["check_dry"]
    checks.append(("(c) check_dry (true +40) has the right sign", r["mu"] > 20,
                   f"mu = {r['mu']:+.1f}"))

    # (c2) H1 = the one-sided non-inferiority test on the ratings
    # (design A / design_a_spec §3).
    # The true value is proposed - professional = -2.0 points and the default margin
    # is Delta = 10 points, so "not worse by 10 points or more" should hold. Also
    # check that it is one-sided rather than a TOST.
    h1n = noninf_contrast_name(cfg)
    r1 = res["mushra"]["contrasts"].get(h1n or "")
    checks.append(("(c2) H1 exists as a non-inferiority test (side=noninf)",
                   bool(r1) and r1.get("side") == "noninf"
                   and "p_noninf" in r1 and "tost_margin" not in r1,
                   f"name={h1n}, keys={sorted(r1)[:6] if r1 else None}"))
    if r1:
        checks.append((f"(c2) H1 non-inferiority: at Delta={r1['noninf_margin']:.0f}pt, "
                       f"one-sided p < 0.05 and noninferior",
                       r1["p_noninf"] < 0.05 and r1["noninferior"] is True,
                       f"p = {r1['p_noninf']:.3g}, one-sided 95% lower bound = "
                       f"{r1['ci95_lower_one_sided']:+.2f}"))
    h1 = res["h1_noninferiority"]
    checks.append(("(c2) the margin comes from the config and carries written grounds",
                   h1.get("margin_points") == cfg["analysis"]["noninf_margin_points"]
                   and len(h1.get("margin_source", "")) > 20,
                   f"Delta = {h1.get('margin_points')}"))
    checks.append((f"(c2) the default Delta ({h1['margin_points']:.0f}pt) is at least the planned MDE "
                   f"{h1['planned_mde_points']}pt -> power at a true difference of 0 is "
                   f">= {cfg['analysis']['power_target']:.0%}",
                   not h1["margin_below_planned_mde"]
                   and h1["power_by_true_gap"]["0.0"]
                   >= cfg["analysis"]["power_target"],
                   f"power(gap=0) = {h1['power_by_true_gap']['0.0']:.3f}"))
    # **Do not hide a Delta that cannot be achieved.** Passing a Delta below 6.27
    # points must be reported as "power does not reach 80%" (design_a_spec §3).
    h1_small = noninferiority_report(
        res["mushra"], deep_merge(cfg, {"analysis": {"noninf_margin_points": 5.0}}))
    checks.append(("(c2) Delta=5pt (below the planned MDE) is flagged as unreachable and the power is printed",
                   h1_small["margin_below_planned_mde"] is True
                   and h1_small["power_by_true_gap"]["0.0"]
                   < cfg["analysis"]["power_target"],
                   f"power(gap=0) = {h1_small['power_by_true_gap']['0.0']:.3f}"))
    # 2AFC: true preference rate P(professional) = Phi(2 / sqrt(tauLC^2+tauSC^2+sigma^2))
    p_true_prof = float(stats.norm.cdf(
        (tm["professional"] - tm["proposed"])
        / math.sqrt(T["tau_LC"] ** 2 + T["tau_SC"] ** 2 + T["sigma_d"] ** 2)))
    p_true_prop = 1 - p_true_prof
    est = res["afc"]["pref_rate_mean"]
    checks.append((f"(c) 2AFC pref({res['afc']['target']}) recovers "
                   f"{p_true_prop:.3f}", abs(est - p_true_prop) <= 0.10,
                   f"est = {est:.3f}"))
    # ABX: true value 0.58. The realised accuracy must land within +-0.06 of the true
    # value, and the demotion verdict must be self-consistent with the rule (group
    # mean <= 0.60); the realised value can straddle 0.60 by chance.
    grand = res["abx"].get("grand_accuracy", float("nan"))
    thr = res["abx"].get("demote_threshold", 0.60)
    checks.append(("(c) ABX realised accuracy within 0.58 +-0.06 of the true value",
                   abs(grand - T["abx_accuracy"]) <= 0.06,
                   f"grand acc = {grand:.3f}"))
    checks.append(("(c) ABX demotion verdict consistent with the rule (grand <= 0.60)",
                   bool(res["abx"].get("demote_H2")) == bool(grand <= thr),
                   f"demote = {res['abx'].get('demote_H2')}"))

    # (d) Swapping out the field map works (schema-rename round-trip)
    alt_fields = {"type": "kind", "pid": "participant", "song": "track",
                  "ratings": "scores", "screen_index": "page",
                  "duration": "elapsed", "choice": "picked",
                  "pair_index": "pair", "trial_index": "idx",
                  "hp_correct": "hp", "ap_correct": "ap",
                  "completed": "done", "total_sec": "sess_sec"}
        # the rest keep their defaults
    rename = {DEFAULT_CONFIG["fields"][k]: v for k, v in alt_fields.items()}
    alt_lines = []
    for ln in open(rec_path):
        r = json.loads(ln)
        alt_lines.append(json.dumps({rename.get(k, k): v for k, v in r.items()}))
    alt_path = os.path.join(outdir, "records_altschema.jsonl")
    open(alt_path, "w").write("\n".join(alt_lines) + "\n")
    alt_cfg = deep_merge(cfg, {"fields": alt_fields})
    alt_data = normalise(read_raw_records(alt_path), alt_cfg, {})
    base_data = normalise(read_raw_records(rec_path), cfg, {})
    ok = all(len(alt_data[k]) == len(base_data[k])
             for k in ("mushra", "afc", "abx", "stage1", "session"))
    checks.append(("(d) identical row counts after swapping the field map", ok,
                   f"alt rows = {[len(alt_data[k]) for k in ('mushra','afc','abx','stage1','session')]}"))

    # (e) Round-trip through the real SPA v1 schema: envelope + payload nesting +
    # blind file names + manifest de-blinding + completion code. Cross-checks the
    # real-data path of --preset spa_v1 against the flat path.
    print("\n" + "-" * 74)
    print("[self-test] (e) SPA v1 envelope round-trip")
    print("-" * 74)
    spa_path, man_path = _wrap_spa_v1(rec_path, outdir, cfg)
    spa_cfg = deep_merge(cfg, SPA_V1_OVERRIDES)
    res_spa = analyse(spa_path, spa_cfg, manifest=man_path,
                      out_path=os.path.join(outdir,
                                            "pilot_analysis_selftest_spa.json"))
    checks.append(("(e) SPA v1: kept listeners match",
                   res_spa["kept"] == res["kept"],
                   f"kept flat={res['n_listeners_kept']} "
                   f"spa={res_spa['n_listeners_kept']}"))
    mu_f = res["mushra"]["contrasts"]["H2_hack"]["mu"]
    mu_s = res_spa["mushra"]["contrasts"]["H2_hack"]["mu"]
    checks.append(("(e) SPA v1: H2 contrast mu matches (<=1e-6)",
                   abs(mu_f - mu_s) <= 1e-6,
                   f"flat {mu_f:+.6f} vs spa {mu_s:+.6f}"))
    pr_f, pr_s = res["afc"]["pref_rate_mean"], res_spa["afc"]["pref_rate_mean"]
    ab_f = res["abx"]["grand_accuracy"]
    ab_s = res_spa["abx"]["grand_accuracy"]
    checks.append(("(e) SPA v1: 2AFC preference rate / ABX accuracy match",
                   abs(pr_f - pr_s) <= 1e-12 and abs(ab_f - ab_s) <= 1e-12,
                   f"pref {pr_f:.4f}/{pr_s:.4f}, abx {ab_f:.4f}/{ab_s:.4f}"))

    # (f) Re-adjudication of the screening questionnaire at analysis time
    #     (single-study design, .
    #     Can the analysis reliably drop an ineligible participant who got a
    #     provisional pass because the server was unreachable? Tested on an
    #     independent, minimal record set so the main data is not polluted.
    #     : the questionnaire became 8 questions and q_device joined the
    #     eligibility criteria.
    #     **If a required field is left out, ok_ans itself becomes ineligible with
    #     missing_data and this check falsely reports "exclusion in the analysis is
    #     broken".**
    #     The authoritative list of required fields is REQUIRED_FOR_JUDGEMENT in
    #     screen_eligibility.py.
    ok_ans = {"q_device": "over_ear", "q_hearing": "normal",
              "q_prod": "y5_10", "q_prod_years": 7, "q_know1": "ratio",
              "q_know2": "send", "q_know3": "lufs", "q_prev": "none"}
    # Ineligible side: fails on two counts, the bogus option (harmonic_gate) and
    # declaring speakers.
    ng_ans = dict(ok_ans, q_device="speakers", q_prod="lt1", q_prod_years=0,
                  q_know1="harmonic_gate")
    def _env(pid, typ, payload, **kw):
        e = {"type": typ, "pid": pid, "payload": payload, "demo": False}
        e.update(kw)
        return e
    elig_raw: List[dict] = []
    for pid, ans, prov in (("EOK", ok_ans, False), ("ENG", ng_ans, True)):
        elig_raw.append(_env(pid, "questionnaire", ans))
        elig_raw.append(_env(pid, "elig_score",
                             {"eligible": True, "reasons": [],
                              "provisional": prov,
                              "note": "server_unavailable" if prov else None}))
        elig_raw.append(_env(pid, "hpscreen",
                             {"score": {"hp_correct": 6, "ap_correct": 6}}))
        elig_raw.append(_env(pid, "session_end",
                             {"code_logical": "MAIN_COMPLETE", "total_sec": 3000},
                             afc_first=True))
    ed = normalise(elig_raw, deep_merge(cfg, SPA_V1_OVERRIDES), {})
    ekept, etable = exclusion_pipeline(ed, deep_merge(cfg, SPA_V1_OVERRIDES))
    eaudit = eligibility_audit(ed)
    st1a = next((r for r in etable if r["stage"] == "1a_ineligible"), None)
    checks.append(("(f) the analysis excludes an ineligible participant on a provisional pass",
                   st1a is not None and st1a["excluded"] == ["ENG"]
                   and "EOK" in ekept and "ENG" not in ekept,
                   f"excluded={st1a['excluded'] if st1a else None}, "
                   f"kept={sorted(ekept)}"))
    checks.append(("(f) the audit table counts provisional passes and disagreements",
                   eaudit["provisional_pass"] == ["ENG"]
                   and eaudit["client_passed_but_ineligible"] == ["ENG"]
                   and eaudit["n_judged"] == 2,
                   f"audit={ {k: eaudit[k] for k in ('n_judged', 'provisional_pass', 'client_passed_but_ineligible')} }"))
    # : the flattened fields were brought in line with the new 8 questions.
    # The column for the dropped q_mixes is gone, and instead q_device (device),
    # which is needed as a covariate, is checked.
    checks.append(("(f) an eligible participant's questionnaire is flattened for the Methods table",
                   list(ed["screen_q"][ed["screen_q"].listener == "EOK"]
                        ["exp_band"]) == ["y5_10"]
                   and list(ed["screen_q"][ed["screen_q"].listener == "EOK"]
                            ["device"]) == ["over_ear"],
                   f"rows={len(ed['screen_q'])}"))

    # (g) **The production configuration of design A**: input with not a single 2AFC
    #     / ABX row. design_a_spec §5.4b / §6. If this check goes missing, nobody
    #     would notice that the primary test for H1 disappeared on production data.
    print("\n" + "-" * 74)
    print("[self-test] (g) design A (ratings only; 0 rows of 2AFC and ABX)")
    print("-" * 74)
    da_path = os.path.join(outdir, "records_design_a.jsonl")
    n_dropped = 0
    with open(da_path, "w") as fh:
        for ln in open(rec_path):
            if not ln.strip():
                continue
            if json.loads(ln).get("type") in ("afc", "abx"):
                n_dropped += 1
                continue
            fh.write(ln)
    res_da = analyse(da_path, cfg, manifest=None,
                     out_path=os.path.join(outdir,
                                           "pilot_analysis_selftest_designA.json"))
    fam_da = res_da["primary_family_holm"]
    checks.append((f"(g) design A: the analysis runs to completion after dropping {n_dropped} 2AFC/ABX rows",
                   bool(res_da["mushra"]) and res_da["n_listeners_kept"] > 0,
                   f"kept = {res_da['n_listeners_kept']}"))
    checks.append(("(g) design A: 2AFC is \"not applicable\" (does not crash, not an empty dict)",
                   res_da["afc"].get("applicable") is False
                   and "reason" in res_da["afc"],
                   f"afc = {res_da['afc'].get('reason')}"))
    checks.append(("(g) design A: ABX is \"not applicable\" too, so no H2 demotion decision happens",
                   res_da["abx"].get("applicable") is False
                   and not res_da["abx"].get("demote_H2"),
                   f"abx = {res_da['abx'].get('reason')}"))
    checks.append((f"(g) design A: the primary family is the 3 tests {h1n} / H2_hack / H3_sota "
                   f"(the Holm family does not change)",
                   sorted(fam_da) == sorted([h1n, "H2_hack", "H3_sota"])
                   and len(fam_da) == cfg["analysis"]["n_primary"],
                   f"family = {sorted(fam_da)}"))
    checks.append(("(g) design A: H1 enters the family as a non-inferiority test",
                   fam_da.get(h1n, {}).get("test")
                   == "one-sided non-inferiority (rating)"
                   and np.isfinite(fam_da.get(h1n, {}).get("p_raw", float("nan"))),
                   f"H1 = {fam_da.get(h1n)}"))
    r1da = res_da["mushra"]["contrasts"][h1n]
    checks.append(("(g) design A: mu of H1 recovers the true value -2.0",
                   abs(r1da["mu"] - (tm["proposed"] - tm["professional"]))
                   <= max(2.5 * r1da["se"], 2.0),
                   f"mu = {r1da['mu']:+.2f} +- {r1da['se']:.2f}"))
    checks.append(("(g) design A: the 2AFC exclusion criterion (repeat agreement) drops nobody",
                   next(row["n_excluded"] for row in res_da["exclusion_table"]
                        if row["stage"] == "5_afc_repeat") == 0
                   and set(res["kept"]) <= set(res_da["kept"]),
                   f"kept flat={len(res['kept'])} designA={len(res_da['kept'])}"))
    # (g2) **Old records containing 2AFC must not drag design A finishers down with
    #      them.** Raised in the  integration check: because the relaxation
    #      of exclusion criterion 2 was decided over the whole dataset, a single
    #      participant's worth of afc rows reinstated need_afc=12 and every design A
    #      finisher who dropped session_end was excluded as "incomplete".
    #      Here we (a) drop the session record of the finisher P000 and (b) mix in
    #      one legacy participant with 2AFC, LEGACY0 (12 trials), and check that P000
    #      survives. We also check that with afc_block_expected=True (= operation
    #      with the blocks restored) it is dropped instead, pinning down that the
    #      relaxation is not "always lenient".
    ct_path = os.path.join(outdir, "records_design_a_contaminated.jsonl")
    n_sess_dropped = 0
    with open(ct_path, "w") as fh:
        for ln in open(da_path):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("type") == "session" and r.get("pid") == "P000":
                n_sess_dropped += 1
                continue
            fh.write(ln)
        for j in range(12):
            fh.write(json.dumps({
                "type": "afc", "pid": "LEGACY0", "pair_index": j,
                "song": "song00", "choice": "proposed",
                "is_repeat": False, "duration_sec": 25.0}) + "\n")
        fh.write(json.dumps({"type": "stage1", "pid": "LEGACY0",
                             "hp_correct": 6, "ap_correct": 6}) + "\n")
    res_ct = analyse(ct_path, cfg, manifest=None,
                     out_path=os.path.join(outdir,
                                           "pilot_analysis_selftest_contam.json"))
    inc_ct = next(row for row in res_ct["exclusion_table"]
                  if row["stage"] == "2_incomplete")
    checks.append((f"(g2) with an old record containing 2AFC mixed in (1 participant, 12 trials), "
                   f"the design A finisher P000 that dropped its session is not excluded",
                   n_sess_dropped == 1 and "P000" not in inc_ct["excluded"]
                   and "P000" in res_ct["kept"],
                   f"2_incomplete excluded = {inc_ct['excluded']}"))
    res_ct_on = analyse(ct_path,
                        deep_merge(cfg, {"screening": {"afc_block_expected": True}}),
                        manifest=None,
                        out_path=os.path.join(outdir,
                                              "pilot_analysis_selftest_contam_on.json"))
    inc_on = next(row for row in res_ct_on["exclusion_table"]
                  if row["stage"] == "2_incomplete")
    checks.append(("(g2) with afc_block_expected=True (operation with the blocks restored), "
                   "the same P000 is excluded because it has no 12 2AFC trials",
                   "P000" in inc_on["excluded"],
                   f"2_incomplete excluded = {inc_on['excluded']}"))

    # (g3) **Drift guard on the planned MDE of 6.27 points.**
    #      noninf_planned_mde_points is a copy of the value computed by
    #      design_tradeoff.py. If S_per or the variance assumptions are moved in the
    #      future, noninferiority_report's printout "Delta >= 6.27, so power is at
    #      least 80%" quietly becomes a lie. Keep the two cross-checked
    #      (raised in the  integration check).
    dt_mde: Optional[float] = None
    dt_why = ""
    try:
        import importlib.util as _ilu
        _dt_path = os.path.join(ROOT, "experiments", "subjective", "pilot_v2",
                                "tools", "design_tradeoff.py")
        _spec = _ilu.spec_from_file_location("design_tradeoff", _dt_path)
        _dt = _ilu.module_from_spec(_spec)          # type: ignore[arg-type]
        _spec.loader.exec_module(_dt)               # type: ignore[union-attr]
        _pw = json.load(open(_dt.POWER_JSON, encoding="utf-8"))
        _cv1 = _pw["contrasts"]["proposed-pro_mix|PQ|elig"]["cv"]
        dt_mde = float(_dt.mde(int(cfg["analysis"]["noninf_planned_L"]),
                               int(cfg["analysis"]["S_per"]), _cv1, 6.0, 25.0))
    except Exception as exc:                        # do not fail where the referenced file is absent
        dt_why = f"cannot cross-check ({type(exc).__name__}: {exc})"
    floor_cfg = float(cfg["analysis"]["noninf_planned_mde_points"])
    checks.append(("(g3) the planned MDE (noninf_planned_mde_points) matches "
                   "the value computed by design_tradeoff.py",
                   dt_mde is None or abs(dt_mde - floor_cfg) <= 0.05,
                   f"config {floor_cfg:.2f} vs design_tradeoff "
                   f"{dt_mde:.4f}" if dt_mde is not None
                   else f"config {floor_cfg:.2f} / {dt_why}"))

    # **The implementations must not have disappeared** (restoring the setting brings
    # them back). The same functions return applicable=True results on the version of
    # the same input that does contain 2AFC/ABX.
    checks.append(("(g) the afc_analysis / abx_analysis implementations are still there "
                   "(they run on input with 2AFC/ABX)",
                   res["afc"].get("applicable") is True
                   and res["abx"].get("applicable") is True
                   and np.isfinite(res["afc"].get("pref_rate_mean", float("nan")))
                   and np.isfinite(res["abx"].get("grand_accuracy", float("nan"))),
                   f"pref={res['afc'].get('pref_rate_mean'):.3f}, "
                   f"abx={res['abx'].get('grand_accuracy'):.3f}"))

    print("\n" + "=" * 74)
    print("SELF-TEST RESULTS")
    print("=" * 74)
    n_fail = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
        n_fail += int(not ok)
    verdict = "PASS" if n_fail == 0 else f"FAIL ({n_fail}/{len(checks)})"
    print(f"\nSELF-TEST OVERALL: {verdict}")
    summary = {"seed": seed, "overall": verdict,
               "checks": [{"name": n, "ok": bool(o), "detail": d}
                          for n, o, d in checks]}
    with open(os.path.join(outdir, "selftest_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", help="JSONL / JSON / directory")
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="built-in schema preset (spa_v1 = the real pilot_v2 SPA schema). "
                         "Applied before --config")
    ap.add_argument("--config", help="JSON that is deep-merged into DEFAULT_CONFIG")
    ap.add_argument("--manifest",
                    help="manifest for de-blinding (pilot_v2/stimuli/manifest.json "
                         "or flat blind_id -> {song, condition})")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "outputs", "analysis", "pilot_analysis.json"))
    ap.add_argument("--full-lmm", action="store_true",
                    help="also run the statsmodels crossed MixedLM cross-check (slow)")
    ap.add_argument("--dump-config", action="store_true",
                    help="print the default config (including the field map) to stdout")
    ap.add_argument("--flow-only", action="store_true",
                    help="print only the participant-flow counts. Ratings, "
                         "between-condition differences and p values are not printed at all. "
                         "A mode for deciding whether to run a second wave without optional stopping (internal notes 0.5.4.2)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    if args.preset:
        cfg = deep_merge(cfg, PRESETS[args.preset])
    if args.config:
        cfg = deep_merge(cfg, json.load(open(args.config)))
    if args.dump_config:
        print(json.dumps(cfg, indent=1, ensure_ascii=False))
        return 0
    if args.self_test:
        return self_test(args.seed)
    if args.flow_only:
        if not args.records:
            raise SystemExit("--flow-only requires --records")
        res = flow_only(args.records, cfg, args.manifest)
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out.replace(".json", "_flow.json"), "w") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        return 0
    if not args.records:
        ap.error("--records is required (or --self-test / --dump-config)")
    analyse(args.records, cfg, args.manifest, args.out, full_lmm=args.full_lmm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
