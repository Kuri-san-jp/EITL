#!/usr/bin/env python3
"""Decide which respondents qualify for the expert group from the screening
survey response CSV.

The authoritative specification is `internal notes`. This script implements its
§2 (questions) and §3 (decision rule) directly; if the two disagree,
SCREENER.md wins.

CPU only, standard library only. Does not use the GPU / slurm.

Usage:
  python3 tools/screen_eligibility.py --print-spec          # print the question spec
  python3 tools/screen_eligibility.py --check-app-js        # check consistency with the main study's app.js
  python3 tools/screen_eligibility.py --self-test           # verify the decision rule on synthetic data
  python3 tools/screen_eligibility.py --csv export.csv \
      --out-ids ops/eligible_ids.txt --out-json ops/verdicts.json

Eligibility criteria (the final form fixed by the  questionnaire
reduction. Do not change them after looking at the results):
  q_hearing = normal                          -- no hearing problems (added 
  AND q_device IN (over_ear, in_ear, tws)     -- headphones / earphones (added 
  AND q_prod  != none                         -- has experience mixing multitrack
  AND q_prod_years >= 0                       -- whole years are not used to exclude (covariate)
  AND at least 2 of the 3 knowledge questions correct
  AND did not pick the fictitious option (harmonic_gate) in q_know1

On  **4 questions were removed from the questionnaire** (the
questions themselves were retired):
  q_instr  years of instrument / singing experience  -- record-only, so removed
  q_mixes  number of songs completed                 -- already dropped from the criteria the same day, so removed
  q_paid   paid / commissioned experience            -- record-only, so removed
  q_recent when they last mixed                      -- **was a criterion but removed** (user decision)
The constants OK_MIXES / OK_RECENT / TIER2_MIXES are kept so the history can be
followed, but all of them are set to None / empty and taken out of the decision
rule.

Why hearing is decided inside the study (user decision, :
  Applying the Prolific prescreeners `hearing-difficulties` /
  `cochlear-implant` halves the addressable pool, but not because many people
  are affected: **everyone who has not answered those optional profile
  questions is excluded across the board**
  (measured: occupation only 394 people -> hearing 175 / cochlear implant 197).
  Since the required 178 entrants cannot be secured, these 2 conditions are
  taken out of the Prolific filter and decided by the in-study questionnaire
  item `q_hearing`, screening ineligible respondents out for $1.60
  (the authoritative amount is SCREENOUT_REWARD_CENTS = 160 in
  tools/prolific_study.py (revised ).

Why the playback device (q_device) **is** a criterion (user decision,
; the old policy was the opposite):
  What the device check in app.js actually measures is only "mobile or
  desktop" and "can it play Opus"; **whether it is headphones or speakers is
  not measured**. Speakers and the like are rejected by the headphone test 12
  minutes in, and everyone who fails there is still paid the full $13.40.
  Rejecting what self-report already tells us at questionnaire time costs only
  $1.60 and does not consume a scarce main-session slot (there are only 44).
  Both the recruitment text and the consent form state up front that
  "speakers will not pass", so it is not a surprise, and because "answering
  speakers reduces what you receive" there is no incentive to misreport.
  tws (AirPods etc.) is **eligible** because the consent form permits wireless.
  However, on-device signal processing can affect the ratings, so it is
  **recorded as a covariate** in the analysis.
  other is **ineligible**: the consent form requirement is
  "Wired or wireless over-ear or in-ear headphones", and the 3 options
  over_ear / in_ear / tws cover it, so other means "an answer that has not been
  confirmed to be headphones" (studio monitors, bone conduction, TV speakers
  and so on).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
APP_JS = os.path.abspath(os.path.join(HERE, "..", "app", "assets", "app.js"))

# Version of the decision criteria. Used to tie the pre-registration and the
# paper's Methods together. Always bump it when the criteria change.
# .a: questionnaire reduced from 12 to 8 questions. q_device added as
#               a criterion; q_recent (a criterion) and q_instr / q_mixes /
#               q_paid (record-only) retired.
#               The wording of q_prod was narrowed to "years mixing multitrack".
CRITERIA_VERSION = ".a"

FREQ = [("daily", "Daily"), ("weekly", "Weekly"), ("monthly", "Monthly"),
        ("rarely", "Rarely"), ("never", "Never")]
YEARS5 = [("none", "None"), ("lt1", "Less than 1 year"), ("y1_3", "1-3 years"),
          ("y3_10", "3-10 years"), ("gt10", "More than 10 years")]


class Q:
    def __init__(self, qid, section, kind, label, options=None, role="record",
                 shared_with_app=False, note=""):
        self.id = qid
        self.section = section
        self.kind = kind                    # "single" | "multi" | "int"
        self.label = label
        self.options = options or []
        self.role = role                    # record / eligibility / knowledge /
                                            # bogus / attention / distractor
        self.shared_with_app = shared_with_app
        self.note = note


# --------------------------------------------------------------------------
# Question specification (1:1 with SCREENER.md §2; the listing order is that order too)
# --------------------------------------------------------------------------
# SPEC is the **union** of two questionnaires.
#   (a) The 8 questions of the production SPA questionnaire
#       (app/assets/app.js SCREENS.questionnaire)
#       = q_device, q_hearing, q_prod, q_prod_years, q_know1, q_know2, q_know3, q_prev
#       These are **the questions participants actually answer** from
#        on, and shared_with_app=True puts a drift guard against
#       app.js on them.
#   (b) The s_* family, which existed only in the old two-stage setup (the
#       screener CSV of a separate study). They are **not** in the SPA, so they
#       cannot be used for production eligibility decisions
#       (the policy is not to add s_bogus / s_attn to the SPA; the role of the
#        fictitious option is played by harmonic_gate in q_know1). They are kept
#       in SPEC for the CSV-reading path (--csv) and its regression tests.
SPEC = OrderedDict()


def _q(*a, **kw):
    q = Q(*a, **kw)
    SPEC[q.id] = q


# The listing order matches app.js, so q_device comes first. A criterion (added .
_q("q_device", "0", "single",
   "Which playback device will you use for this study?",
   [("over_ear", "Over-ear / on-ear headphones"),
    ("in_ear", "In-ear earphones / earbuds (wired)"),
    ("tws", "True-wireless earbuds (e.g. AirPods)"),
    ("speakers", "Loudspeakers (computer / external)"),
    ("other", "Other")],
   role="eligibility", shared_with_app=True,
   note="only over_ear / in_ear / tws are eligible (OK_DEVICE). "
        "**The options must match app.js exactly** (the deployed app.js is authoritative). "
        "speakers does not meet the requirement (headphones), so it is ineligible. "
        "other is ineligible too: the consent form's "
        "'Wired or wireless over-ear or in-ear headphones' is covered by the 3 "
        "options above, so other means 'an answer not confirmed to be headphones'. "
        "tws is eligible, but on-device signal processing can affect the ratings, so "
        "**it is recorded as a covariate in the analysis** (which of the 3 was picked is stored).")

_q("q_hearing", "0", "single",
   "How would you describe your hearing?",
   [("normal", "Normal"),
    ("slight", "Slight difficulties, uncorrected"),
    ("aided", "Corrected (hearing aid)"),
    ("unsure", "Not sure")],
   role="eligibility", shared_with_app=True,
   note="only normal is eligible. **The options must match app.js exactly** (the "
        "deployed app.js is authoritative and SPEC is aligned to it). "
        "slight / aided / unsure are ineligible: the study description "
        "(REQUIREMENTS in prolific_settings.md §9) states "
        "'no hearing difficulties and no cochlear implant' up front, and cochlear "
        "implant users fall under aided, so the requirement is met without adding a question. "
        "ITU-R BS.1534 also assumes normal hearing. "
        "NOTE : a 'NORMAL HEARING' item was also added to the Requirements "
        "in the consent form, internal notes §A, and reflected in CONSENT_TEXT in app.js. "
        "It is now stated up front in 2 places, the description and the consent form "
        "(app.js was updated, so **it needs redeploying**)")

_q("s_listen", "A", "single",
   "In a typical week, roughly how many hours do you listen to music through "
   "headphones or earphones?",
   [("lt1", "Less than 1 hour"), ("h1_5", "1 to 5 hours"),
    ("h5_15", "5 to 15 hours"), ("h15_30", "15 to 30 hours"),
    ("gt30", "More than 30 hours")], role="record")

_q("s_device_habit", "A", "single",
   "Which do you use most often when you listen to music at home?",
   [("over_ear", "Over-ear or on-ear headphones"),
    ("in_ear", "Wired in-ear earphones"),
    ("tws", "True-wireless earbuds"),
    ("speakers", "Loudspeakers or studio monitors"),
    ("builtin", "Built-in phone or laptop speakers")], role="record")

_q("s_podcast", "A", "single",
   "How often do you listen to podcasts, audiobooks or radio?",
   FREQ, role="distractor")

_q("s_video", "A", "single",
   "How often do you edit video, for any purpose?", FREQ, role="distractor")

_q("s_attn", "A", "single",
   "This question checks that the instructions are being read. "
   "Please select \"Monthly\".", FREQ, role="attention",
   note="correct answer monthly. Record-only by default (--strict-attention makes it required)")

# : q_instr (years of instrument / singing experience) was **removed
# entirely**. It was record-only and used neither in the eligibility decision
# nor in the paper's claims, so shortening the questionnaire wins.
# It is being removed from app.js too (a different owner; it stays in the SPA
# until the app/ side deletion lands).

_q("s_roles", "B", "multi",
   "Which of the following have you done at any point, even occasionally? "
   "Select all that apply.",
   [("perform", "Performed or recorded as a musician or singer"),
    ("compose", "Composed or produced original music"),
    ("record", "Recorded other people's instruments or vocals"),
    ("mix", "Mixed multitrack recordings"),
    ("master", "Mastered finished tracks"),
    ("live", "Operated live sound at events"),
    ("dj", "DJed"),
    ("postpro", "Sound design or dialogue editing for video or games"),
    ("broadcast", "Edited audio for broadcast or podcasts"),
    ("none", "None of these")], role="record")

_q("q_prod", "C", "single",
   # Wording changed . The old text, "How many years of music
   # production / mixing experience do you have?", used the compound
   # "music production / mixing", which let someone who only composes or
   # programs pick gt10 and become eligible (q_prod_years = 0 = no multitrack
   # experience still passes, because MIN_PROD_YEARS = 0).
   # The never-exclusion in q_recent happened to plug that hole, but retiring
   # that question exposes it. Narrowing the wording makes
   # **q_prod != none directly mean "has experience mixing multitrack"**.
   # As a by-product, inconsistent_years (band vs whole-number contradiction)
   # becomes a comparison of the same quantity, which also removes the "honest
   # answer yet flagged as inconsistent" problem.
   # **The options are not changed** (app.js is already deployed; label changes
   # are done by the app/ owner).
   "How many years have you been mixing multitrack recordings?",
   [("none", "None"), ("lt1", "Less than 1 year"),
    ("y1_2", "1 to under 2 years"), ("y2_3", "2 to under 3 years"),
    ("y3_5", "3 to under 5 years"), ("y5_10", "5 to under 10 years"),
    ("gt10", "10 years or more")],
   role="eligibility", shared_with_app=True,
   note="anything but none is eligible. --check-app-js only looks at the option "
        "codes, so changing the wording alone does not trip the drift guard "
        "(replacing the wording on the app.js side is the app/ owner's job)")

_q("q_prod_years", "C", "int",
   "In whole years, for how long have you been mixing multitrack recordings? "
   "Enter a whole number, and enter 0 if you have never done this.",
   [], role="eligibility",
   note="an integer 0-70. ANDed with the band q_prod (SCREENER.md §3.2)")

# : the following 3 questions were **removed entirely** (also deleted
# from SPEC).
#   q_mixes  "How many full songs have you mixed ... (start to finish)?"
#            options none / n1_2 / n3_9 / n10_29 / n30.
#            It had been dropped from the criteria the same day
#            (OK_MIXES = None), so it was left in for recording only. None of
#            the 5 prior works (FxNorm / Koo / ITO-Master / MEGAMI / StemFX)
#            makes song count a recruitment condition.
#   q_recent "When did you last mix a song?"
#            options never / gt2y / y1_2 / m6_12 / lt6m. **It was a criterion
#            but was removed** (user decision). The wording change in q_prod
#            takes over the job of rejecting never.
#   q_paid   "Have you ever been paid or commissioned to mix music ...?"
#            record-only.
# The constants OK_RECENT / OK_MIXES / TIER2_MIXES are kept below but taken out
# of the decision rule.

_q("s_daw", "C", "single",
   "Which software do you use most when mixing?",
   [("protools", "Pro Tools"), ("logic", "Logic Pro"),
    ("ableton", "Ableton Live"), ("cubase", "Cubase or Nuendo"),
    ("reaper", "Reaper"), ("flstudio", "FL Studio"),
    ("studioone", "Studio One"), ("other", "Other software"),
    ("na", "I do not mix")], role="record")

_q("s_monitor", "C", "single",
   "When you mix, what do you mainly listen on?",
   [("monitors", "Studio monitors"), ("headphones", "Headphones"),
    ("consumer", "Consumer speakers"), ("varies", "It varies"),
    ("na", "I do not mix")], role="record")

_q("s_bogus", "C", "multi",
   "Which of the following have you used yourself when working on audio? "
   "Select all that apply. It is fine to select none.",
   [("eq", "A parametric EQ"), ("comp", "A bus compressor"),
    ("deesser", "A de-esser"), ("convrev", "A convolution reverb"),
    ("fake_verrian", "A Verrian VX-7 stereo aligner"),
    ("none", "None of these")], role="bogus",
   note="fake_verrian is a fictitious product. Re-confirm that it does not exist before publishing")

_q("q_know1", "D", "single",
   "In a compressor, which control sets how much the signal is reduced once "
   "it passes the threshold?",
   [("skip", "I don't know / prefer not to answer"), ("ratio", "Ratio"),
    ("attack", "Attack"), ("makeup", "Make-up gain"),
    ("harmonic_gate", "Harmonic gate")],
   role="knowledge", shared_with_app=True,
   note="correct answer ratio. harmonic_gate is fictitious = bogus (tallied separately from wrong answers)")

_q("q_know2", "D", "single",
   "You want the same reverb on several tracks at once. How is this normally "
   "routed?",
   [("skip", "I don't know / prefer not to answer"),
    ("send", "Via a send to an aux/return bus"),
    ("insert_each", "A separate insert on every track"),
    ("insert_master", "One insert on the master bus"),
    ("sidechain", "Through the sidechain input")],
   role="knowledge", shared_with_app=True, note="correct answer send")

_q("q_know3", "D", "single",
   "Which unit is used for the integrated loudness of a finished track?",
   [("skip", "I don't know / prefer not to answer"), ("lufs", "LUFS"),
    ("dbfs_peak", "dBFS peak"), ("rms_db", "RMS dB"), ("hz", "Hz")],
   role="knowledge", shared_with_app=True,
   note="correct answer lufs. All 3 distractors are real units that are easy to "
        "confuse with LUFS. "
        "Unified with the app.js side (dBFS peak / RMS dB / Hz) on ")

_q("q_prev", "E", "single",
   "How many online listening tests have you taken in the last month?",
   [("none", "None"), ("few", "1-2"), ("many", "3 or more")],
   role="record", shared_with_app=True,
   note="record-only (not used in the eligibility decision). It is one of the 8 "
        "questions kept in the  reduction, so it is listed in SPEC to "
        "put a drift guard on its options")

# --------------------------------------------------------------------------
# Decision criteria (1:1 with the SQL in SCREENER.md §3.1)
# --------------------------------------------------------------------------
# For hearing, **only normal** is eligible. slight (mild, uncorrected) /
# aided (corrected with a hearing aid etc.) / unsure (does not know) are all
# ineligible. Because aided also catches cochlear implant users, this single
# question meets the requirement stated up front without splitting "hearing"
# and "cochlear implant" into separate questions.
OK_HEARING = ("normal",)
# Playback device. **Added as a criterion** on  (it was record-only
# until then). Only the 3 headphone / earphone options are eligible.
# speakers does not meet the requirement, and other is "an answer not confirmed
# to be headphones", so both are ineligible.
# See the module docstring and the note on q_device for the detailed reasoning.
OK_DEVICE = ("over_ear", "in_ear", "tws")
# User instruction : relaxed to **anything but None is eligible**
# (previously: 2 years or more). On the band side only "none" is rejected.
# The whole-number side (MIN_PROD_YEARS) has to be relaxed to 0 at the same
# time, otherwise someone who picked "less than 1 year" would be rejected on
# the whole number 0, which is contradictory.
OK_PROD = ("lt1", "y1_2", "y2_3", "y3_5", "y5_10", "gt10")
# 2 -> 0 to match the above. The whole-number input is **not used to exclude**;
# it is still collected as a covariate so the paper can report a point estimate
# of "mean X.X years".
# The band/whole-number contradiction detection (inconsistent_years) still works.
MIN_PROD_YEARS = 0
# User instruction : relaxed to "3 or more completed songs"
# (previously: 10 or more).
# The old option "1-4" straddles 3 and so could not express "3 or more", so the
# boundaries were recut as 1-2 / 3-9 (the app.js options were changed at the
# same time).
# User decision : **the number of completed songs was dropped from the
# criteria** (record-only). None of the 5 prior works (FxNorm / Koo /
# ITO-Master / MEGAMI / StemFX) makes song count a recruitment condition, so
# there is no basis for it. Recruitment volume takes priority.
# "People who have never mixed" are rejected by q_prod="none" (the wording was
# narrowed to "years mixing multitrack", so none directly means no experience).
#  (same day, later): **the question q_mixes itself was retired**, so
# the value cannot even be obtained any more.
# The constant is kept only so that it can be restored in one line if the
# decision is made to bring the condition back.
OK_MIXES = None   # None = not used in the eligibility decision (the question is retired too)
# User instruction  (earlier): relaxed to "within the last 2 years"
# (previously: within the last 12 months).
# That left only never (never done it) and gt2y (more than 2 years ago) being
# rejected.
# User decision  (later): **the question q_recent itself was deleted**,
# so this condition ceased to exist. Its value just before retirement was
# ("y1_2", "m6_12", "lt6m").
# The job of "rejecting never (people who have never mixed)" is taken over by
# the wording change in q_prod
# (music production / mixing -> mixing multitrack recordings).
OK_RECENT = None  # None = not used in the eligibility decision (the question is retired too)
KNOW_ANSWERS = OrderedDict([("q_know1", "ratio"), ("q_know2", "send"),
                            ("q_know3", "lufs")])
MIN_KNOW_CORRECT = 2
# Fictitious options. **The only one live in the SPA is harmonic_gate in
# q_know1** (s_bogus is not a question in the SPA, so it has no effect on the
# production decision. The policy of not adding s_bogus / s_attn to the SPA was
# also confirmed on .
BOGUS = OrderedDict([("q_know1", "harmonic_gate"), ("s_bogus", "fake_verrian")])
ATTN_ANSWER = ("s_attn", "monthly")
# : since eligibility was relaxed to 3 or more songs, tier-2 (the
# contingency that relaxes it one step further) becomes the next band down,
# "1-2 songs". The old value "n5_9" is wrong because it is now on the eligible
# side.
# Since the number of completed songs is no longer a criterion, a tier-2 that
# relaxes it by one step lost its meaning as well.
# It is emptied and thereby disabled (even with --tier2 it yields 0 people).
TIER2_MIXES = ()                 # the pre-registered contingency (SCREENER.md §6.3)

# The year range [lo, hi) each q_prod band represents. Used to detect
# contradictions with the whole-number answer.
PROD_RANGE = {"none": (0, 1), "lt1": (0, 1), "y1_2": (1, 2), "y2_3": (2, 3),
              "y3_5": (3, 5), "y5_10": (5, 10), "gt10": (10, 71)}

# Question IDs retired on . They are gone from SPEC, but removing
# them from app.js (the deployed SPA) is the app/ owner's job, so they stay on
# the SPA side for a while. --check-app-js reports "retired questions still
# present" to make a missed deletion visible.
# Retiring them does not break the decision: they are not in APP_REQUIRED, so
# any extra answers the SPA sends are simply ignored.
RETIRED_QIDS = ("q_instr", "q_mixes", "q_paid", "q_recent")

# The items required for the decision. **If you add to this, you must also add
# to the test payload in backend/aws/verify_deploy.py and to BASE_ANSWERS in
# tools/live_check.py.** If you do not, an expert payload comes out "ineligible
# via missing_data" and falsely reports the deployment as broken (this actually
# happened).
# : q_device added, q_recent (retired question) removed.
REQUIRED_FOR_JUDGEMENT = ("q_hearing", "q_device", "q_prod", "q_prod_years",
                          "q_know1", "q_know2", "q_know3",
                          "s_bogus")

# Order of the waterfall (SCREENER.md §3.4). Only the first reason is counted
# per person.
# fail_hearing is placed **before** years of experience. It is a prerequisite of
# a listening test, and for the participant-flow section it reads better to see
# "rejected on hearing" before "rejected on years of experience".
# fail_device was inserted **right after fail_hearing and before fail_years**
# . Same reasoning: read the prerequisites of the listening
# environment (ears and headphones) first, then the breakdown of expertise
# (experience and knowledge).
# fail_songs / fail_recency **can no longer occur** because their questions were
# retired. The names are kept so that old verdict JSONs can still be read (and
# the order is not moved either).
WATERFALL = ["missing_data", "fail_hearing", "fail_device", "fail_years",
             "fail_songs", "fail_recency", "fail_knowledge", "fail_bogus",
             "fail_attention"]
REASON_JA = {
    "missing_data": "a required item is missing or has an uninterpretable value",
    "fail_hearing": "hearing is not normal (slight / aided / unsure)",
    "fail_device": "the playback device is not headphones / earphones (speakers / other)",
    "fail_years": "no experience mixing multitrack (q_prod = none)",
    "fail_songs": "(retired) the completed-mix count condition. Removed along with the question q_mixes",
    "fail_recency": "(retired) has not mixed recently. Removed along with the question q_recent",
    "fail_knowledge": "fewer than 2 knowledge questions correct",
    "fail_bogus": "picked a fictitious (bogus) item",
    "fail_attention": "failed the attention check",
}

ID_COL_CANDIDATES = ["participantid", "participant_id", "prolificpid",
                     "prolificid", "prolific_pid", "id", "submissionid",
                     "responseid"]
DURATION_COL_CANDIDATES = ["timetaken", "durationinseconds", "duration",
                           "timetakenseconds"]

# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
_DASHES = "‐‑‒–—―−"


def nrm(v) -> str:
    """Normalization for comparison: drop punctuation, whitespace and case
    differences, keeping only alphanumerics."""
    if v is None:
        return ""
    s = str(v)
    for d in _DASHES:
        s = s.replace(d, "-")
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"'))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _value_index(q: Q):
    """Build an index that resolves to the same code whether the value arrives
    as a code or as a label."""
    idx = {}
    for code, label in q.options:
        for key in (nrm(code), nrm(label)):
            if key and key in idx and idx[key] != code:
                raise ValueError(
                    "normalized option keys collide in question %s: %r -> %r / %r"
                    % (q.id, key, idx[key], code))
            if key:
                idx[key] = code
    return idx


VALUE_INDEX = {qid: _value_index(q) for qid, q in SPEC.items() if q.options}


def parse_int_years(raw):
    if raw is None:
        return None
    m = re.search(r"-?\d+", str(raw).replace(",", ""))
    if not m:
        return None
    n = int(m.group(0))
    if n < 0 or n > 70:
        return None
    return n


def parse_value(qid, raw):
    """Map a raw CSV value to a spec code. None if it cannot be interpreted
    (= treated as missing).

    For multi-select, return the list of codes that resolved and put the unknown
    tokens in the second return value.
    """
    q = SPEC[qid]
    if raw is None:
        return (None, [])
    s = str(raw).strip()
    if s == "":
        return (None, [])
    if q.kind == "int":
        return (parse_int_years(s), [])
    idx = VALUE_INDEX[qid]
    if q.kind == "single":
        code = idx.get(nrm(s))
        return (code, [] if code else [s])
    # multi
    codes, unknown = [], []
    for tok in re.split(r"[,;\n]+", s):
        tok = tok.strip()
        if not tok:
            continue
        code = idx.get(nrm(tok))
        if code:
            if code not in codes:
                codes.append(code)
        else:
            unknown.append(tok)
    return ((codes if codes else None), unknown)


# --------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------
def classify(values, strict_attention=False):
    """values: {qid: code|list|int|None} -> a verdict dict."""
    need = list(REQUIRED_FOR_JUDGEMENT)
    if strict_attention:
        need.append("s_attn")
    missing = [q for q in need
               if values.get(q) is None or values.get(q) == []]

    hearing = values.get("q_hearing")
    device = values.get("q_device")
    prod = values.get("q_prod")
    years = values.get("q_prod_years")
    # q_mixes / q_recent were retired along with their questions on .
    # They are not in SPEC, so they never enter values and the following are
    # always None (the constants are None too, so they have no effect on the
    # decision).
    mixes = values.get("q_mixes")
    recent = values.get("q_recent")
    bogus_sel = values.get("s_bogus") or []

    n_know = sum(1 for q, a in KNOW_ANSWERS.items() if values.get(q) == a)
    bogus_hits = []
    if values.get("q_know1") == BOGUS["q_know1"]:
        bogus_hits.append("q_know1")
    if BOGUS["s_bogus"] in bogus_sel:
        bogus_hits.append("s_bogus")

    years_bin_ok = prod in OK_PROD
    years_int_ok = isinstance(years, int) and years >= MIN_PROD_YEARS
    inconsistent_years = False
    if prod in PROD_RANGE and isinstance(years, int):
        lo, hi = PROD_RANGE[prod]
        inconsistent_years = not (lo <= years < hi)

    fails = []
    if missing:
        fails.append("missing_data")
    if hearing not in OK_HEARING:
        fails.append("fail_hearing")
    # The playback device was added as a criterion on . speakers /
    # other are ineligible.
    if device not in OK_DEVICE:
        fails.append("fail_device")
    if not (years_bin_ok and years_int_ok):
        fails.append("fail_years")
    # The number of completed songs was dropped from the criteria on 
    # (OK_MIXES = None) and the question was retired the same day. The same goes
    # for when they last mixed (q_recent / OK_RECENT).
    # The constants and branches are kept so that, if the decision is made to
    # bring the conditions back, restoring them is a one-line change to the
    # constants.
    if OK_MIXES is not None and mixes not in OK_MIXES:
        fails.append("fail_songs")
    if OK_RECENT is not None and recent not in OK_RECENT:
        fails.append("fail_recency")
    if n_know < MIN_KNOW_CORRECT:
        fails.append("fail_knowledge")
    if bogus_hits:
        fails.append("fail_bogus")
    attn_ok = values.get(ATTN_ANSWER[0]) == ATTN_ANSWER[1]
    if strict_attention and not attn_ok:
        fails.append("fail_attention")

    reason = next((r for r in WATERFALL if r in fails), None)
    eligible = not fails

    # tier-2: respondents rejected *only* because q_mixes is 1-2 songs (the
    # pre-registered contingency).
    # With TIER2_MIXES = () and the question retired, nobody matches right now.
    tier2 = (not eligible and mixes in TIER2_MIXES
             and set(fails) == {"fail_songs"})

    return {
        "eligible": eligible,
        "tier2": tier2,
        "reason": reason,
        "fails": fails,
        "missing": missing,
        "n_know_correct": n_know,
        "bogus_hits": bogus_hits,
        "inconsistent_years": inconsistent_years,
        "attention_ok": attn_ok,
        "years_bin_ok": years_bin_ok,
        "years_int_ok": years_int_ok,
    }


# --------------------------------------------------------------------------
# Eligibility decision for the main study SPA questionnaire (app/assets/app.js)
# --------------------------------------------------------------------------
# In the single-study setup , screening and the main study are
# merged into one session and the eligibility decision is made server-side
# (POST /api/elig_score in backend/aws/backend_app.py) right after the
# questionnaire. The decision rule uses the constants above (OK_HEARING /
# OK_DEVICE / OK_PROD / MIN_PROD_YEARS / KNOW_ANSWERS / MIN_KNOW_CORRECT /
# BOGUS; the retired OK_MIXES / OK_RECENT are None) and classify() **as they
# are**. Thresholds and correct answers must not be redefined here (there is
# exactly one source of truth).
#
# Differences from the screener CSV:
#   - The app.js questionnaire has neither s_bogus (multi-select of fictitious
#     products) nor s_attn. The bogus decision reduces to
#     q_know1 = harmonic_gate.
#   - The q_know3 options were unified with the app.js side on 
#     (lufs / dbfs_peak / rms_db / hz). Before that, SPEC held dbfs/dbspl/dbtp
#     and parse_value could not resolve the app.js answers, so they became
#     "missing". For knowledge questions only "does it equal the correct code"
#     matters, so a **non-empty** answer that cannot be resolved is treated as a
#     wrong answer (not as missing). Without this handling, if the options drift
#     apart again, "eligible respondents with 2/3 knowledge correct" would be
#     wiped out by missing_data.
#   - q_hearing was promoted to **a criterion** on . Instead of the
#     Prolific prescreeners (hearing-difficulties / cochlear-implant), this
#     single question decides it inside the study (see the module docstring).
#     It has existed in app.js for a while, and its options are authoritative.
#     SPEC is aligned to app.js, and the drift guard in self-test [9]
#     mechanically watches that the two agree.
#   - q_device (self-reported playback device) was **promoted to a criterion**
#     on  (the old policy was "do not make it a condition, since we
#     rely solely on measurement"). The device check in app.js only measures the
#     device type and whether Opus plays; whether it is headphones or speakers
#     is not measured. It does not duplicate the measurement, so there is no
#     conflict. See the module docstring for the detailed reasoning.
#     **It is a required item, so forgetting to list it here makes everyone
#     ineligible via missing_data.**
#   - q_mixes / q_recent / q_instr / q_paid were retired along with their
#     questions on . They are out of APP_REQUIRED too, so the decision
#     works even if the SPA does not send them (and conversely, extra answers
#     are ignored even before the questions are removed from app.js).
APP_REQUIRED = ("q_hearing", "q_device", "q_prod", "q_prod_years",
                "q_know1", "q_know2", "q_know3")
# s_bogus does not exist in the SPA questionnaire. To satisfy the required items
# of classify(), insert a sentinel meaning "did not pick a fictitious item". As
# long as it does not equal BOGUS["s_bogus"] it has no effect whatsoever on the
# decision.
APP_NO_BOGUS = "__not_asked__"
# Sentinel for a knowledge-question answer that could not be resolved to an
# option (= a wrong answer).
APP_WRONG_ANSWER = "__unrecognised__"
KNOWLEDGE_QIDS = tuple(KNOW_ANSWERS.keys())


def classify_app_questionnaire(answers):
    """Decide eligibility from the SPA questionnaire answers (the source of
    truth for the server-side /api/elig_score).

    answers: {qid: raw value}. The same shape as the payload of
    log("questionnaire") in app.js.
    Values are accepted either as codes ("y5_10") or as labels
    ("5 to under 10 years").

    Returns: {"eligible": bool, "reasons": [waterfall names...], ...}
    """
    if not isinstance(answers, dict):
        answers = {}
    values, unknown = {}, {}
    for qid in APP_REQUIRED:
        raw = answers.get(qid)
        v, unk = parse_value(qid, raw)
        if v is None and unk and qid in KNOWLEDGE_QIDS:
            # An unresolvable knowledge answer is a "wrong answer", not missing
            # (see the comment above)
            v = APP_WRONG_ANSWER
        values[qid] = v
        if unk:
            unknown[qid] = unk
    values["s_bogus"] = [APP_NO_BOGUS]
    v = classify(values, strict_attention=False)
    return {
        "eligible": bool(v["eligible"]),
        "reasons": list(v["fails"]),
        "reason": v["reason"],
        "n_know_correct": v["n_know_correct"],
        "bogus_hits": v["bogus_hits"],
        "inconsistent_years": v["inconsistent_years"],
        "missing": v["missing"],
        "unknown_values": unknown,
        "criteria_version": CRITERIA_VERSION,
    }


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------
def resolve_columns(headers, colmap=None):
    """question ID -> CSV column name. Resolution order per SCREENER.md §4.2."""
    colmap = colmap or {}
    hn = {h: nrm(h) for h in headers}
    resolved, ambiguous = {}, {}
    for qid, q in SPEC.items():
        if qid in colmap:
            if colmap[qid] not in headers:
                raise SystemExit("the --map column name does not exist in the CSV: %r (question %s)"
                                 % (colmap[qid], qid))
            resolved[qid] = colmap[qid]
            continue
        key = nrm(qid)
        exact = [h for h in headers if hn[h] == key]
        if len(exact) == 1:
            resolved[qid] = exact[0]
            continue
        if len(exact) > 1:
            ambiguous[qid] = exact
            continue
        bylabel = [h for h in headers if hn[h] == nrm(q.label)]
        if len(bylabel) == 1:
            resolved[qid] = bylabel[0]
            continue
        if len(bylabel) > 1:
            ambiguous[qid] = bylabel
            continue
        # When the column name contains the ID as a token (to avoid mixing up
        # q_prod and q_prod_years, the longer ID is settled first and the rest
        # is looked at afterwards)
        token = [h for h in headers
                 if re.search(r"(^|[^a-z0-9])" + re.escape(qid) +
                              r"([^a-z0-9]|$)", h.lower())]
        token = [h for h in token if h not in resolved.values()]
        if len(token) == 1:
            resolved[qid] = token[0]
        elif len(token) > 1:
            ambiguous[qid] = token
    # Check that no two questions resolved to the same column.
    # Real example: if the CSV has no column for `q_prod` but only a
    # `q_prod_years` column, rule 3 (token containment) makes `q_prod` grab the
    # `q_prod_years` column, everyone's q_prod becomes "uninterpretable" =
    # missing_data, and the result is silently 0 eligible people.
    # Never proceed silently; always stop.
    byhead = {}
    for qid, h in resolved.items():
        byhead.setdefault(h, []).append(qid)
    collide = {h: qs for h, qs in byhead.items() if len(qs) > 1}
    return resolved, ambiguous, collide


def find_col(headers, candidates, override=None):
    if override:
        if override not in headers:
            raise SystemExit("the specified column does not exist in the CSV: %r" % override)
        return override
    for h in headers:
        if nrm(h) in candidates:
            return h
    return None


def load_rows(fileobj, colmap=None, id_col=None, duration_col=None):
    reader = csv.DictReader(fileobj)
    headers = reader.fieldnames or []
    resolved, ambiguous, collide = resolve_columns(headers, colmap)
    unresolved = [q for q in SPEC if q not in resolved]
    if unresolved or ambiguous or collide:
        msg = ["could not map the CSV columns onto the questions. Specify them explicitly with --map."]
        if unresolved:
            msg.append("  unresolved question ID: " + ", ".join(unresolved))
        for qid, hs in ambiguous.items():
            msg.append("  question %s has multiple candidates: %s" % (qid, hs))
        for h, qs in collide.items():
            msg.append("  the same column %r resolved for multiple questions: %s" % (h, ", ".join(qs)))
        msg.append("  list of CSV column names:")
        msg += ["    - " + h for h in headers]
        raise SystemExit("\n".join(msg))
    idc = find_col(headers, ID_COL_CANDIDATES, id_col)
    if idc is None:
        raise SystemExit(
            "could not find the participant ID column. Specify it with --id-col.\n"
            "  list of CSV column names: " + ", ".join(headers))
    durc = find_col(headers, DURATION_COL_CANDIDATES, duration_col)

    rows = []
    for raw in reader:
        vals, unknown = {}, {}
        for qid in SPEC:
            v, unk = parse_value(qid, raw.get(resolved[qid]))
            vals[qid] = v
            if unk:
                unknown[qid] = unk
        dur = None
        if durc:
            try:
                dur = float(re.sub(r"[^0-9.]", "", str(raw.get(durc) or "")))
            except ValueError:
                dur = None
        rows.append({"participant_id": (raw.get(idc) or "").strip(),
                     "values": vals, "unknown": unknown, "duration_sec": dur})
    return rows, {"id_col": idc, "duration_col": durc, "columns": resolved}


# --------------------------------------------------------------------------
# aggregation and reporting
# --------------------------------------------------------------------------
def wilson(k, n, z=1.959963985):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def summarize(rows, strict_attention=False):
    seen, uniq, dup, blank_pid = set(), [], 0, 0
    for r in rows:
        pid = r["participant_id"]
        if pid and pid in seen:
            dup += 1
            continue
        if pid:
            seen.add(pid)
        else:
            # Blank PID. Without an ID they cannot be attached to a participant
            # group, so they are always dropped from the eligible list and the
            # count is reported explicitly (never silently write out a blank line).
            blank_pid += 1
        uniq.append(r)
    for r in uniq:
        r["verdict"] = classify(r["values"], strict_attention)

    n = len(uniq)
    elig = [r for r in uniq if r["verdict"]["eligible"]]
    tier2 = [r for r in uniq if r["verdict"]["tier2"]]
    waterfall = Counter(r["verdict"]["reason"] for r in uniq
                        if r["verdict"]["reason"])
    independent = Counter()
    for r in uniq:
        for f in r["verdict"]["fails"]:
            independent[f] += 1

    know = {q: sum(1 for r in uniq if r["values"].get(q) == a)
            for q, a in KNOW_ANSWERS.items()}
    bogus_any = [r for r in uniq if r["verdict"]["bogus_hits"]]
    bogus_by = Counter()
    for r in bogus_any:
        for h in r["verdict"]["bogus_hits"]:
            bogus_by[h] += 1
    # "Claims to be experienced yet picks a fictitious item" = a direct
    # indicator of misreporting
    bogus_experienced = [r for r in bogus_any
                         if r["values"].get("q_prod") in ("y5_10", "gt10")]
    inconsistent = [r for r in uniq if r["verdict"]["inconsistent_years"]]
    attn_fail = [r for r in uniq if not r["verdict"]["attention_ok"]]

    durs = sorted(r["duration_sec"] for r in uniq
                  if isinstance(r["duration_sec"], float))
    med = durs[len(durs) // 2] if durs else None
    speeders = ([r for r in uniq
                 if isinstance(r["duration_sec"], float)
                 and r["duration_sec"] < 0.4 * med] if med else [])

    def dist(qid, subset):
        c = Counter()
        for r in subset:
            v = r["values"].get(qid)
            if isinstance(v, list):
                for x in v:
                    c[x] += 1
            else:
                c[v if v is not None else "(missing)"] += 1
        return c

    years = [r["values"]["q_prod_years"] for r in elig
             if isinstance(r["values"].get("q_prod_years"), int)]
    ystat = None
    if years:
        m = sum(years) / len(years)
        sd = (math.sqrt(sum((y - m) ** 2 for y in years) / (len(years) - 1))
              if len(years) > 1 else 0.0)
        ystat = {"n": len(years), "mean": m, "sd": sd,
                 "min": min(years), "max": max(years)}

    # Missing counts per question used in the decision. Reported so that a
    # question like s_bogus, which "passes even when unanswered", can be caught
    # by eye if it is dropping everyone into missing_data.
    missing_by_q = {q: sum(1 for r in uniq
                           if r["values"].get(q) is None or r["values"].get(q) == [])
                    for q in REQUIRED_FOR_JUDGEMENT + ("s_attn",)}

    elig_ids = [r["participant_id"] for r in elig if r["participant_id"]]
    elig_noid = sum(1 for r in elig if not r["participant_id"])

    return {
        "criteria_version": CRITERIA_VERSION,
        "strict_attention": strict_attention,
        "n_rows": len(rows), "n_duplicate_ids": dup, "n_unique": n,
        "n_blank_participant_id": blank_pid,
        "n_eligible_without_id": elig_noid,
        "missing_by_question": missing_by_q,
        "n_eligible": len(elig), "n_tier2": len(tier2),
        "eligible_rate": (len(elig) / n if n else 0.0),
        "eligible_ci95": wilson(len(elig), n),
        "waterfall": dict(waterfall), "independent_fails": dict(independent),
        "knowledge_correct": know,
        "n_bogus_any": len(bogus_any), "bogus_by_item": dict(bogus_by),
        "n_bogus_experienced": len(bogus_experienced),
        "n_inconsistent_years": len(inconsistent),
        "n_attention_fail": len(attn_fail),
        "median_duration_sec": med, "n_speeders": len(speeders),
        "years_eligible": ystat,
        # : q_paid / q_mixes / q_recent / q_instr were retired along
        # with their questions, so their distributions cannot be taken either
        # (they are gone from SPEC, so dist() would raise KeyError).
        # Instead we report q_device (a criterion, and the question whose tws
        # value is reported as a covariate).
        "dist_eligible": {q: dict(dist(q, elig))
                          for q in ("q_device", "q_prod", "q_prev",
                                    "s_daw", "s_monitor")},
        "dist_all_roles": dict(dist("s_roles", uniq)),
        "eligible_ids": elig_ids,
        "tier2_ids": [r["participant_id"] for r in tier2 if r["participant_id"]],
        "_unique_rows": uniq,
    }


def methods_sentence(s):
    n, m = s["n_unique"], s["n_eligible"]
    lo, hi = s["eligible_ci95"]
    y = s["years_eligible"]
    de = s["dist_eligible"]
    # With the  questionnaire reduction, paid experience (q_paid) and
    # when they last mixed (q_recent) no longer exist as questions, so they
    # cannot go in Methods either. We report the playback device breakdown
    # instead (it is a criterion, and tws has to be reported as a covariate).
    dev = de["q_device"]
    parts = [
        "Of the %d respondents who completed the screening questionnaire, "
        "%d (%.1f%%, 95%% CI %.1f-%.1f) met all pre-registered eligibility "
        "criteria." % (n, m, 100.0 * s["eligible_rate"], 100 * lo, 100 * hi)]
    if y:
        parts.append(
            "Eligible respondents reported a mean of %.1f years (SD %.1f, "
            "range %d-%d) of experience mixing multitrack recordings."
            % (y["mean"], y["sd"], y["min"], y["max"]))
    parts.append(
        "Eligible respondents listened over on-ear or over-ear headphones "
        "(%d), wired in-ear earphones (%d) or true-wireless earbuds (%d); "
        "participants reporting loudspeakers or an unspecified device were "
        "screened out at the questionnaire."
        % (dev.get("over_ear", 0), dev.get("in_ear", 0), dev.get("tws", 0)))
    parts.append(
        "%d respondents endorsed a fictitious item and were excluded; %d of "
        "them were excluded on that ground alone."
        % (s["n_bogus_any"], s["waterfall"].get("fail_bogus", 0)))
    return " ".join(parts)


def print_report(s, src):
    p = print
    p("=" * 68)
    p("automix_01 screening eligibility decision  (criteria %s)" % s["criteria_version"])
    p("=" * 68)
    p("input: %s" % src)
    p("response rows: %d  / respondents after removing duplicate IDs: %d  (for %d duplicate rows the first one wins)"
      % (s["n_rows"], s["n_unique"], s["n_duplicate_ids"]))
    if s["n_blank_participant_id"]:
        p("!! rows with a blank participant ID: %d   (of which %d met the eligibility criteria)"
          % (s["n_blank_participant_id"], s["n_eligible_without_id"]))
        p("   Without an ID they cannot be registered into a participant group. Check the "
          "--id-col setting and the PROLIFIC_PID prefill. They were excluded from the eligible list.")
    lo, hi = s["eligible_ci95"]
    p("")
    p("[eligible] %d / %d = %.1f%%   (Wilson 95%% CI %.1f-%.1f%%)"
      % (s["n_eligible"], s["n_unique"], 100 * s["eligible_rate"],
         100 * lo, 100 * hi))
    p("[tier-2 (ineligible on 1-2 songs alone)] %d people  NOTE: used only when the pre-registered contingency fires"
      % s["n_tier2"])
    if s["strict_attention"]:
        p("NOTE --strict-attention: the attention check was included as a required condition")
    p("")
    p("-- exclusion breakdown (waterfall: only the first reason per person) --")
    tot = 0
    for r in WATERFALL:
        if r in s["waterfall"]:
            p("  %-16s %4d   %s" % (r, s["waterfall"][r], REASON_JA[r]))
            tot += s["waterfall"][r]
    p("  %-16s %4d" % ("(total)", tot))
    p("")
    p("-- exclusion breakdown (independent: one person can match several) --")
    for r in WATERFALL:
        if r in s["independent_fails"]:
            p("  %-16s %4d" % (r, s["independent_fails"][r]))
    p("")
    p("-- missing counts for the questions used in the decision (if non-zero, suspect the column mapping and the required settings) --")
    for q, c in s["missing_by_question"].items():
        if c:
            p("  %-14s %4d" % (q, c))
    if not any(s["missing_by_question"].values()):
        p("  (nothing missing)")
    p("")
    p("-- correct answers on the knowledge questions (out of %d respondents) --" % s["n_unique"])
    for q, a in KNOW_ANSWERS.items():
        c = s["knowledge_correct"][q]
        p("  %-8s correct %-6s %4d  (%.1f%%)"
          % (q, a, c, 100.0 * c / s["n_unique"] if s["n_unique"] else 0.0))
    p("")
    p("-- misreporting indicators --")
    p("  picked at least one bogus item     : %d" % s["n_bogus_any"])
    for k, v in s["bogus_by_item"].items():
        p("    %-28s : %d  (%s)" % (k, v, BOGUS[k]))
    p("  of those, self-report 5+ years     : %d  <- contradiction" % s["n_bogus_experienced"])
    p("  band q_prod vs whole-number q_prod_years contradiction : %d" % s["n_inconsistent_years"])
    p("  failed the attention check         : %d%s"
      % (s["n_attention_fail"],
         "" if s["strict_attention"] else "  (not used to exclude by default)"))
    if s["median_duration_sec"]:
        p("  median completion time             : %.0f s / speeders(<0.4x median): %d"
          % (s["median_duration_sec"], s["n_speeders"]))
    p("")
    p("-- attribute distribution of the eligible (q_device is used as a covariate in the analysis) --")
    for q in ("q_device", "q_prod", "q_prev", "s_daw", "s_monitor"):
        d = s["dist_eligible"][q]
        order = [c for c, _ in SPEC[q].options if c in d]
        order += [c for c in d if c not in order]
        p("  %-12s %s" % (q, "  ".join("%s=%d" % (c, d[c]) for c in order)))
    p("")
    p("-- draft text for the paper's Methods --")
    for line in methods_sentence(s).split(". "):
        if line:
            p("  " + line.strip() + ("" if line.endswith(".") else "."))
    p("=" * 68)


# --------------------------------------------------------------------------
# consistency check against app.js
# --------------------------------------------------------------------------
def extract_appjs_options(path):
    src = open(path, encoding="utf-8").read()
    starts = [(m.group(1), m.start()) for m in
              re.finditer(r'selectQ\(\s*"([a-z0-9_]+)"', src)]
    out = OrderedDict()
    for i, (qid, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(src)
        block = src[pos:end]
        pairs = re.findall(r'\[\s*"([a-z0-9_]+)"\s*,\s*"([^"]*)"\s*\]', block)
        out.setdefault(qid, []).extend([c for c, _ in pairs])
    return out


def check_app_js(path):
    if not os.path.exists(path):
        print("app.js not found: %s" % path, file=sys.stderr)
        return 2
    app = extract_appjs_options(path)
    shared = [q for q in SPEC if SPEC[q].shared_with_app]
    print("app.js: %s" % path)
    print("questions extracted from app.js: %s" % ", ".join(app.keys()))
    print("")
    ndiff = 0
    for qid in shared:
        mine = [c for c, _ in SPEC[qid].options]
        theirs = app.get(qid)
        if theirs is None:
            print("  [MISSING] %-8s does not exist in app.js" % qid)
            ndiff += 1
            continue
        if mine == theirs:
            print("  [MATCH] %-8s %s" % (qid, ",".join(mine)))
        else:
            print("  [DIFF] %-8s" % qid)
            print("         SCREENER: %s" % ",".join(mine))
            print("         app.js  : %s" % ",".join(theirs))
            print("         SCREENER only: %s / app.js only: %s"
                  % (sorted(set(mine) - set(theirs)) or "-",
                     sorted(set(theirs) - set(mine)) or "-"))
            ndiff += 1
    for qid in SPEC:
        if not SPEC[qid].shared_with_app and qid in app:
            print("  [NOTE] %-8s not treated as shared in SCREENER but present in app.js"
                  % qid)
    # Check whether questions retired on  are still left in the SPA.
    # Even if they are, the decision does not break (they are not in
    # APP_REQUIRED, so they are ignored), but participants are being shown
    # pointless questions, so always surface it as a missed deletion.
    still = [q for q in RETIRED_QIDS if q in app]
    if still:
        print("")
        print("  [retired questions still present in app.js] %s" % ", ".join(still))
        print("     Questions removed from the questionnaire on . Remove them "
              "from SCREENS.questionnaire in app/assets/app.js.")
        print("     The decision side does not include them in APP_REQUIRED, so even if "
              "they remain the eligibility decision does not break (the answers are "
              "ignored), but participants see pointless questions.")
    print("")
    print("questions with a difference: %d (only the option codes are compared. The "
          "question text is not compared, so check the q_prod wording with "
          "check_criteria_drift.py and app/test/flow_test.mjs)" % ndiff)
    # : made it **return an exit code**. Previously it always returned
    # 0 even with differences or leftover retired questions, so check_all.py
    # (which only looks at the returncode) sailed straight past it = it was not
    # a check at all.
    if ndiff or still:
        print("NG: %d option-code differences / %d retired questions still present"
              % (ndiff, len(still)))
        return 1
    print("OK: the option codes of SPEC and app.js match. No retired questions remain.")
    return 0


def print_spec():
    sec = {"0": "Section 0 - Playback device & hearing"
                " (both are criteria. Moved over from the Prolific prescreeners)",
           "A": "Section A - Everyday media habits",
           "B": "Section B - Creative and technical activities",
           "C": "Section C - If you work with multitrack recordings",
           "D": "Section D - Audio terminology",
           "E": "Section E - Previous listening tests (record-only)"}
    cur = None
    for qid, q in SPEC.items():
        if q.section != cur:
            cur = q.section
            print("\n" + "=" * 68)
            print(sec[cur])
            print("=" * 68)
        tag = {"eligibility": "criterion", "knowledge": "knowledge question",
               "bogus": "bogus", "attention": "attention check",
               "distractor": "distractor", "record": "record-only"}[q.role]
        shared = " / shared with app.js" if q.shared_with_app else ""
        print("\n[%s]  (%s, %s%s)" % (qid, q.kind, tag, shared))
        print("  Q: %s" % q.label)
        for code, label in q.options:
            mark = ""
            if KNOW_ANSWERS.get(qid) == code:
                mark = "   <- correct"
            if BOGUS.get(qid) == code:
                mark = "   <- BOGUS(fictitious)"
            if ATTN_ANSWER[0] == qid and ATTN_ANSWER[1] == code:
                mark = "   <- correct"
            print("     %-14s %s%s" % (code, label, mark))
        if q.note:
            print("  note: %s" % q.note)
    print("\n" + "=" * 68)
    print("eligibility criteria (criteria %s):" % CRITERIA_VERSION)
    print("  q_hearing IN %s" % str(OK_HEARING))
    print("  AND q_device IN %s" % str(OK_DEVICE))
    print("  AND q_prod IN %s AND q_prod_years >= %d"
          % (str(OK_PROD), MIN_PROD_YEARS))
    print("  AND (number of knowledge questions correct) >= %d" % MIN_KNOW_CORRECT)
    print("  AND (number of bogus items picked) = 0  NOTE: in the SPA only q_know1 = %s applies"
          % BOGUS["q_know1"])
    print("")
    print("  retired questions : %s" % ", ".join(RETIRED_QIDS))
    print("    q_recent was a criterion but was removed (OK_RECENT = %s)"
          % ("None" if OK_RECENT is None else str(OK_RECENT)))
    print("    q_mixes had already been dropped from the criteria (OK_MIXES = %s)"
          % ("None" if OK_MIXES is None else str(OK_MIXES)))
    print("    q_instr / q_paid were record-only")
    print("  q_prod_years (whole years) is not used to exclude; it is only aggregated as a covariate")
    return 0


# --------------------------------------------------------------------------
# self-verification
# --------------------------------------------------------------------------
def _row(**kw):
    """Build one synthetic response row in label form (eligible by default).

    The questions retired on  (q_instr / q_mixes / q_paid / q_recent)
    were taken out of base. The tests do pass them via kw, but since they are
    not in SPEC they have no effect at all on the decision (this is the
    regression test for "retiring them does not change the decision").
    """
    base = {
        "q_device": "Over-ear / on-ear headphones",
        "q_hearing": "Normal",
        "s_listen": "5 to 15 hours",
        "s_device_habit": "Over-ear or on-ear headphones",
        "s_podcast": "Weekly", "s_video": "Rarely", "s_attn": "Monthly",
        "s_roles": "Composed or produced original music, Mixed multitrack recordings",
        "q_prod": "5 to under 10 years", "q_prod_years": "7",
        "s_daw": "Reaper",
        "s_monitor": "Studio monitors",
        "s_bogus": "A parametric EQ, A bus compressor, A de-esser",
        "q_know1": "Ratio", "q_know2": "Via a send to an aux/return bus",
        "q_know3": "LUFS", "q_prev": "None",
    }
    base.update(kw)
    return base


def self_test():
    fails = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s" % name)
        else:
            print("  FAIL  %s   %s" % (name, detail))
            fails.append(name)

    print("[1] internal consistency of the spec")
    # , / ; are rejected **only for multi-select**, because parse_value uses
    # them as separators only in the multi case. A comma inside a single-select
    # label is normal (e.g. q_hearing "Slight difficulties, uncorrected" is the
    # real text from app.js), and rejecting it here would make matching app.js
    # impossible.
    for qid, q in SPEC.items():
        if q.kind != "multi":
            continue
        for code, label in q.options:
            if "," in label or ";" in label:
                fails.append("label-separator:%s" % qid)
    check("multi-select option labels contain no , / ;", not fails)
    try:
        {qid: _value_index(q) for qid, q in SPEC.items() if q.options}
        check("normalized option keys do not collide within a question", True)
    except ValueError as e:
        check("normalized option keys do not collide within a question", False, str(e))
    for qid, ans in list(KNOW_ANSWERS.items()) + [ATTN_ANSWER]:
        codes = [c for c, _ in SPEC[qid].options]
        check("correct answer %s is among the options of %s" % (ans, qid), ans in codes)
    for qid, code in BOGUS.items():
        check("bogus %s is among the options of %s" % (code, qid),
              code in [c for c, _ in SPEC[qid].options])

    print("[2] decision logic (synthetic data)")
    cases = [
        ("eligible (label form)", _row(), True, None),
        ("eligible (code form)",
         _row(q_device="over_ear", q_prod="y5_10",
              q_know1="ratio", q_know2="send", q_know3="lufs",
              s_bogus="eq,comp"), True, None),
        ("eligible (knowledge 2/3, 1 skipped)", _row(q_know3="I don't know / prefer not to answer"),
         True, None),
        # Retired questions have no effect even when passed via kw (they are not in SPEC, so they are ignored)
        ("eligible (no paid experience = q_paid was retired entirely)", _row(q_paid="No"), True, None),
        ("eligible (hearing normal)", _row(q_hearing="Normal"), True, None),
        ("eligible (hearing normal, code form)", _row(q_hearing="normal"), True, None),
        # --- playback device (added as a criterion on  -----------
        ("eligible: wired earphones (in_ear)",
         _row(q_device="In-ear earphones / earbuds (wired)"), True, None),
        ("eligible: true wireless (tws; the consent form permits wireless. Recorded as a covariate)",
         _row(q_device="True-wireless earbuds (e.g. AirPods)"), True, None),
        ("eligible: device in code form (tws)", _row(q_device="tws"), True, None),
        ("ineligible: loudspeakers (speakers)",
         _row(q_device="Loudspeakers (computer / external)"), False,
         "fail_device"),
        ("ineligible: other (other = an answer not confirmed to be headphones)",
         _row(q_device="Other"), False, "fail_device"),
        ("ineligible: device unanswered (cannot decide)", _row(q_device=""), False,
         "missing_data"),
        ("ineligible: device value uninterpretable", _row(q_device="my hi-fi I guess"), False,
         "missing_data"),
        ("ineligible: hearing slight (mild, uncorrected)",
         _row(q_hearing="Slight difficulties, uncorrected"), False, "fail_hearing"),
        ("ineligible: hearing aided (hearing aid / cochlear implant)",
         _row(q_hearing="Corrected (hearing aid)"), False, "fail_hearing"),
        ("ineligible: hearing unsure (does not know)", _row(q_hearing="Not sure"),
         False, "fail_hearing"),
        ("ineligible: hearing unanswered (cannot decide)", _row(q_hearing=""), False,
         "missing_data"),
        ("ineligible: hearing value uninterpretable", _row(q_hearing="pretty good I guess"),
         False, "missing_data"),
        ("eligible: 1-2 years of experience (relaxed on  to allow anything but None)",
         _row(q_prod="1 to under 2 years", q_prod_years="1"), True, None),
        ("ineligible: no experience (None)",
         _row(q_prod="None", q_prod_years="0"), False, "fail_years"),
        # A band/whole-number contradiction is **only recorded, never used to
        # exclude** (so that honest respondents who rounded at a boundary are
        # not rejected). The inconsistent_years flag remains.
        ("eligible but flagged inconsistent (gt10 yet whole number 1)",
         _row(q_prod="10 years or more", q_prod_years="1"), True, None),
        ("eligible but flagged inconsistent (lt1 yet whole number 12)",
         _row(q_prod="Less than 1 year", q_prod_years="12"), True, None),
        # Under the old wording "music production / mixing experience", someone
        # who only composes or programs could pick gt10 and still pass with the
        # whole number 0 (no multitrack experience). The q_recent="never"
        # exclusion happened to plug that hole, but deleting that question
        # exposes it.
        # Narrowing the wording to "mixing multitrack recordings" makes this
        # combination **contradictory in the meaning of the question**. The
        # decision does not reject it (to protect honest respondents who rounded
        # at a boundary); it is marked with inconsistent_years. Checked in [3].
        ("eligible but flagged inconsistent (the 'composer only' case the old wording let through: gt10 yet whole number 0)",
         _row(q_prod="10 years or more", q_prod_years="0"), True, None),
        # A "composer only" person answering the new wording honestly picks none, so they are rejected here.
        ("ineligible: a composer-only person answering the new wording honestly (none)",
         _row(q_prod="None", q_prod_years="0"), False, "fail_years"),
        # The number of completed songs was dropped from the criteria on
        #  and the question was retired the same day.
        # The next 2 cases are the regression test for "no answer has any effect".
        ("eligible: 3-9 completed songs (q_mixes is retired)", _row(q_mixes="3-9"), True, None),
        ("eligible: even 1-2 completed songs pass (q_mixes is retired)",
         _row(q_mixes="1-2"), True, None),
        # q_recent was retired too, so previously ineligible answers no longer fail (user decision).
        ("eligible: last mixed 1-2 years ago (q_recent is retired)",
         _row(q_recent="1-2 years ago"), True, None),
        ("eligible: even more than 2 years ago passes (because q_recent was retired)",
         _row(q_recent="More than 2 years ago"), True, None),
        ("eligible: answering that they have never mixed does not fail on q_recent "
         "(what rejects them is q_prod = none)", _row(q_recent="Never"), True, None),
        ("ineligible: knowledge 1/3", _row(q_know2="One insert on the master bus",
                                  q_know3="dBFS peak"), False, "fail_knowledge"),
        ("ineligible: bogus (q_know1 = Harmonic gate)", _row(q_know1="Harmonic gate"),
         False, "fail_bogus"),
        ("ineligible: bogus (fictitious product in s_bogus)",
         _row(s_bogus="A parametric EQ; A Verrian VX-7 stereo aligner"),
         False, "fail_bogus"),
        ("ineligible: a required item is empty (hearing)", _row(q_hearing=""), False, "missing_data"),
        # The old version ran this check with q_recent="whenever I feel like it",
        # but since that question was retired it was moved onto the **required**
        # item q_prod (an uninterpretable value in a retired question no longer
        # counts as missing).
        ("ineligible: uninterpretable value", _row(q_prod="on and off for ages"), False,
         "missing_data"),
        ("eligible: an uninterpretable value in a retired question is ignored",
         _row(q_recent="whenever I feel like it"), True, None),
        ("ineligible: non-numeric years", _row(q_prod_years="a while"), False,
         "missing_data"),
    ]
    for name, row, want_elig, want_reason in cases:
        vals = {q: parse_value(q, row.get(q))[0] for q in SPEC}
        v = classify(vals)
        ok = (v["eligible"] == want_elig
              and (want_reason is None or v["reason"] == want_reason))
        check(name, ok, "eligible=%s reason=%s (expected %s/%s)"
              % (v["eligible"], v["reason"], want_elig, want_reason))

    print("[3] individual flags")
    v = classify({q: parse_value(q, _row(q_prod="10 years or more",
                                         q_prod_years="1").get(q))[0]
                  for q in SPEC})
    check("the inconsistent_years flag is raised", v["inconsistent_years"])
    # The combination the old wording let a "composer only" person through with
    # (gt10 yet whole number 0). After narrowing the wording, the band and the
    # whole number measure **the same quantity**, so this is always marked as a
    # contradiction.
    v = classify({q: parse_value(q, _row(q_prod="10 years or more",
                                         q_prod_years="0").get(q))[0]
                  for q in SPEC})
    check("an answer the old wording let through (gt10 / whole number 0) is eligible but flagged inconsistent",
          v["eligible"] and v["inconsistent_years"])
    # tier-2 (the contingency that picks up people rejected on song count alone)
    # was disabled on  when song count stopped being a criterion
    # (TIER2_MIXES = ()). Check that nobody matches.
    v = classify({q: parse_value(q, _row(q_mixes="1-2").get(q))[0] for q in SPEC})
    check("tier-2 is disabled (nobody matches, since song count is no longer a condition)",
          not v["tier2"] and v["eligible"])
    r = _row(s_attn="Weekly")
    vals = {q: parse_value(q, r.get(q))[0] for q in SPEC}
    check("failing attention leaves them eligible by default", classify(vals)["eligible"])
    check("ineligible under --strict-attention",
          classify(vals, strict_attention=True)["reason"] == "fail_attention")
    codes = parse_value("s_bogus", " A parametric EQ ;A de-esser,\nA bus compressor")[0]
    check("multi-select separators (, ; newline space) are absorbed", codes == ["eq", "deesser", "comp"],
          str(codes))
    _, unk = parse_value("s_bogus", "A parametric EQ, A Klingon phase inverter")
    check("unknown multi-select tokens are reported", unk == ["A Klingon phase inverter"], str(unk))

    print("[4] CSV pipeline (end to end)")
    # 4 ineligible cases (hearing / bogus / no experience / device is speakers)
    # and 1 eligible case + 1 duplicate row.
    # Row 5 was q_recent="More than 2 years ago" until ; since that
    # question was retired, the same "fails on exactly one item" role was moved
    # to the new criterion q_device.
    rows_src = [_row(), _row(q_hearing="Not sure"), _row(q_know1="Harmonic gate"),
                _row(q_prod="None", q_prod_years="0"),
                _row(q_device="Loudspeakers (computer / external)"), _row()]
    hdr = ["Participant id"] + list(SPEC.keys()) + ["Time taken"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(hdr)
    for i, r in enumerate(rows_src):
        # Row 6 reuses the ID of row 1 to exercise duplicate detection
        pid = "PID%03d" % (0 if i == 5 else i)
        w.writerow([pid] + [r.get(q, "") for q in SPEC] + [str(200 + 10 * i)])
    buf.seek(0)
    rows, meta = load_rows(buf)
    check("columns for every question resolve from the CSV", len(meta["columns"]) == len(SPEC))
    check("the participant ID column is detected", meta["id_col"] == "Participant id")
    check("the completion time column is detected", meta["duration_col"] == "Time taken")
    s = summarize(rows)
    check("1 duplicate ID detected", s["n_duplicate_ids"] == 1, str(s["n_duplicate_ids"]))
    check("5 respondents, 1 eligible",
          s["n_unique"] == 5 and s["n_eligible"] == 1,
          "n=%d elig=%d" % (s["n_unique"], s["n_eligible"]))
    check("each of the 4 active criteria rejects exactly 1 person",
          all(s["waterfall"].get(k, 0) >= 1
              for k in ("fail_hearing", "fail_device", "fail_years",
                        "fail_bogus")),
          str(s["waterfall"]))
    check("nobody is rejected by the retired conditions (fail_songs / fail_recency)",
          not s["waterfall"].get("fail_songs")
          and not s["waterfall"].get("fail_recency"), str(s["waterfall"]))
    check("waterfall total = number of ineligible respondents",
          sum(s["waterfall"].values()) == s["n_unique"] - s["n_eligible"])
    check("1 person detected as bogus", s["n_bogus_any"] == 1)
    check("tier-2 is 0 people (disabled)", s["n_tier2"] == 0)
    check("the eligible IDs can be extracted", s["eligible_ids"] == ["PID000"],
          str(s["eligible_ids"]))
    check("the Methods sentence is generated", "95% CI" in methods_sentence(s))

    print("[5] fallbacks in column-name resolution")
    hdr2 = ["Prolific PID"] + [SPEC[q].label for q in SPEC]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(hdr2)
    r = _row()
    w.writerow(["PIDX"] + [r.get(q, "") for q in SPEC])
    buf.seek(0)
    rows, meta = load_rows(buf)
    check("question text also resolves as a column name (Google Forms style)",
          len(meta["columns"]) == len(SPEC))
    s2 = summarize(rows)
    check("eligibility can be decided in the Forms style too", s2["n_eligible"] == 1)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Participant id", "q_prod", "q_hearing"])
    w.writerow(["PIDY", "y5_10", "normal"])
    buf.seek(0)
    try:
        load_rows(buf)
        check("a CSV with missing columns stops with an explicit error", False, "no exception was raised")
    except SystemExit as e:
        check("a CSV with missing columns stops with an explicit error", "unresolved question ID" in str(e))

    # If the column name for q_prod is renamed, rule 3 (token containment) makes
    # q_prod grab the q_prod_years column and everyone silently became
    # missing_data.
    hdr3 = ["Participant id"] + [("mixing_experience_band" if q == "q_prod" else q)
                                 for q in SPEC]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(hdr3)
    r = _row()
    w.writerow(["PIDZ"] + [r.get(q, "") for q in SPEC])
    buf.seek(0)
    try:
        load_rows(buf)
        check("a q_prod / q_prod_years column collision is detected and stops the run", False, "no exception was raised")
    except SystemExit as e:
        check("a q_prod / q_prod_years column collision is detected and stops the run",
              "the same column" in str(e), str(e).splitlines()[0])

    print("[6] rows with a blank participant ID")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Participant id"] + list(SPEC.keys()))
    for pid in ("", "", "PIDA"):
        r = _row()
        w.writerow([pid] + [r.get(q, "") for q in SPEC])
    buf.seek(0)
    rows, _ = load_rows(buf)
    s3 = summarize(rows)
    check("the number of blank IDs is reported", s3["n_blank_participant_id"] == 2,
          str(s3["n_blank_participant_id"]))
    check("blank IDs do not appear in the eligible list", s3["eligible_ids"] == ["PIDA"],
          str(s3["eligible_ids"]))
    check("the number of eligible respondents with a blank ID is reported separately", s3["n_eligible_without_id"] == 2,
          str(s3["n_eligible_without_id"]))

    print("[7] missing counts per question")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Participant id"] + list(SPEC.keys()))
    for i in range(3):
        r = _row(s_bogus="")          # the case of submitting the multi-select unanswered
        w.writerow(["PIDB%d" % i] + [r.get(q, "") for q in SPEC])
    buf.seek(0)
    rows, _ = load_rows(buf)
    s4 = summarize(rows)
    check("an unanswered s_bogus shows up in missing_by_question",
          s4["missing_by_question"]["s_bogus"] == 3,
          str(s4["missing_by_question"]))
    check("an unanswered s_bogus is ineligible via missing_data",
          s4["n_eligible"] == 0 and s4["waterfall"].get("missing_data") == 3)

    print("[8] eligibility for the SPA questionnaire (classify_app_questionnaire / the source of truth for the server-side API)")

    def _app(**kw):
        # Of the 8 items the SPA sends after the  reduction, the 7
        # used in the decision. q_prev is record-only so it is not included (it
        # would be ignored anyway).
        base = {"q_device": "over_ear", "q_hearing": "normal",
                "q_prod": "y5_10", "q_prod_years": 7,
                "q_know1": "ratio", "q_know2": "send", "q_know3": "lufs"}
        base.update(kw)
        return base

    app_cases = [
        ("eligible (code form)", _app(), True, None),
        ("eligible (label form)",
         _app(q_device="Over-ear / on-ear headphones",
              q_prod="5 to under 10 years", q_know1="Ratio",
              q_know2="Via a send to an aux/return bus", q_know3="LUFS"),
         True, None),
        # --- playback device (added as a criterion on  -----------
        ("eligible: in_ear", _app(q_device="in_ear"), True, None),
        ("eligible: tws (the consent form permits wireless. Recorded as a covariate)",
         _app(q_device="tws"), True, None),
        ("ineligible: speakers (loudspeakers do not meet the requirement)",
         _app(q_device="speakers"), False, "fail_device"),
        ("ineligible: other (an answer not confirmed to be headphones)",
         _app(q_device="other"), False, "fail_device"),
        ("ineligible: device unanswered (cannot decide)", _app(q_device=""), False,
         "missing_data"),
        ("ineligible: device value uninterpretable", _app(q_device="AirPods Max"), False,
         "missing_data"),
        ("ineligible: the device key is absent (a POST from an old SPA)",
         {"q_hearing": "normal", "q_prod": "y5_10", "q_prod_years": 7,
          "q_know1": "ratio", "q_know2": "send", "q_know3": "lufs"},
         False, "missing_data"),
        ("eligible (knowledge 2/3, q_know3 is an app.js distractor)",
         _app(q_know3="rms_db"), True, None),
        ("eligible (knowledge 2/3, skip)", _app(q_know3="skip"), True, None),
        ("eligible (boundary: exactly 2 years)", _app(q_prod="y2_3", q_prod_years=2),
         True, None),
        ("eligible (hearing normal, label form)", _app(q_hearing="Normal"), True, None),
        ("ineligible: hearing slight", _app(q_hearing="slight"), False, "fail_hearing"),
        ("ineligible: hearing aided (hearing aid / cochlear implant)", _app(q_hearing="aided"),
         False, "fail_hearing"),
        ("ineligible: hearing unsure", _app(q_hearing="unsure"), False, "fail_hearing"),
        ("ineligible: hearing unanswered (cannot decide)", _app(q_hearing=""), False,
         "missing_data"),
        ("eligible: 1-2 years of experience (relaxed on  to allow anything but None)",
         _app(q_prod="y1_2", q_prod_years=1), True, None),
        ("ineligible: no experience (None)", _app(q_prod="none", q_prod_years=0),
         False, "fail_years"),
        # A band/whole-number contradiction is **only recorded, never used to
        # exclude** (after the  relaxation), because using it to
        # exclude would reject honest respondents who rounded at a boundary.
        # The inconsistent_years flag remains and appears in the report and in
        # the paper's participant-flow section.
        ("eligible but flagged inconsistent (gt10 yet whole number 1)",
         _app(q_prod="gt10", q_prod_years=1), True, None),
        # Under the old wording "music production / mixing", someone who only
        # composes could pick gt10 and pass (whole number 0 still passes because
        # MIN_PROD_YEARS = 0). q_recent="never" happened to plug that hole, but
        # retiring that question exposes it.
        # Narrowing the wording to "years mixing multitrack" makes this
        # combination contradictory in the meaning of the question, so it is
        # always marked with inconsistent_years.
        ("eligible but flagged inconsistent (a composer-only person the old wording let through: gt10 yet whole number 0)",
         _app(q_prod="gt10", q_prod_years=0), True, None),
        # The next 4 cases are the regression test for "however you answer a retired question, it has no effect".
        ("eligible: 3-9 completed songs (q_mixes was retired on ",
         _app(q_mixes="n3_9"), True, None),
        ("eligible: even 1-2 completed songs pass (q_mixes is retired)",
         _app(q_mixes="n1_2"), True, None),
        ("eligible: last mixed 1-2 years ago (q_recent is retired)",
         _app(q_recent="y1_2"), True, None),
        ("eligible: even more than 2 years ago passes (because q_recent was retired)",
         _app(q_recent="gt2y"), True, None),
        ("eligible: answering that they have never mixed does not fail on q_recent "
         "(what rejects them is q_prod = none)", _app(q_recent="never"), True, None),
        ("ineligible: knowledge 1/3", _app(q_know2="insert_master", q_know3="hz"),
         False, "fail_knowledge"),
        ("ineligible: bogus (harmonic_gate)", _app(q_know1="harmonic_gate"),
         False, "fail_bogus"),
        ("ineligible: a required item is empty (hearing)", _app(q_hearing=""), False, "missing_data"),
        ("ineligible: non-numeric years", _app(q_prod_years="a while"), False,
         "missing_data"),
        ("ineligible: answers is not a dict", None, False, "missing_data"),
    ]
    for name, ans, want_elig, want_reason in app_cases:
        v = classify_app_questionnaire(ans)
        ok = (v["eligible"] == want_elig
              and (want_reason is None or v["reason"] == want_reason))
        check(name, ok, "eligible=%s reason=%s (expected %s/%s)"
              % (v["eligible"], v["reason"], want_elig, want_reason))
    check("the bogus decision does not misfire when the s_bogus question is absent",
          classify_app_questionnaire(_app())["bogus_hits"] == [])
    check("criteria_version is returned",
          classify_app_questionnaire(_app())["criteria_version"] == CRITERIA_VERSION)

    print("[9] whether the app.js option codes resolve in the decision (drift guard)")
    # If this breaks, everyone becomes missing_data = 0 eligible, and it happens
    # silently. Make sure this check catches any change to the app.js options.
    if not os.path.exists(APP_JS):
        check("app.js is found", False, APP_JS)
    else:
        appq = extract_appjs_options(APP_JS)
        # : q_mixes / q_recent were removed from SPEC, so they cannot
        # be passed to parse_value (it would raise KeyError). We look at the new
        # criterion q_device instead.
        for qid in ("q_device", "q_hearing", "q_prod", "q_know1",
                    "q_know2", "q_know3"):
            codes = appq.get(qid) or []
            bad = [c for c in codes if parse_value(qid, c)[0] != c]
            check("every code of %s in app.js resolves" % qid,
                  bool(codes) and not bad, "unresolvable: %s" % bad)
        k3 = appq.get("q_know3") or []
        check("q_know3 in app.js contains the correct code %s" % KNOW_ANSWERS["q_know3"],
              KNOW_ANSWERS["q_know3"] in k3, str(k3))
        # If the SPEC and app.js options differ by even one character, the
        # decision side cannot resolve the answers and "everyone scores 0 on the
        # knowledge questions" becomes possible. Check identity directly.
        check("the q_know3 options in app.js match SPEC exactly",
              k3 == [c for c, _ in SPEC["q_know3"].options],
              "SPEC=%s app.js=%s" % ([c for c, _ in SPEC["q_know3"].options], k3))
        check("q_know1 in app.js contains the bogus code %s" % BOGUS["q_know1"],
              BOGUS["q_know1"] in (appq.get("q_know1") or []))
        # Hearing was promoted to a criterion on . app.js is already
        # deployed, so we align in the direction of **fitting SPEC to app.js**.
        # If this drifts, parse_value cannot resolve the answers of eligible
        # participants and everyone becomes missing_data = 0 eligible.
        h = appq.get("q_hearing") or []
        check("the q_hearing options in app.js match SPEC exactly",
              h == [c for c, _ in SPEC["q_hearing"].options],
              "SPEC=%s app.js=%s" % ([c for c, _ in SPEC["q_hearing"].options], h))
        check("q_hearing in app.js contains the eligible code %s" % str(OK_HEARING),
              all(c in h for c in OK_HEARING), str(h))
        # q_device was **promoted to a criterion** on  (previously:
        # record-only). Forgetting to list it as a required item rejects nobody,
        # and drifting options make everyone missing_data, so both directions
        # are checked.
        d = appq.get("q_device") or []
        check("q_device is included in the criteria",
              "q_device" in SPEC and "q_device" in REQUIRED_FOR_JUDGEMENT
              and "q_device" in APP_REQUIRED)
        check("the q_device options in app.js match SPEC exactly",
              d == [c for c, _ in SPEC["q_device"].options],
              "SPEC=%s app.js=%s" % ([c for c, _ in SPEC["q_device"].options], d))
        check("q_device in app.js contains the eligible codes %s" % str(OK_DEVICE),
              all(c in d for c in OK_DEVICE), str(d))
        check("speakers / other exist in app.js but are not eligible",
              all(c in d for c in ("speakers", "other"))
              and not any(c in OK_DEVICE for c in ("speakers", "other")),
              "app.js=%s OK_DEVICE=%s" % (d, OK_DEVICE))
        # Check that no retired question is left on the decision side (if one
        # were, everyone would become missing_data the moment the SPA stops
        # sending it). Leaving them on the app.js side is acceptable.
        for qid in RETIRED_QIDS:
            check("the retired question %s is not used in the decision" % qid,
                  qid not in SPEC and qid not in REQUIRED_FOR_JUDGEMENT
                  and qid not in APP_REQUIRED)
        check("the retired set conditions (OK_MIXES / OK_RECENT) are None",
              OK_MIXES is None and OK_RECENT is None)

    print("")
    if fails:
        print("SELF-TEST FAILED: %d checks -> %s" % (len(fails), fails))
        return 1
    print("SELF-TEST PASSED")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="decide eligibility from the screening survey CSV "
                    "(spec: internal notes)")
    ap.add_argument("--csv", help="the Prolific / Google Forms response CSV")
    ap.add_argument("--map", help="JSON mapping question ID -> CSV column name")
    ap.add_argument("--id-col", help="explicitly specify the participant ID column name")
    ap.add_argument("--duration-col", help="explicitly specify the completion time column name")
    ap.add_argument("--out-ids", help="output path for the eligible IDs (one ID per line)")
    ap.add_argument("--out-tier2-ids", help="output path for the tier-2 IDs (only with --tier2)")
    ap.add_argument("--out-json", help="output path for the verdict JSON of all respondents")
    ap.add_argument("--strict-attention", action="store_true",
                    help="include the attention check as a required condition (decide this before launching)")
    ap.add_argument("--tier2", action="store_true",
                    help="output tier-2 from the pre-registered contingency")
    ap.add_argument("--target-analysed", type=int, default=20,
                    help="the number of experts L required for the analysis (default 20)")
    ap.add_argument("--target-eligible", type=int, default=None,
                    help="specify the required number of eligible respondents directly (the 3 rates below are not used when given)")
    ap.add_argument("--uptake", type=float, default=0.80,
                    help="rate of eligible -> starting the main study (default 0.80, an unmeasured assumption)")
    ap.add_argument("--hp-pass", type=float, default=0.75,
                    help="pass rate of the headphone / anti-phase test at the start of the main study "
                         "(default 0.75 = expert assumption. The general-population assumption is 0.60. An unmeasured assumption)")
    ap.add_argument("--post-screening", type=float, default=0.85,
                    help="rate of completers surviving the post-hoc exclusion criteria (default 0.85, an unmeasured assumption)")
    ap.add_argument("--reward", type=float, default=0.80,
                    help="reward per screener respondent in GBP (used to estimate the extra cost)")
    ap.add_argument("--print-spec", action="store_true", help="print the question spec")
    ap.add_argument("--check-app-js", nargs="?", const=APP_JS, default=None,
                    help="cross-check the question IDs and option codes against the main study's app.js")
    ap.add_argument("--self-test", action="store_true", help="self-verify on synthetic data")
    args = ap.parse_args()

    if args.print_spec:
        return print_spec()
    if args.check_app_js:
        return check_app_js(args.check_app_js)
    if args.self_test:
        return self_test()
    if not args.csv:
        ap.print_help()
        return 2

    colmap = json.load(open(args.map, encoding="utf-8")) if args.map else None
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows, meta = load_rows(f, colmap, args.id_col, args.duration_col)
    s = summarize(rows, args.strict_attention)
    print_report(s, args.csv)

    unknown_rows = [(r["participant_id"], r["unknown"]) for r in rows
                    if r["unknown"]]
    if unknown_rows:
        print("")
        print("!! there are answer values that could not be interpreted (treated as missing; check the column mapping):")
        for pid, unk in unknown_rows[:20]:
            print("   %s  %s" % (pid, unk))
        if len(unknown_rows) > 20:
            print("   ... %d more" % (len(unknown_rows) - 20))

    if args.out_ids:
        with open(args.out_ids, "w", encoding="utf-8") as f:
            f.write("\n".join(s["eligible_ids"]) + ("\n" if s["eligible_ids"] else ""))
        print("\nwrote the eligible IDs: %s (%d IDs)"
              % (args.out_ids, len(s["eligible_ids"])))
        print("  NOTE pseudonymised personal data. Do not place it in a public location.")
    if args.tier2 and args.out_tier2_ids:
        with open(args.out_tier2_ids, "w", encoding="utf-8") as f:
            f.write("\n".join(s["tier2_ids"]) + ("\n" if s["tier2_ids"] else ""))
        print("wrote the tier-2 IDs: %s (%d IDs) NOTE do not mix them with the primary list"
              % (args.out_tier2_ids, len(s["tier2_ids"])))
    elif args.tier2:
        print("\n--tier2 was given but --out-tier2-ids is missing, so nothing is written")

    if args.out_json:
        rec = []
        for r in s["_unique_rows"]:
            v = r["verdict"]
            rec.append({"participant_id": r["participant_id"],
                        "eligible": v["eligible"], "tier2": v["tier2"],
                        "reason": v["reason"], "fails": v["fails"],
                        "n_know_correct": v["n_know_correct"],
                        "bogus_hits": v["bogus_hits"],
                        "inconsistent_years": v["inconsistent_years"],
                        "attention_ok": v["attention_ok"],
                        "duration_sec": r["duration_sec"],
                        "values": r["values"]})
        out = {k: v for k, v in s.items() if not k.startswith("_")}
        out["generated_utc"] = datetime.now(timezone.utc).isoformat()
        out["input_csv"] = os.path.abspath(args.csv)
        out["columns"] = meta["columns"]
        out["methods_sentence"] = methods_sentence(s)
        out["records"] = rec
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("verdict JSON: %s" % args.out_json)

    # Deciding whether to increase the number of places (SCREENER.md §6.2)
    #
    # Fixed : the old version hard-coded need=25, which folded in only
    # uptake 0.80 and dropped the HP/AP pass rate and the post-screening
    # survival rate.
    # This screener plays no audio, so HP/AP lives on the main-study side
    # (= design B). Therefore
    #   required eligible = L / (uptake x HP_AP_pass x post_screening_survival)
    # With the defaults, 20 / (0.80 x 0.75 x 0.85) = 39.2 -> 40.
    # If HP/AP pass is taken as 0.60 it is 49. It can be overridden explicitly
    # with --target-eligible.
    need = args.target_eligible
    if need is None:
        need = math.ceil(args.target_analysed
                         / (args.uptake * args.hp_pass * args.post_screening))
    e, n = s["n_eligible"], s["n_unique"]
    if n:
        print("")
        print("-- deciding whether to increase places (SCREENER.md §6.1-6.2) --")
        if args.target_eligible is None:
            print("   required eligible = %d / (uptake %.2f x HP/AP pass %.2f x "
                  "post-screening %.2f) = %d people"
                  % (args.target_analysed, args.uptake, args.hp_pass,
                     args.post_screening, need))
            print("   NOTE all 3 of these rates are unmeasured assumptions. They can be "
                  "changed with --uptake / --hp-pass / --post-screening.")
        else:
            print("   target: %d eligible people (specified explicitly with --target-eligible)" % need)
        if e >= need:
            print("   %d eligible >= %d. No extra launch needed." % (e, need))
        elif e == 0:
            print("   0 eligible. Consider the §6.3 contingency (switching the recruitment channel).")
        else:
            add = math.ceil((need - e) * n / e)
            print("   measured eligibility rate %.1f%% -> roughly %d extra places needed (cost about GBP %.0f "
                  "@ GBP %.2f/person)"
                  % (100.0 * e / n, add, add * args.reward * 1.333, args.reward))
            if add > 394 - n:
                print("   NOTE the population of 394 (of whom %d have already been served) cannot cover it. "
                      "Trigger the §6.3 contingency." % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
