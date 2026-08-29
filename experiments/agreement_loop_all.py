"""Agreement-reward loop runner over all 150 songs (train100 + dev24 + test26), launch-ready.

Built on agreement_loop_v2.py (single song) and extended into a CLI that can run
multiple splits in chunks. This file is about **code preparation**; GPU jobs are not
launched from here (run it via slurm after the allowlist entry is added; see note).

Design:
  - Target songs = the splits given by --splits (train/dev/test) concatenated in
    order, then cut into a chunk window by --offset / --limit (so a large song set
    can be submitted as several jobs).
  - For each song:
      1. Render & save 4 baselines: dry / P2(analysis_driven) / P6(profile_matched)
         / MEGAMI. Used for comparison on 3 metrics (PQ / SB / downstream KAD).
      2. Run the agreement loop (same eval-gated greedy as v2) and save the proposed mix.
    Everything is saved to outputs/runs/<run>/mixes/<method>__<song_id>.wav
    (matching the naming convention where kad_eval.py takes the method via
    `name.split("__")[0]`).
  - resume: skip songs that already have a per-song result
    (outputs/runs/<run>/songs/<safe_id>.json) with status=="done".
  - partial saving: on each song completion, atomically overwrite the per-song JSON
    and the aggregate index.

To keep unit testing possible on CPU, the pure logic (song list expansion, resume
decision, mix path naming, method sets, etc.) is factored out into module-level
functions, and the scoring that requires a GPU is lazy-imported inside main().
argparse is built by build_arg_parser() so it can be tested.

Real GPU run (after the allowlist entry is added):
  IMAGE_NAME=songbench-image:latest JOB_TIME=06:00:00 \
    scripts/the local job wrapper python experiments/agreement_loop_all.py \
      --splits train --offset 0 --limit 25 --run-name agree_all_v1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# experiments/_common puts ROOT/src on sys.path and configures the HF cache.
from _common import ROOT, enumerate_songs, load_one  # noqa: E402


# Baseline method names (also used as the mixes file prefix). proposed is the
# agreement loop output.
# equal_lufs = naive baseline (lower bound) that just matches every stem to the same
# LUFS and sums them.
BASELINE_METHODS = ("dry", "equal_lufs", "P2", "P6", "MEGAMI")
PROPOSED_METHOD = "proposed"
ALL_METHODS = BASELINE_METHODS + (PROPOSED_METHOD,)

_VALID_SPLITS = ("train", "dev", "test")

# Search reward forms (matching the forms in agreement_reward.py).
#   min      : R = min(z_PQ, z_SB)            Default. Agreement of both metrics (anti-hack).
#   weighted : R = w*z_PQ + (1-w)*z_SB        Weighted average.
#   pq_only  : R = z_PQ                       Audiobox-PQ alone = for reproducing reward hacking.
#                                             SB is held out (scored only, not used for optimization).
#   sb_only  : R = z_SB                       SongBench-Mixing alone.
_REWARD_FORMS = ("min", "weighted", "pq_only", "sb_only")


#: How the scorers are constructed. ``local`` loads the models inside the search
#: process, as before. ``remote`` sends requests over TCP to a shared scoring
#: server (tools/scoring_server.py).
SCORER_MODES = ("local", "remote")

#: Provenance written into the run JSON when remote. For local we **do not write the
#: key at all** (so the JSON of a default run does not change by a single byte; same
#: precedent as render_cache).
SCORING_PROVENANCE: Dict[str, Any] = {}


def build_scorers(mode: str = "local", addr: Optional[str] = None,
                  client_index: int = 0) -> Dict[str, Any]:
    """Build ``{"pq": scorer, "sb": scorer}``.

    ``mode='local'``
        Load the models in-process, as before. **Bit-identical to existing runs.**
    ``mode='remote'``
        Return ``tools/scoring_client.RemoteScorer``. It is a drop-in for
        ``score(audio, sr) -> float``, so the caller does not change by a single
        line. The search process does not import torch, so it **creates no CUDA
        context and uses not one byte of GPU memory**. This lets us pack dozens of
        search processes into a single Slurm job (approved by the design .

    **No fallback.** If the connection fails, raise. Silently dropping back to
    in-process scoring produces "a run where you cannot tell which scorer was used",
    and it keeps holding a GPU while running, so you never notice.

    In remote mode, right after startup we score a 12-second dummy once per scorer to
    check the connection, the fingerprint match, and that both models are alive
    (fail-fast). Failing only after loading one song would throw away several minutes.
    """
    m = (mode or "local").strip().lower()
    if m not in SCORER_MODES:
        raise ValueError(f"unknown scorer mode={mode!r}; valid={SCORER_MODES}")
    if m == "local":
        # Lazy construction of the real models (no proxies allowed).
        from mix_orchestrator.eval.audiobox_pq import AudioboxPQScorer
        from mix_orchestrator.ears.tier6_music_reward import SongBenchMixingEar
        return {"pq": AudioboxPQScorer(), "sb": SongBenchMixingEar()}

    import sys as _sys
    tools_dir = str(Path(ROOT) / "tools")
    if tools_dir not in _sys.path:
        _sys.path.insert(0, tools_dir)
    from scoring_client import RemoteScorer          # noqa: E402

    kw: Dict[str, Any] = {"client_index": int(client_index)}
    if addr:
        kw["servers"] = addr
    scorers = {"pq": RemoteScorer("pq", **kw), "sb": RemoteScorer("sb", **kw)}

    # fail-fast probe: push a 12 s, stereo, non-silent signal through each scorer once.
    probe = np.zeros((2, 12 * 44100), dtype=np.float32)
    probe[0, ::100] = 0.1
    probe[1, ::150] = -0.1
    t0 = time.time()
    p_pq = float(scorers["pq"].score(probe, 44100))
    p_sb = float(scorers["sb"].score(probe, 44100))
    el = time.time() - t0
    print(f"[scorer] mode=remote addr={addr or '(env)'} client_index={client_index} "
          f"probe pq={p_pq:.6f} sb={p_sb:.6f} ({el*1000:.0f}ms)", flush=True)

    SCORING_PROVENANCE.clear()
    SCORING_PROVENANCE.update({
        "mode": "remote",
        "addr": addr or os.environ.get("AUTOMIX_SCORER_ADDR", ""),
        "client_index": int(client_index),
        "probe_pq": p_pq,
        "probe_sb": p_sb,
    })
    return scorers


def _no_sb_in_search(default: bool = False) -> bool:
    """Environment variable ``AUTOMIX_NO_SB_IN_SEARCH``. Returns ``default`` if unset.

    **For reward_form=pq_only, pass default=True** (specified by the caller).
    User feedback : for the instruction "the experiment does not need to
    look at sb" / "I want to run an experiment that references only pq without Sb",
    we had merely set the report weight to 0 while still scoring SongBench on every
    step and still showing it in the prompt. The cause was that the state violating
    the instruction was the **default**, so we flip the default. Only passing 0
    explicitly restores the old behaviour.

    In the random arm the proposer does not look at the prompt, so enabling this
    keeps the PQ trajectory bit-identical (reward=z_PQ never references sb).
    Comparability with the existing exc10k_random_pq is preserved. For the LLM arm
    the prompt changes, so the run is naturally a different one (that is what was
    asked for).

    When enabled, scoring inside the search loop is PQ only (sb=nan), so a whole
    SongBench forward disappears per step. With reward_form=pq_only the weight is 0,
    so reward is unchanged. **The final full-song scoring is left untouched**, so the
    SongBench value used for reporting is still measured once per song (needed to
    refute reward hacking).

    Reads the same env as agreement_loop_qwen.no_sb_in_search(). Importing it would
    create a circular reference, so these 4 lines are duplicated. **Fix both at once.**
    """
    v = os.environ.get("AUTOMIX_NO_SB_IN_SEARCH")
    # **Return default.** Until  this was `return False`: the function took
    # a default argument but never used it in the body. The caller passes
    # default=(reward_form == "pq_only") yet always got False back, so **the design's
    #  instruction to drop SongBench from the search had no effect**
    # (SongBench was scored every step and its value kept appearing in the prompt).
    # agreement_loop_qwen.no_sb_in_search correctly returns default, so the two copies
    # had diverged, against the docstring's "fix both at once".
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def compute_reward_form(pq: float, sb: float, calib, form: str = "min",
                        w: float = 0.5) -> float:
    """Reward dispatcher for the reward-form ablation (pure CPU computation).

    Reuses the existing forms in agreement_reward.py and adds pq_only / sb_only.
    pq_only=Audiobox alone (reproduces reward hacking), sb_only=SongBench alone.
    The SB held-out configuration corresponds to --reward-form pq_only (SB is not
    used for optimization, only for scoring).
    """
    from mix_orchestrator.eval.agreement_reward import (
        compute_agreement, compute_weighted)
    if form == "min":
        return compute_agreement(pq, sb, calib)
    if form == "weighted":
        return compute_weighted(pq, sb, calib, w=w)
    if form == "pq_only":
        return float(calib.z_pq(pq))
    if form == "sb_only":
        return float(calib.z_sb(sb))
    raise ValueError(f"unknown reward form: {form!r}; valid={_REWARD_FORMS}")


# ============================================================================
# Pure logic (unit-testable on CPU)
# ============================================================================
def parse_splits(splits_arg: str) -> List[str]:
    """Normalize a spec like 'train,dev,test' / 'train dev' into a list of splits."""
    raw = re.split(r"[,\s]+", splits_arg.strip())
    out: List[str] = []
    for s in raw:
        if not s:
            continue
        if s not in _VALID_SPLITS:
            raise ValueError(f"unknown split {s!r}; valid={_VALID_SPLITS}")
        if s not in out:
            out.append(s)
    if not out:
        raise ValueError("Specify at least one split.")
    return out


def safe_song_id(song_id: str) -> str:
    """Make song_id filename-safe (whitespace/symbols -> '_'). Used in mix filenames."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(song_id)).strip("_")


# ============================================================================
# Pure logic for the randomized-stem recovery experiment (unit-testable on CPU)
#
# Motivation (note : in MUSDB18 the mixture = sum of stems = the
# professional mix, so there is a weakness that dry ~= professional mix (the dry
# baseline is unfairly strong). We deliberately break the professional balance by
# applying a random gain U(LOW, HIGH) to each stem, and test whether the proposed
# method (proposed_rand) can restore a good mix from that corrupted state
# (random_dry).
#
# The seed is derived deterministically from the song index and the seed index
# (random_seed_for), and the gain multiplier is sampled linearly from the rng per
# stem (sample_random_gains). Hence the same (song_index, seed_idx) always
# reproduces the same corruption.
# ============================================================================
def random_seed_for(song_index: int, seed_idx: int, base_seed: int = 0,
                    stride: int = 100) -> int:
    """Build a deterministic, reproducible seed from (song_index, seed_idx).

    Each song_index gets a window of width stride, and seed_idx offsets within it.
    base_seed can be added to shift the whole run. If stride > the expected number of
    seeds, the seed spaces of different songs never collide.
    """
    return int(base_seed) + int(song_index) * int(stride) + int(seed_idx)


def sample_random_gains(tracks: Sequence[str], rng: "np.random.Generator",
                        low: float, high: float,
                        gain_db_range: Optional[Tuple[float, float]] = None,
                        ) -> Dict[str, float]:
    """Assign a linear gain multiplier to each track (one draw per track from the rng
    in sorted order; fixed for reproducibility).

    Default (gain_db_range=None): a linear multiplier from U(low, high)
    (e.g. 0.1 corresponds to -20 dB).
    With gain_db_range=(lo_db, hi_db): sample gain_db ~ U(lo_db, hi_db) on the dB axis
    and return the linear multiplier 10**(gain_db/20) (-18 dBFS is the reference of
    minimum attenuation; e.g. (-43, -18)).
    All conditions later pass through normalize_for_eval(-14 LUFS), so the absolute
    attenuation cancels out and only the destruction of the relative balance between
    stems takes effect (maximum spread on the dB axis is |hi_db-lo_db|).
    The return value is {track: gain_multiplier}.
    """
    if gain_db_range is not None:
        lo_db, hi_db = float(gain_db_range[0]), float(gain_db_range[1])
        if not (hi_db >= lo_db):
            raise ValueError(
                f"random gain dB range requires low<=high: low={lo_db} high={hi_db}")
        return {t: float(10.0 ** (rng.uniform(lo_db, hi_db) / 20.0))
                for t in sorted(tracks)}
    if not (high >= low):
        raise ValueError(f"random gain range requires low<=high: low={low} high={high}")
    return {t: float(rng.uniform(low, high)) for t in sorted(tracks)}


def apply_stem_gains(main_stems: Dict[str, "np.ndarray"],
                     gains: Dict[str, float]) -> Dict[str, "np.ndarray"]:
    """Return a new dict with a linear gain multiplier applied to each stem
    (pure numpy, CPU).

    The original arrays are not modified; the gain is applied to a copy (the caller's
    stems are not destroyed). A track missing from gains gets multiplier 1.0
    (pass-through). The stem shape ((C, N) / (N,)) is preserved.
    """
    out: Dict[str, np.ndarray] = {}
    for name, arr in main_stems.items():
        g = float(gains.get(name, 1.0))
        out[name] = (np.asarray(arr, dtype=np.float32) * g).astype(np.float32)
    return out


def mix_filename(method: str, song_id: str) -> str:
    """Return the kad_eval-compatible `<method>__<safe_song_id>.wav`.

    The method itself must not contain '__' (kad_eval takes the method via
    split('__')[0]).
    """
    if "__" in method:
        raise ValueError(f"method must not contain '__': {method!r}")
    return f"{method}__{safe_song_id(song_id)}.wav"


# ============================================================================
# Search on a 12-second excerpt (--search-excerpt-sec). Default 0 = disabled =
# full song as before.
#
# Motivation :
#   renderer.render() re-applies the whole chain from the raw stems every time, so the
#   cost of one step grows in proportion to the number of accepted actions and the
#   total cost becomes O(K^2). Measured at 0.45 s/effect (220-second song). At 10000
#   steps that is 40 days per song, i.e. infeasible.
#
# Rationale:
#   The listening-test stimuli are 12-second excerpts (EXCERPT_SEC=12.0;
#   experiments/subjective/pilot_v2/build_stimuli_v2.py:71). Running the search on the
#   same window makes the objective and the evaluated material coincide. The window is
#   determined by select_excerpts.analyse_song **from the dry stems**, so it is
#   identical across all conditions regardless of condition / corruption seed, and the
#   stimuli and the search use exactly the same sample range.
#
# Validity:
#   The 9 action types used in the search (gain/pan/width/eq/comp/reverb/delay/
#   saturation/deesser) are all static/insert effects and **have no absolute-time
#   parameters** (no start_sec/breakpoints like section_* / *_automation). Therefore
#   the operation of transplanting a chain decided on the excerpt onto the full-song
#   stems and rendering once is well defined, and the meaning of the chain does not
#   depend on the stem length.
# ============================================================================

#: JSON holding the listening-stimulus windows (generated by select_excerpts.py).
SUBJECTIVE_EXCERPTS_REL = "outputs/analysis/subjective_excerpts.json"

#: In-process cache so the JSON is not re-read for every song. key = resolved
#: absolute path.
_EXCERPT_INDEX_CACHE: Dict[str, Tuple[Dict[str, Dict[str, Any]], Optional[float]]] = {}


def canon_song_id(s: str) -> str:
    """Absorb song_id spelling variations (same definition as build_stimuli_v2.canon).

    Runs differ in how they treat whitespace/apostrophes/hyphens, so we reduce to
    alphanumerics only before comparing. Unless this is the same function as in the
    stimulus builder, the window lookup goes out of sync.
    """
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def load_subjective_excerpt_index(json_path: Optional[str] = None
                                  ) -> Tuple[Dict[str, Dict[str, Any]], Optional[float]]:
    """Read subjective_excerpts.json and return an index of canon(song_id) -> row.

    Returns:
        (index, config_length_sec). If the default path does not exist, return
        ({}, None) and every song falls back to ``analyse_song`` (the same rule as
        the stimuli, so the windows still match).

    Raises:
        SystemExit: when the path **explicitly given** by ``--search-excerpt-json``
            is broken as JSON. Passing a broken file is always an accident, so we
            fail fast. If the file merely does not exist, warn and continue without
            an index (the window is then decided by ``analyse_song`` = the same rule
            as the stimuli).

    JSON that is missing ``config.length`` is returned with an unknown length
    (``None``) **without discarding the index**. Previously the exception was
    swallowed here and the whole index was emptied, so a user who passed their own
    JSON silently fell back for every song (which can make the window differ from the
    stimuli). Note that when the length is unknown, ``pick_excerpt_start_sec`` skips
    the length check, so a row is adopted even if the requested length and the JSON
    length differ (hence the warning).
    """
    p = Path(json_path) if json_path else (ROOT / SUBJECTIVE_EXCERPTS_REL)
    key = str(p)
    if key in _EXCERPT_INDEX_CACHE:
        return _EXCERPT_INDEX_CACHE[key]
    explicit = json_path is not None
    index: Dict[str, Dict[str, Any]] = {}
    length: Optional[float] = None
    try:
        data = json.loads(p.read_text("utf-8"))
    except json.JSONDecodeError as ex:
        if explicit:
            raise SystemExit(
                f"[excerpt] --search-excerpt-json {p} is broken as JSON: "
                f"{ex!r}\n  -> Fix the path, or drop the flag and let the default "
                f"{SUBJECTIVE_EXCERPTS_REL} / analyse_song handle it.")
        _EXCERPT_INDEX_CACHE[key] = ({}, None)
        return {}, None
    except OSError as ex:
        if explicit:
            print(f"[excerpt] WARN cannot open --search-excerpt-json {p} "
                  f"({ex!r}). Continuing without an index; windows are decided by "
                  f"analyse_song (the same rule as the stimulus builder).", flush=True)
        _EXCERPT_INDEX_CACHE[key] = ({}, None)
        return {}, None
    raw_len = (data.get("config") or {}).get("length")
    try:
        length = float(raw_len)
    except (TypeError, ValueError):
        length = None
        print(f"[excerpt] WARN {p}: config.length is missing "
              f"({raw_len!r}). Using the index without checking the window length.",
              flush=True)
    for row in (data.get("songs") or []):
        index[canon_song_id(row.get("song_id", ""))] = row
    if explicit and not index:
        print(f"[excerpt] WARN {p}: songs is empty. Falling back to analyse_song "
              f"for every song.", flush=True)
    _EXCERPT_INDEX_CACHE[key] = (index, length)
    return index, length


def _analyse_song_window(song_id: str, split: str, length_sec: float
                         ) -> Optional[Dict[str, Any]]:
    """Call select_excerpts.analyse_song **with its default arguments** (the same rule
    as the stimuli).

    Fallback for songs not listed in the JSON (only 77 of the 150 songs are in it).
    build_stimuli_v2.py does the same for missing songs (line 508 of that file).
    Returns None on failure, e.g. when librosa/soundfile is unavailable.
    """
    subj = str(ROOT / "experiments" / "subjective")
    import sys as _sys
    # **append, not insert(0).** experiments/subjective/ contains modules with
    # generic names such as build_stimuli.py / master_power.py / pilot_analysis.py,
    # and inserting at the front would permanently shadow same-named project modules
    # / 3rd-party packages. Appending leaves the existing resolution order untouched.
    if subj not in _sys.path:
        _sys.path.append(subj)
    try:
        from select_excerpts import analyse_song  # type: ignore
        return analyse_song(song_id, split, float(length_sec))
    except Exception as ex:                                     # noqa: BLE001
        print(f"[excerpt] WARN analyse_song failed for {song_id!r}: {ex!r}", flush=True)
        return None


def pick_excerpt_start_sec(song_id: str, split: str, length_sec: float,
                           duration_sec: float,
                           excerpt_index: Optional[Dict[str, Dict[str, Any]]] = None,
                           index_length_sec: Optional[float] = None,
                           analyse_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
                           ) -> Tuple[float, str]:
    """Return the excerpt window start time [s] and a label for where it came from
    (pure logic, injectable).

    Priority:
      1. The matching row in ``subjective_excerpts.json`` (= exactly the same window
         as the listening stimuli). Adopted only when the excerpt length matches and
         the row is not ``excluded``.
      2. Re-call ``select_excerpts.analyse_song`` with default arguments (the same
         rule). Even for ``excluded`` songs (dropped by listening-test eligibility)
         we use ``start_sec`` if present: exclusion judges "is it worth presenting to
         subjects", not the validity of the window.
      3. A deterministic centered window (for songs where neither works). 0 if the
         excerpt is longer than the song.

    Every path returns a deterministic value decided only by song_id and length_sec.
    """
    idx = excerpt_index if excerpt_index is not None else {}
    row = idx.get(canon_song_id(song_id))
    if (row is not None and not row.get("excluded")
            and row.get("start_sec") is not None
            and (index_length_sec is None
                 or abs(float(index_length_sec) - float(length_sec)) < 1e-6)):
        return float(row["start_sec"]), "subjective_excerpts.json"

    fn = analyse_fn if analyse_fn is not None else _analyse_song_window
    res = fn(song_id, split, length_sec)
    if res and res.get("start_sec") is not None:
        tag = "analyse_song"
        if res.get("excluded"):
            tag = f"analyse_song(excluded:{res.get('exclude_reason', '')})"
        return float(res["start_sec"]), tag

    return max(0.0, (float(duration_sec) - float(length_sec)) / 2.0), "centered_fallback"


def excerpt_sample_window(start_sec: float, length_sec: float, sr: int,
                          total_samples: int) -> Tuple[int, int]:
    """Return the sample window (start_n, n) from the start second / excerpt length.
    Pure arithmetic.

    The rounding is identical to the stimulus builder (``int(round(sec * sr))``;
    build_stimuli_v2.read_excerpt / select_excerpts.render_stimuli). Without the same
    rounding we would search a range offset by one sample.

    If the window runs past the end of the song it is shifted earlier. If the song
    itself is shorter than the excerpt, the whole song is returned.
    """
    total = int(total_samples)
    n = int(round(float(length_sec) * int(sr)))
    if n <= 0 or n >= total:
        return 0, total
    start_n = int(round(float(start_sec) * int(sr)))
    start_n = min(max(0, start_n), total - n)
    return start_n, n


def slice_stems(stems: Dict[str, "np.ndarray"], start_n: int, n: int
                ) -> Dict[str, "np.ndarray"]:
    """Slice every stem with the same sample window [start_n, start_n+n) (pure numpy).

    Only the last axis (= the sample axis) is cut. If stems have different lengths
    (which does not happen in MUSDB) the result is truncated to the shorter one.
    Returns C-contiguous copies; the original arrays are not modified.
    """
    out: Dict[str, np.ndarray] = {}
    for name, arr in stems.items():
        x = np.asarray(arr)
        seg = x[..., start_n:start_n + n]
        out[name] = np.ascontiguousarray(seg.astype(np.float32, copy=False))
    return out


def time_absolute_reason(state) -> Optional[str]:
    """Return a reason string if the state contains an effect with absolute time,
    otherwise None.

    This is exactly the condition under which ``swap_state_stems`` decides that
    "transplanting is impossible". Separating the check from raising lets the caller
    branch **before** it crashes (a run that allows automation in excerpt mode has to
    switch from transplanting to reporting the excerpt only; see the discussion of
    :data:`EXCERPT_ONLY_ENV`).
    """
    if getattr(state, "section_processors", None):
        return ("section_processors carry absolute time, so the stems cannot be "
                "transplanted")
    if getattr(state, "section_boundaries", None):
        return "a state with section_boundaries cannot have its stems transplanted"
    _chains = (list(state.static_processors.items())
               + list(state.transient_processors.items())
               + [("<master>", tuple(state.master_chain))])
    for trk, chain in _chains:
        for eff in chain:
            if hasattr(eff, "breakpoints") or hasattr(eff, "start_sec"):
                return (f"effect {type(eff).__name__} carries absolute time "
                        f"(track={trk}), so the stems cannot be transplanted")
    return None


def swap_state_stems(state, stems: Dict[str, "np.ndarray"]):
    """Return a new state with only the stems replaced, keeping the MixState chain.

    The operation used to "transplant" a chain searched on the excerpt onto the full
    song. All 9 action types are time-invariant effects stacked into
    ``static_processors``, so the meaning of the chain does not depend on the stem
    length (there is no absolute-time parameter such as start_sec of section_* or the
    breakpoints of *_automation). If the state does contain an effect with absolute
    time, the transplant is invalid, so fail immediately.

    The stems go through the same normalization as ``MixState.initial_from_stems``
    (2-D coercion + float32), so the result is bit-identical to
    ``initial_from_stems(full)`` with the same chain.
    """
    from dataclasses import replace as _dc_replace
    from mix_orchestrator.dsp.mix_state import MixState, _ensure_2d

    if not isinstance(state, MixState):
        raise TypeError(f"a MixState is required: {type(state).__name__}")
    # The check is centralized in time_absolute_reason (so the caller can branch on
    # the same condition beforehand). It scans section / static / transient / master
    # alike. The current MasterEffects (MasterEQ / MasterCompressor / MasterLimiter)
    # carry no absolute time, so master_chain always passes for now, but we do not
    # leave a hole in the guard.
    _reason = time_absolute_reason(state)
    if _reason is not None:
        raise ValueError(_reason)
    # Equivalent to initial_from_stems (orientation and dtype). copy=False is used so
    # that we do not add a 300 MB-scale copy of the full-song stems for every seed.
    # The values are identical, so the render output is bit-identical to going
    # through initial_from_stems.
    normed = {k: _ensure_2d(np.asarray(v)).astype(np.float32, copy=False)
              for k, v in stems.items()}
    if set(normed) != set(state.stems):
        raise KeyError(f"different stem sets: state={sorted(state.stems)} "
                       f"new={sorted(normed)}")
    # Rebuild state_id. MixState._content_hash does not hash the **contents** of the
    # stems, so a plain replace would give the excerpt state and the full-song state
    # the same ID. agent/memory.py and baselines/*.py look states up by state_id, so
    # if the state is reused there it silently hits the wrong entry.
    # Include parent_state_id in the hash so the ID differs, and keep the lineage.
    return _dc_replace(state, stems=normed,
                       parent_state_id=state.state_id, state_id="",
                       action_summary="swap_stems")


# ---------------------------------------------------------------------------
# The choice not to transplant the chain onto the full song in excerpt mode
# (added 
# ---------------------------------------------------------------------------
#: **Why this is needed**: excerpt mode is designed as "search on 12 s -> transplant
#: the chain onto the full-song stems -> score the full song exactly once", and its
#: soundness rests on ``swap_state_stems`` rejecting effects with absolute time. The
#: breakpoints of gain automation are precisely such absolute times, so putting
#: automation into the action space guarantees a crash at the end of the search.
#:
#: **There were 2 options**:
#:   (a) do not transplant onto the full song in excerpt mode, and report only the
#:       12-second result
#:   (b) shift the breakpoints by the excerpt start time and transplant
#:
#: **We take (a). Reasons**:
#:   1. With (b) the behaviour outside the window is undefined. The renderer's
#:      ``_apply_gain_automation`` uses ``np.interp``, so outside the first/last
#:      breakpoint the value is **constant** at the endpoint. Placing 12 seconds of
#:      breakpoints at 111-123 s of a 220-second song means the gain of the first
#:      point applies over 0-111 s and the gain of the last point **keeps applying to
#:      the entire rest of the song** over 123-220 s. The LLM has only listened to 12
#:      seconds and what it does to the remaining 97 seconds has never been
#:      evaluated. That amounts to "modifying a range that was never searched without
#:      verification", and it becomes impossible to tell whether the full-song PQ/SB
#:      value is a result of the search or of luck. One could pin the endpoints to
#:      0 dB so that outside the window is pass-through, but then the implementation
#:      would be silently adding breakpoints the LLM never proposed, which breaks the
#:      meaning of the eval gate ("evaluate the proposed chain").
#:   2. The practical harm of (a) is small. **The listening-test stimuli are exactly
#:      the 12-second excerpts** (experiments/subjective/select_excerpts.py), so the
#:      audio presented for subjective evaluation coincides exactly with the search
#:      domain of excerpt mode. Full-song numbers are already available from the
#:      existing "without automation" runs.
#:
#: The default is **to transplant as before** (= raise for a state containing
#: automation). The transplant is skipped only when ``AUTOMIX_EXCERPT_ONLY=1`` or
#: ``--excerpt-only`` is given. The JSON of such a run contains
#: ``report_domain='excerpt'`` and ``no_full_transplant_reason``, so it cannot be
#: mixed into the full-song table.
EXCERPT_ONLY_ENV = "AUTOMIX_EXCERPT_ONLY"
_TRUTHY_STR = ("1", "true", "yes", "on")


def excerpt_only_default() -> bool:
    """Read the env var ``AUTOMIX_EXCERPT_ONLY``. False if unset (old behaviour)."""
    return (os.environ.get(EXCERPT_ONLY_ENV) or "").strip().lower() in _TRUTHY_STR


def excerpt_beat_times(stems: Dict[str, "np.ndarray"], sr: int,
                       start_sample: int, n_samples: int,
                       hop_length: int = 512,
                       analysis_sr: int = 22050
                       ) -> Tuple[List[float], Optional[float]]:
    """Return the beat times inside the excerpt window as **seconds relative to the
    start of the excerpt (= 0)**.

    Returns:
        ``(beat_times_rel, tempo_bpm)``. ``([], None)`` if detection fails.

    Design points:

    * Beats are obtained by running ``librosa.beat.beat_track`` over the **whole
      song** and then cutting out the window. Analysing only 12 seconds makes the
      tempo estimate unstable and also yields a different grid from the one
      ``select_excerpts.analyse_song`` used to snap the window start, which breaks
      the assumption "start of excerpt = start of a beat".
    * **Match the analysis conditions to analyse_song.** That code averages channels
      with ``select_excerpts.read_mono``, decimates by an integer factor down to
      ~22050 Hz, and then calls ``beat_track(hop_length=512, units='time')``.
      Analysing at 44100 Hz with hop=512 doubles the time resolution and shifts the
      grid phase by up to 1 hop (~23 ms) (measured: the beat at the start of the
      excerpt began at 0.406 s instead of 0.000). Applying the same decimation
      reproduces exactly the grid used when the window was snapped.
    * The input is assumed to be the **dry stems before corruption**. The beat
      positions must not change per corruption seed (beats are a property of the
      source material, not a function of the condition). Since the MUSDB18 mixture is
      the sum of the 4 stems, the stem sum is numerically equivalent to mixture.wav.
    * The returned times are in the coordinate system passed to breakpoints (start of
      excerpt = 0). Beats outside the window are dropped.
    """
    try:
        import librosa
    except Exception:                                           # noqa: BLE001
        return [], None
    try:
        acc: Optional[np.ndarray] = None
        for v in stems.values():
            a = np.asarray(v, dtype=np.float32)
            m = a.mean(axis=0) if a.ndim == 2 else a
            acc = m.astype(np.float32) if acc is None else acc + m[:acc.shape[0]]
        if acc is None or acc.size == 0:
            return [], None
        # Same integer decimation as select_excerpts.read_mono.
        step = max(1, int(round(float(sr) / float(analysis_sr))))
        y = np.ascontiguousarray(acc[::step]) if step > 1 else np.ascontiguousarray(acc)
        y_sr = int(sr) // step if step > 1 else int(sr)
        tempo, beats = librosa.beat.beat_track(
            y=y, sr=y_sr, hop_length=int(hop_length), units="time")
        if beats is None or len(beats) == 0:
            return [], None
        t0 = float(start_sample) / float(sr)
        t1 = float(start_sample + n_samples) / float(sr)
        # **Tolerance so the beat at the start of the window is not dropped.** The
        # window is decided by analyse_song snapping to the beat grid, so a beat is
        # almost always sitting at the start of the excerpt. However the detection
        # time resolution is only 1 hop (512/22050 ~= 23.2 ms), so that beat can be
        # quantized slightly before the window start. Cutting naively with
        # ``t0 <= b`` drops only the first beat and **makes the whole grid look
        # shifted by one beat** (measured: the first beat started at 0.418 s = exactly
        # one beat later, instead of 0.000). A beat pulled earlier by at most 1 hop is
        # treated as "the beat at the start of the window" and rounded to 0.0.
        tol = float(hop_length) / float(y_sr)
        rel: List[float] = []
        for b in beats:
            bt = float(b)
            if bt < t0 - tol or bt >= t1:
                continue
            rel.append(round(max(0.0, bt - t0), 6))
        bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else None
        return rel, bpm
    except Exception as ex:                                     # noqa: BLE001
        print(f"[excerpt] WARN beat detection failed: {ex!r}", flush=True)
        return [], None


def resolve_search_excerpt(song_id: str, split: str, length_sec: float, sr: int,
                           total_samples: int,
                           json_path: Optional[str] = None,
                           preroll_sec: float = 0.0) -> Dict[str, Any]:
    """Decide the search excerpt window for one song and collect it into a metadata dict.

    If ``preroll_sec > 0``, a run-up region is added **before the scoring window**
    when slicing (and discarded after rendering). It exists to push the warm-up
    transient — caused by the internal state of reverb/delay/IIR starting from 0 at
    the head of the buffer — out of the scoring window. The scoring window itself
    (start_sample..start_sample+n_samples) is unchanged by preroll.

    Returns:
        {'length_sec','start_sec','end_sec','start_sample','n_samples','sr',
         'requested_start_sec','source',
         'preroll_sec','preroll_samples','slice_start_sample','slice_n_samples'}.
    """
    index, index_len = load_subjective_excerpt_index(json_path)
    duration_sec = float(total_samples) / float(sr)
    start_sec, source = pick_excerpt_start_sec(
        song_id, split, length_sec, duration_sec,
        excerpt_index=index, index_length_sec=index_len)
    start_n, n = excerpt_sample_window(start_sec, length_sec, sr, total_samples)
    # If there is not enough audio before the song start, give up on the shortfall and
    # use only what is actually available (the window is not shifted earlier).
    pre_n = min(int(round(max(0.0, float(preroll_sec)) * int(sr))), start_n)
    return {
        "length_sec": float(length_sec),
        "start_sec": round(start_n / float(sr), 6),
        "end_sec": round((start_n + n) / float(sr), 6),
        "start_sample": int(start_n),
        "n_samples": int(n),
        "sr": int(sr),
        "requested_start_sec": round(float(start_sec), 6),
        "source": source,
        "preroll_sec": round(pre_n / float(sr), 6),
        "preroll_samples": int(pre_n),
        "slice_start_sample": int(start_n - pre_n),
        "slice_n_samples": int(pre_n + n),
    }


def build_song_plan(splits: List[str], dataset: str = "musdb18",
                    enumerate_fn: Optional[Callable[..., List[Optional[str]]]] = None,
                    offset: int = 0, limit: Optional[int] = None,
                    per_split_limit: int = 1000) -> List[Dict[str, str]]:
    """Build the song list by concatenating the splits in order, then cut it with the
    offset/limit window.

    Args:
        splits: an ordered sub-list of ['train','dev','test'].
        enumerate_fn: function mapping split -> list of song_id (injectable in tests).
                      Uses experiments._common.enumerate_songs when None.
        offset/limit: the chunk window over the concatenated list.
    Returns:
        A list of [{'dataset', 'split', 'song_id'}].
    """
    fn = enumerate_fn or enumerate_songs
    rows: List[Dict[str, str]] = []
    for sp in splits:
        ids = [s for s in fn(dataset, limit=per_split_limit, split=sp) if s is not None]
        for sid in ids:
            rows.append({"dataset": dataset, "split": sp, "song_id": str(sid)})
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def song_result_path(run_dir: Path, song_id: str) -> Path:
    return run_dir / "songs" / f"{safe_song_id(song_id)}.json"


def is_song_done(run_dir: Path, song_id: str) -> bool:
    """Resume check: if the per-song JSON exists and status is terminal, treat the
    song as complete.

    The terminal states are ``done`` and ``excluded``. ``excluded`` marks "a song
    that is definitively impossible to run in this environment and must not be
    retried on resume" (added . The only such song at present is
    ``Lushlife - Toynbee Suite``: full-song SB scoring needs ~47 GiB, which does not
    fit the cluster's A100-40GB (confirmed by job 18550). Without the mark, every
    resume spends ~20 minutes per arm waiting for an OOM. ``error`` may be a
    transient failure, so it is retried as before.
    """
    p = song_result_path(run_dir, song_id)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text("utf-8")).get("status") in ("done", "excluded")
    except Exception:                                           # noqa: BLE001
        return False


def filter_pending(plan: List[Dict[str, str]], run_dir: Path,
                   resume: bool = True) -> List[Dict[str, str]]:
    """With resume=True, return the list with already-done songs removed."""
    if not resume:
        return list(plan)
    return [r for r in plan if not is_song_done(run_dir, r["song_id"])]


#: Corruption settings that must match the existing songs on resume.
#: Resuming with a mismatch here mixes **songs corrupted under a different protocol**
#: into the same run and breaks the premise of the CRN paired comparison.
_CORRUPTION_FIELDS = ("normalize_corrupted_lufs", "gain_mode", "random_init_gain_range",
                      "random_seeds", "random_base_seed")

#: Fields added on/after . Compared **only when present in the done JSON**.
#: Older run JSONs lack some fields, and treating a missing field as a mismatch would
#: block resuming into an existing run with a false positive (required for backward
#: compatibility).
_CORRUPTION_FIELDS_OPTIONAL = ("random_seeds",)


def _search_domain_of_record(rec: Dict[str, Any]) -> Tuple[float, float]:
    """Recover (search_excerpt_sec, preroll_sec) from a done JSON.

    **Missing = full-song search (0.0, 0.0)**. ``search_excerpt_sec`` is only written
    in excerpt mode, so a missing value can be asserted to mean "full song" rather
    than "unknown" (unlike ``prompt_show_budget``, this default is historically
    unambiguous).
    """
    sec = float(rec.get("search_excerpt_sec") or 0.0)
    pre = float((rec.get("search_excerpt") or {}).get("preroll_sec") or 0.0)
    return sec, pre


def assert_search_domain_consistent(run_dir: Path, args) -> None:
    """On resume, check that the **search domain** (full song / N-second excerpt)
    matches.

    ``--search-excerpt-sec`` changes the objective function itself. If songs searched
    on the full song and songs searched on a 12-second excerpt end up in one run, the
    run becomes a mixture of different experiments that cannot be told apart without
    inspecting the per-song JSON. We fail fast, like
    ``assert_corruption_consistent``, to mechanically stop forgetting or over-adding
    the flag.

    Can be disabled deliberately with ``ALLOW_SEARCH_DOMAIN_MISMATCH=1`` (record the
    reason).
    """
    if os.environ.get("ALLOW_SEARCH_DOMAIN_MISMATCH") == "1":
        print("[guard] ALLOW_SEARCH_DOMAIN_MISMATCH=1, skipping the search-domain "
              "consistency check",
              flush=True)
        return
    songs_dir = run_dir / "songs"
    if not songs_dir.is_dir():
        return
    want = (float(getattr(args, "search_excerpt_sec", 0.0) or 0.0),
            float(getattr(args, "search_excerpt_preroll_sec", 0.0) or 0.0))
    mismatches: List[str] = []
    for p in sorted(songs_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("status") != "done":
            continue
        have = _search_domain_of_record(rec)
        # The effective preroll can be smaller than requested because we "give up on
        # the shortfall at the song start". Apart from requested 0 / effective 0, only
        # sec is checked strictly.
        if abs(have[0] - want[0]) > 1e-9 or (want[0] > 0 and have[0] > 0
                                             and abs(have[1] - want[1]) > 0.5):
            mismatches.append(
                f"  {rec.get('song_id')}: existing excerpt_sec={have[0]!r} "
                f"preroll={have[1]!r} / this run excerpt_sec={want[0]!r} "
                f"preroll={want[1]!r}")
    if mismatches:
        head = mismatches[:10]
        more = (f"\n  ... {len(mismatches)-10} more" if len(mismatches) > 10 else "")
        raise SystemExit(
            f"[guard] the search domain (--search-excerpt-sec) differs from the "
            f"existing run ({len(mismatches)} songs). Resuming would mix **songs "
            f"with different objective functions** into the same run:\n"
            + "\n".join(head) + more
            + "\n  -> Match the arguments to the existing run, or use a different "
              "--run-name."
              "\n  -> Use ALLOW_SEARCH_DOMAIN_MISMATCH=1 only if you intend to mix them.")


def sb_chunk_sec_effective() -> float:
    """The chunk length (seconds) actually used by the SongBench scorer.

    The env var SB_CHUNK_SEC overrides the default 300.0 of
    songbench_mixing._score_7dim. Values <= 0 disable chunking (= a single forward
    over the full song). The effective value is recorded in the per-song JSON so
    scorer drift (chunking was introduced  can be tracked across runs.
    """
    env_cs = os.environ.get("SB_CHUNK_SEC")
    if env_cs is None:
        return 300.0
    try:
        return float(env_cs)
    except ValueError:
        return 300.0


def assert_corruption_consistent(run_dir: Path, args) -> None:
    """Check that this run's arguments match the corruption settings of the existing
    run, and fail immediately if they differ.

    Countermeasure for the  accident: `sweep_qwen32b_n100` (10 songs) /
    `sweep_qwen72b_n100` (3 songs) were re-run to recover from a vLLM server timeout
    without `--normalize-corrupted-lufs`, so they ran with
    `normalize_corrupted_lufs=None` (which affects the effect size in the paper).
    The same class of accident also happened with `sweep_random_ns10-40` on
    . Failing fast mechanically prevents "mixing without noticing".

    Can be disabled deliberately with the env var `ALLOW_CORRUPTION_MISMATCH=1`
    (record the reason).
    """
    if os.environ.get("ALLOW_CORRUPTION_MISMATCH") == "1":
        print("[guard] ALLOW_CORRUPTION_MISMATCH=1, skipping the corruption "
              "consistency check",
              flush=True)
        return
    songs_dir = run_dir / "songs"
    if not songs_dir.is_dir():
        return
    want = {
        "normalize_corrupted_lufs": getattr(args, "normalize_corrupted_lufs", None),
        "gain_mode": ("db" if getattr(args, "random_gain_db_range", None) is not None
                      else "linear"),
        "random_init_gain_range": list(getattr(args, "random_gain_db_range", None)
                                       or getattr(args, "random_init_gain_range", []) or []),
        "random_seeds": int(getattr(args, "random_seeds", 0) or 0),
        "random_base_seed": int(getattr(args, "random_base_seed", 0) or 0),
    }
    mismatches: List[str] = []
    for p in sorted(songs_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("status") != "done":
            continue
        for f in _CORRUPTION_FIELDS:
            # For random_base_seed we can assert "missing = 0" (runs from before the
            # flag was introduced on  always ran with shift 0; the JSON only
            # records a non-zero shift). Skipping on missing would let songs from a
            # different seed band silently mix in when resuming an existing run with
            # --random-base-seed, so we normalize to 0 and compare strictly. A default
            # resume (shift 0) passes through as before, since 0==0.
            if f == "random_base_seed":
                have_rbs = int(rec.get(f) or 0)
                exp_rbs = int(want.get(f) or 0)
                if have_rbs != exp_rbs:
                    mismatches.append(
                        f"  {rec.get('song_id')}: {f} existing={have_rbs!r} "
                        f"this run={exp_rbs!r}")
                continue
            # Backward compatibility: newly added check items may be absent from
            # existing JSONs. Skip when missing (treating a missing field as a
            # mismatch would fail resumes into old runs with a false positive).
            if f in _CORRUPTION_FIELDS_OPTIONAL and f not in rec:
                continue
            have = rec.get(f)
            exp = want.get(f)
            if isinstance(have, list) or isinstance(exp, list):
                have, exp = list(have or []), list(exp or [])
            if have != exp:
                mismatches.append(
                    f"  {rec.get('song_id')}: {f} existing={have!r} this run={exp!r}")
    if mismatches:
        head = mismatches[:10]
        more = (f"\n  ... {len(mismatches)-10} more" if len(mismatches) > 10 else "")
        raise SystemExit(
            f"[guard] the corruption settings differ from the existing run "
            f"({len(mismatches)} songs). Resuming would mix songs from a different "
            f"protocol:\n"
            + "\n".join(head) + more
            + "\n  -> Match the arguments to the existing run, or use a different "
              "--run-name."
              "\n  -> If the existing songs are the wrong ones, move their JSON aside "
              "before resuming."
              "\n  -> Use ALLOW_CORRUPTION_MISMATCH=1 only if you intend to mix them.")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Agreement-reward loop runner over all 150 songs")
    ap.add_argument("--dataset", default="musdb18")
    ap.add_argument("--splits", default="train,dev,test",
                    help="comma/whitespace separated splits (train/dev/test)")
    ap.add_argument("--offset", type=int, default=0,
                    help="start index into the concatenated song list "
                         "(for chunked submission)")
    ap.add_argument("--limit", type=int, default=None,
                    help="number of songs this job processes (chunk size)")
    ap.add_argument("--run-name", default="agree_all_v1")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="redo already-done songs too (default is resume=skip)")
    ap.add_argument("--duration-sec", type=float, default=0.0,
                    help="<=0 means the full song (default)")
    ap.add_argument("--n-random-calib", type=int, default=12,
                    help="number of random candidates added to the calibration pool "
                         "(raised from v2's 5 so the SB guard takes effect; "
                         "internal notes)")
    ap.add_argument("--n-search", type=int, default=28)
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-form", choices=_REWARD_FORMS, default="min",
                    help="search reward form. min=agreement of both metrics "
                         "(default; anti-hack), weighted=weighted average, "
                         "pq_only=Audiobox alone (reproduces reward hacking; SB is "
                         "held out and only scored), sb_only=SongBench alone")
    ap.add_argument("--reward-weight", type=float, default=0.5,
                    help="PQ weight w for --reward-form weighted "
                         "(R = w*z_PQ + (1-w)*z_SB)")
    # --- Where the scorers live (added  ---
    # **Do not make this an environment variable.** The container launch in the cluster job wrapper
    # only passes the variables listed with -e, so with an env-var scheme forgetting
    # to pass it would silently fall back to in-process scoring and produce
    # bit-identical results while holding a GPU. That is an undetectable failure.
    # We wasted 22 songs this way with ANTI_REPEAT before, so make it an argument.
    ap.add_argument("--scorer", choices=list(SCORER_MODES), default="local",
                    help="where the scorers live. local=inside the search process "
                         "(default, as before), remote=over TCP to a shared scoring "
                         "server (tools/scoring_server.py). With remote the search "
                         "process does not import torch, so it uses no GPU memory and "
                         "dozens of them fit in one job.")
    ap.add_argument("--scorer-addr", default=None,
                    help="'host:port' to connect to for --scorer remote. Multiple "
                         "entries can be given comma-separated. "
                         "Falls back to AUTOMIX_SCORER_ADDR when omitted.")
    # --- warm start (added  ---
    # Apply the first N actions of an existing run's best_chain before starting the
    # search. A way to probe the large-budget regime (k>=1000) at the cost of one
    # ordinary wave.
    ap.add_argument("--warm-start-run", default=None,
                    help="name of the run to warm start from (e.g. exc10k_random_pq). "
                         "Applies the first --warm-start-steps actions of its best_chain")
    ap.add_argument("--warm-start-steps", type=int, default=0,
                    help="number of actions to apply. 0 disables it (default, as before)")
    ap.add_argument("--scorer-client-index", type=int, default=0,
                    help="dispatch index when several servers are used "
                         "(primary = index %% number of servers). "
                         "**Use a different value for each process within one job.** "
                         "If all are 0, everything piles onto the first server and the "
                         "second one idles.")
    ap.add_argument("--log-proposals", dest="log_proposals", action="store_true",
                    help="save every proposal of the search (including rejected ones) "
                         "to proposed_rand.search_log in the per-song JSON (for "
                         "diversity analysis and budget curves). "
                         "OFF by default (the JSON output stays compatible).")
    ap.add_argument("--search-excerpt-sec", type=float, default=0.0,
                    help="run the search on an N-second excerpt (default 0 = disabled "
                         "= full song, as before). A positive value enables excerpt "
                         "mode: slice out the stems of the same window as the "
                         "listening stimuli (12 s), run calibration and the greedy "
                         "search on them, then transplant the resulting chain onto "
                         "the full-song stems and render/score exactly once at the "
                         "end. The reported PQ/SB are the full-song values; the "
                         "excerpt values are also written to the *_excerpt fields. "
                         "Use 12 to match the stimuli.")
    ap.add_argument("--search-excerpt-json", type=str, default=None,
                    help=f"path to the excerpt-window JSON "
                         f"(default {SUBJECTIVE_EXCERPTS_REL}). For songs not listed "
                         "there, select_excerpts.analyse_song is re-called with "
                         "default arguments to pick the window under the same rule.")
    ap.add_argument("--search-excerpt-preroll-sec", type=float, default=0.0,
                    help="in excerpt mode, the run-up region [s] added before the "
                         "scoring window (default 0). It is discarded after "
                         "rendering, so the scoring window is unchanged. "
                         "The internal state of reverb/delay starts from 0 at the "
                         "head of the buffer, so without a run-up a warm-up transient "
                         "rides on the first 1-2 seconds of the excerpt and it "
                         "deviates from the stimulus (audio cut from the full song) "
                         "by up to -12.5 dBFS "
                         "(measured: [D] in experiments/verify_search_excerpt.py). "
                         "With 2 it drops to -91.6 dBFS and nearly matches the "
                         "stimulus. The cost is only 12 s -> 14 s (+17%).")
    ap.add_argument("--excerpt-beats", dest="excerpt_beats", action="store_true",
                    default=False,
                    help="in excerpt mode, compute the **beat times** inside the "
                         "excerpt window with librosa and pass them to the LLM prompt "
                         "as seconds relative to the excerpt start "
                         "(default OFF = neither computed nor shown = as before). "
                         "This is what lets the model write gain automation aligned to "
                         "the beat. Beats are a physical property of the source "
                         "material (the same librosa.beat.beat_track as beat_snap in "
                         "select_excerpts), not a measurement from the search, so this "
                         "does not amount to handing over statistics of the random arm "
                         "(answer leakage). beat_track is run once per song over the "
                         "whole song in mono, which adds a few seconds of cost.")
    ap.add_argument("--excerpt-only", dest="excerpt_only", action="store_true",
                    default=None,
                    help="in excerpt mode, **do not transplant** the searched chain "
                         "onto the full song (default OFF = transplant and re-score on "
                         "the full song, as before). Effects with absolute time such "
                         "as gain automation do not satisfy the premise of the "
                         "transplant (time invariance), so this is mandatory for runs "
                         "with automation in the action space. The reported values "
                         "become the PQ/SB of the 12-second excerpt and the JSON gets "
                         "report_domain='excerpt'. "
                         "**They must not be placed in the same column as the "
                         "full-song table.** "
                         "Can also be set with the env var AUTOMIX_EXCERPT_ONLY=1.")
    ap.add_argument("--render-cache", dest="render_cache", action="store_true",
                    default=False,
                    help="enable incremental rendering (default OFF = re-apply the "
                         "whole chain from the raw stems every time, as before). When "
                         "ON, the render cost of one step no longer grows with the "
                         "number of accepted actions "
                         "(O(K^2) -> O(K)). The output is **bit-identical** "
                         "(verified by tests/test_render_cache.py). One RenderCache is "
                         "created per song and passed only to the search norm_fn. "
                         "Unlike the env var MIXORCH_RENDER_CACHE=1, it does not share "
                         "the cache with solo-stem renders, so it does not thrash.")
    ap.add_argument("--render-cache-mb", type=float, default=8192.0,
                    help="LRU budget [MiB] for --render-cache (default 8192). "
                         "For a 220-second, 4-stem song one buffer is about 78 MiB. "
                         "Exceeding the budget only causes LRU eviction and does not "
                         "change the results (a deep search runs under constant "
                         "eviction).")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="skip computing the baselines "
                         "(dry/equal_lufs/P2/P6/MEGAMI) (for debugging)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print the song plan without scoring "
                         "(no GPU needed; for checking the plan)")
    # ---- randomized-stem recovery experiment (active only when specified;
    #      default None = old behaviour) ----
    ap.add_argument("--random-init-gain-range", type=float, nargs=2,
                    metavar=("LOW", "HIGH"), default=None,
                    help="enables random mode: apply a linear gain U(LOW,HIGH) to each "
                         "stem to break the professional balance, and test whether the "
                         "proposed method (proposed_rand) can restore a good mix from "
                         "that corrupted state (random_dry). e.g. 0.1 1.0. "
                         "Without it, the old (non-random) behaviour applies.")
    ap.add_argument("--random-gain-db-range", type=float, nargs=2,
                    metavar=("LOW_DB", "HIGH_DB"), default=None,
                    help="the dBFS version of random mode. Samples "
                         "gain_db ~ U(LOW_DB,HIGH_DB) per stem and applies the linear "
                         "multiplier 10**(gain_db/20) (-18 dBFS is the reference of "
                         "minimum attenuation). Mutually exclusive with "
                         "--random-init-gain-range (linear). e.g. -43 -18.")
    ap.add_argument("--normalize-corrupted-lufs", type=float, default=None,
                    metavar="TARGET_LUFS",
                    help="in random mode, apply a common scalar gain to all stems after "
                         "corruption (per-stem gain) so that their plain sum matches "
                         "TARGET_LUFS (the relative balance, i.e. the corruption, is "
                         "unchanged; only the absolute level is normalized). Prevents "
                         "artifacts from the absolute-dBFS silence gates in MEGAMI and "
                         "others. e.g. -14. Without it there is no common normalization "
                         "(old behaviour).")
    ap.add_argument("--random-seeds", type=int, default=2,
                    help="number of seeds tried per song in random mode (default 2). "
                         "The seed is fixed deterministically as "
                         "song index*100 + seed_idx and is reproducible.")
    # For seed replications (added , in response to an external audit).
    # With the default 0 it is bit-identical to before.
    ap.add_argument("--random-base-seed", type=int, default=0,
                    help="shift **added** to --seed for the base_seed of the random-mode "
                         "seed derivation (random_seed_for) (default 0 = as before). "
                         "The corruption and the search proposal sequence are tied to "
                         "the same seed value, so **both change together** (a "
                         "replication with a different base seed = a different "
                         "corruption + a different search sequence). Mixing into the "
                         "same run-name as an existing run is detected by "
                         "assert_corruption_consistent. Use a new run-name.")
    return ap


def is_random_mode(args) -> bool:
    """Random mode is on if either --random-init-gain-range or
    --random-gain-db-range is given."""
    return (getattr(args, "random_init_gain_range", None) is not None
            or getattr(args, "random_gain_db_range", None) is not None)


def _resolve_excerpt_for_song(main_stems: Dict[str, "np.ndarray"], sr: int,
                              song_id: str, split: str, args
                              ) -> Optional[Dict[str, Any]]:
    """Return the excerpt window metadata if --search-excerpt-sec is positive;
    None if it is 0 or unset.

    The window follows the select_excerpts rule of analysing the **dry MUSDB stems
    before corruption**, so it is fixed to one window per song regardless of the
    corruption seed or the condition (= exactly the same range as the listening
    stimuli). Hence it only needs to be resolved once per song.
    """
    sec = float(getattr(args, "search_excerpt_sec", 0.0) or 0.0)
    if sec <= 0:
        return None
    total = max(int(np.asarray(v).shape[-1]) for v in main_stems.values())
    info = resolve_search_excerpt(
        song_id, split, sec, sr, total,
        json_path=getattr(args, "search_excerpt_json", None),
        preroll_sec=float(getattr(args, "search_excerpt_preroll_sec", 0.0) or 0.0))
    print(f"[excerpt] {song_id}: {info['start_sec']:.3f}-{info['end_sec']:.3f} s "
          f"({info['n_samples']} samples @ {sr} Hz, source={info['source']}, "
          f"preroll={info['preroll_sec']:.3f} s)", flush=True)
    if info["source"] == "centered_fallback":
        # This is the only path where it is certain that "the search used a different
        # range from the listening stimuli", so it must not pass silently. We only get
        # here when neither the json nor analyse_song was usable.
        print(f"[excerpt] WARN {song_id}: fell back to a deterministic centered "
              f"window. It may be a **different range from the listening stimuli** "
              f"(not in subjective_excerpts.json, and "
              f"select_excerpts.analyse_song also failed). If it has to match the "
              f"stimuli, fix the cause before running.",
              flush=True)
    # ---- Beat grid (added ; only when --excerpt-beats is given) ----
    # So that automation breakpoints can be placed on beats, compute the beat times
    # inside the excerpt as **seconds relative to the excerpt start (= 0)** and attach
    # them to args.
    # The proposer (size_sweep_run) is created per song but is created before
    # _run_one_song_random, so it is handed over by referencing args lazily from the
    # closure.
    # By default (without the flag) the computation is not performed at all, so the
    # previous runtime is unchanged.
    args.excerpt_beat_times_rel = None
    args.excerpt_tempo_bpm = None
    if bool(getattr(args, "excerpt_beats", False)):
        # Computed from the dry stems. Fixed to one grid per song regardless of the
        # corruption seed (the same principle as how the window is chosen).
        # Downmixing the whole song to mono + beat_track takes a few seconds.
        beats, bpm = excerpt_beat_times(
            main_stems, sr, info["start_sample"], info["n_samples"])
        info["beat_times_rel"] = beats
        info["tempo_bpm"] = bpm
        args.excerpt_beat_times_rel = beats
        args.excerpt_tempo_bpm = bpm
        if beats:
            print(f"[excerpt] {song_id}: {len(beats)} beats inside the excerpt "
                  f"(tempo~{bpm:.1f} BPM, first {beats[0]:.3f} s, "
                  f"last {beats[-1]:.3f} s)", flush=True)
        else:
            print(f"[excerpt] WARN {song_id}: could not detect beats inside the "
                  f"excerpt. The beat grid will not be shown in the prompt.",
                  flush=True)
    return info


def make_render_cache(args):
    """Create a ``RenderCache`` only when ``--render-cache`` is given; None otherwise.

    When None is returned, the caller calls ``render(state)`` **as is**, so the run is
    bit-identical to before (the behaviour of the env var MIXORCH_RENDER_CACHE is also
    unchanged).
    """
    if not getattr(args, "render_cache", False):
        return None
    from mix_orchestrator.dsp.renderer import RenderCache
    mb = float(getattr(args, "render_cache_mb", 8192.0) or 8192.0)
    return RenderCache(max_bytes=int(mb * 1024 * 1024))


def make_norm_fn(sr: int, target_lufs: float, cache=None):
    """Return a function that takes a MixState/ndarray and normalizes it for eval.

    With ``cache=None`` (default) it is the old path that calls
    ``render(state_or_audio)`` plainly. Passing a ``RenderCache`` switches to
    incremental rendering, but the output is bit-identical.

    **A cache is dedicated to "one set of stems".** Use separate instances for the
    search domain (excerpt) and the full song. Sharing one causes a rebind every time
    and is slower.
    """
    from mix_orchestrator.dsp.mix_state import MixState
    from mix_orchestrator.dsp.renderer import render
    from mix_orchestrator.dsp.loudness_norm import normalize_for_eval

    def _norm(state_or_audio) -> np.ndarray:
        if isinstance(state_or_audio, MixState):
            raw = (render(state_or_audio) if cache is None
                   else render(state_or_audio, cache=cache))
        else:
            raw = state_or_audio
        return normalize_for_eval(raw, sr, target_lufs=target_lufs)

    return _norm


def make_excerpt_norm_fn(norm_fn, preroll_samples: int, cache=None):
    """Normalization function for the excerpt domain. Discards the run-up region after
    rendering, then normalizes.

    With a run-up of 0 it returns the original ``norm_fn`` unchanged (the call count
    and the types are unchanged). The trim happens **before normalize**, because the
    stimulus builder also does "excerpt first, then re-normalize"
    ("Re-normalisation happens AFTER excerpting" in build_stimuli_v2) and the range
    over which LUFS is measured must coincide with the scoring window.
    """
    if preroll_samples <= 0:
        return norm_fn

    def _norm_trimmed(state_or_audio):
        from mix_orchestrator.dsp.mix_state import MixState
        from mix_orchestrator.dsp.renderer import render as _render
        if isinstance(state_or_audio, MixState):
            raw = (_render(state_or_audio) if cache is None
                   else _render(state_or_audio, cache=cache))
        else:
            raw = np.asarray(state_or_audio)
        # Passing a view through would add implicit copies downstream
        # (pyloudnorm / true-peak), so make it contiguous once here.
        return norm_fn(np.ascontiguousarray(raw[..., preroll_samples:]))

    return _norm_trimmed


# ============================================================================
# The real work that requires a GPU (lazy-imported inside main). Never called by
# the CPU tests.
# ============================================================================
def _run_one_song(row: Dict[str, str], run_dir: Path, args,
                  scorers: Dict[str, Any]) -> Dict[str, Any]:
    """One song: run the 4 baselines + the agreement loop, save the mixes, and return
    the result dict.

    scorers = {'pq': AudioboxPQScorer, 'sb': SongBenchMixingEar} (already lazily
    constructed).
    Uses a GPU, so it is out of scope for CPU unit tests (tests cover only the pure
    logic functions).
    """
    import soundfile as sf
    from mix_orchestrator.dsp.mix_state import MixState

    sid = row["song_id"]
    mixes_dir = run_dir / "mixes"
    mixes_dir.mkdir(parents=True, exist_ok=True)
    pq_scorer, sb_ear = scorers["pq"], scorers["sb"]
    sr_target_lufs = args.target_lufs

    t_song0 = time.perf_counter()
    stems, sr = load_one(args.dataset, song_id=sid,
                         duration_sec=args.duration_sec, seed=args.seed,
                         split=row["split"])
    main_stems = {k: v for k, v in stems.items() if k != "mixture"}

    # Cache for incremental rendering (non-None only when --render-cache is given).
    # **Attached to the search domain only.** The baselines / dry / final full-song
    # mix are each rendered exactly once, and MixState.initial_from_stems creates a
    # new ndarray every time, so the cache would always rebind = zero benefit.
    # Worse, the final rendering would hold prefixes for every accepted action (one
    # buffer is 78 MiB for a 220-second song) up to the full budget, so attaching it
    # would be pure harm.
    rc_search = make_render_cache(args)
    _norm = make_norm_fn(sr, sr_target_lufs, cache=None)

    def _save(method: str, audio: np.ndarray) -> None:
        sf.write(str(mixes_dir / mix_filename(method, sid)),
                 audio.T, sr, subtype="FLOAT")

    def _score(audio: np.ndarray) -> Tuple[float, float]:
        return float(pq_scorer.score(audio, sr)), float(sb_ear.score(audio, sr))

    # ---- baselines ----
    baseline_audio: Dict[str, np.ndarray] = {}
    baseline_scores: Dict[str, Dict[str, float]] = {}
    if not args.skip_baselines:
        baseline_audio.update(_build_baselines(main_stems, stems, sr, _norm))
        for m, au in baseline_audio.items():
            _save(m, au)
            pq, sb = _score(au)
            baseline_scores[m] = {"pq": pq, "sb": sb}

    # ---- Search domain (default = full song / 12-second excerpt with
    #      --search-excerpt-sec>0) ----
    excerpt_info = _resolve_excerpt_for_song(main_stems, sr, sid, row["split"], args)
    if excerpt_info is None:
        search_stems, full_for_render = main_stems, None
        # The search is on the full-song domain too. Keep the cache as a
        # search-dedicated instance (the baseline / random_dry renders create a new
        # ndarray with initial_from_stems every time, so putting them on the same
        # cache would mix in rebinds).
        _norm_search = (_norm if rc_search is None
                        else make_norm_fn(sr, sr_target_lufs, cache=rc_search))
        _norm_full = None
    else:
        search_stems = slice_stems(main_stems, excerpt_info["slice_start_sample"],
                                   excerpt_info["slice_n_samples"])
        full_for_render = main_stems
        # When preroll=0, make_excerpt_norm_fn returns its first argument unchanged,
        # so the inner norm_fn itself also has to be built with the search cache.
        _norm_inner = (_norm if rc_search is None
                       else make_norm_fn(sr, sr_target_lufs, cache=rc_search))
        _norm_search = make_excerpt_norm_fn(
            _norm_inner, excerpt_info["preroll_samples"], cache=rc_search)
        _norm_full = _norm

    # ---- KB initial mix + calibration + eval-gated greedy search (common pipeline) ----
    # The default non-random behaviour is completely unchanged: _propose_pipeline is
    # called with the same rng=default_rng(args.seed) and the same operation order as
    # before.
    pipe = _propose_pipeline(search_stems, sr, args, baseline_audio,
                             _norm_search, _score, seed=args.seed,
                             full_stems=full_for_render, full_norm_fn=_norm_full,
                             song_id=sid)
    _save(PROPOSED_METHOD, pipe["proposed_audio"])
    calib = pipe["calib"]

    return {
        "status": "done",
        "song_id": sid, "split": row["split"], "dataset": args.dataset,
        "sr": sr,
        **({"search_excerpt_sec": float(args.search_excerpt_sec),
            "search_excerpt": excerpt_info} if excerpt_info else {}),
        "calibration": {
            "pq_mean": calib.pq_mean, "pq_std": calib.pq_std,
            "sb_mean": calib.sb_mean, "sb_std": calib.sb_std,
            "n_samples": calib.n_samples,
            "sb_floored": calib.sb_floored, "pq_floored": calib.pq_floored,
            "sb_guard_reliable": calib.is_sb_guard_reliable,
        },
        "baselines": baseline_scores,
        "reward_form": pipe["reward_form"],
        "reward_weight": pipe["reward_weight"],
        "proposed": {"pq": pipe["proposed"]["pq"], "sb": pipe["proposed"]["sb"],
                     "reward": pipe["proposed"]["reward"],
                     "n_accepted": pipe["proposed"]["n_accepted"],
                     "best_chain": pipe["proposed"]["best_chain"]},
        "initial": pipe["initial"],
        # Without --render-cache the key is not written at all (the default JSON is
        # unchanged).
        **({"render_cache": {"search": rc_search.stats(),
                             "max_mb": float(args.render_cache_mb)}}
           if rc_search is not None else {}),
        "elapsed_sec": round(time.perf_counter() - t_song0, 2),
    }


def _propose_pipeline(main_stems: Dict[str, "np.ndarray"], sr: int, args,
                      baseline_audio: Dict[str, np.ndarray],
                      norm_fn, score_fn, seed: int,
                      full_stems: Optional[Dict[str, "np.ndarray"]] = None,
                      full_norm_fn=None,
                      search_score_fn=None,
                      song_id: str = "",
                      ) -> Dict[str, Any]:
    """Run KB initial mix -> per-song calibration -> eval-gated greedy search.

    Runs the existing proposal pipeline as is on the input ``main_stems``
    (randomized_stems in random mode). To preserve the same rng seed and the same
    operation order as the old non-random call path, the logic was extracted from the
    original _run_one_song without modification.

    Args:
        main_stems: stems of the **search domain**. In excerpt mode these are the
            stems cut to 12 seconds.
        full_stems: the **full-song** stems, passed only in excerpt mode. With None
            (default) the final render is done in the main_stems domain and returned,
            as before (= bit-identical to existing runs). Only when it is passed do we
            transplant the chain onto the full song after the search, render once, and
            re-score on the full song.
        full_norm_fn: normalization function for the full-song render. When ``norm_fn``
            is the excerpt-only version that discards the run-up region
            (--search-excerpt-preroll-sec>0), pass the plain normalization function
            here. Uses ``norm_fn`` when None.

    Returns:
        {'proposed_audio', 'proposed':{pq,sb,reward,n_accepted,best_chain},
         'initial':{pq,sb,reward}, 'calib', 'reward_form', 'reward_weight'}.
        In excerpt mode, proposed additionally gets pq_excerpt/sb_excerpt/
        reward_excerpt/search_domain/init_gain_db.
    """
    from mix_orchestrator.dsp.mix_state import MixState
    from mix_orchestrator.eval.agreement_reward import (
        fit_calibration, warn_if_weak_calibration)
    from mix_orchestrator.tools import action_tools as A
    from mix_orchestrator.strategies.knowledge_base_mix import build_knowledge_base_mix
    from agreement_loop_v2 import (_propose_action, _apply_action,
                                   random_automation_mode,
                                   RANDOM_AUTOMATION_ENV)

    # ---- Action space of the random proposer (added  ----------------
    # By default (AUTOMIX_RANDOM_AUTOMATION unset = 'off') _rand_kw is an empty dict,
    # making the call **exactly the same** as _propose_action(rng, tracks)
    # (bit-identical to existing runs; verified by
    # tools/check_propose_action_bitexact.py).
    #
    # Only when automation is enabled do we pass the proposer
    #   duration_sec : the length of the search domain (12 seconds in excerpt mode)
    #   beat_times   : the **same** beat time array that goes into the LLM prompt
    # The grid is taken from args so that we never create a state where only one side
    # knows the beats.
    _rand_auto = random_automation_mode()
    _rand_kw: Dict[str, Any] = {}
    if _rand_auto != "off":
        _dur = max(int(np.asarray(v).shape[-1])
                   for v in main_stems.values()) / float(sr)
        _rand_kw = {"duration_sec": _dur,
                    "beat_times": getattr(args, "excerpt_beat_times_rel", None)}
        # Accepting automation in excerpt mode puts absolute time into the chain, so
        # swap_state_stems (the transplant onto the full song) is guaranteed to fail.
        # Failing at the end of a 10000-step search is the worst case, so we reject
        # **before the search starts**.
        # This is the same constraint as on the LLM side
        # (AUTOMIX_ACTION_SET=automation), with the same remedy: run it as an
        # --excerpt-only run that reports the excerpt only.
        # Keep the resolution order identical to the transplant branch below
        # (_excerpt_only): when --excerpt-only is unspecified (= None), look at the
        # environment variable.
        _eo = getattr(args, "excerpt_only", None)
        if _eo is None:
            _eo = excerpt_only_default()
        if full_stems is not None and not _eo:
            raise ValueError(
                f"{RANDOM_AUTOMATION_ENV}={_rand_auto} is incompatible with the "
                "excerpt->full-song transplant (automation breakpoints carry "
                "absolute time, so swap_state_stems rejects them). "
                "Add --excerpt-only (or AUTOMIX_EXCERPT_ONLY=1) and run it as a run "
                "that reports only the 12-second excerpt result.")

    # ---- initial KB mix (eq=False) ----
    init_raw, init_meta = build_knowledge_base_mix(main_stems, sr, eq=False)
    init_state = MixState.initial_from_stems(main_stems, sr)
    # Record the KB initial gains so excerpt mode can be reproduced
    # (chain = init_gain + best_chain).
    init_gain_db: Dict[str, float] = {}
    for name, dec in init_meta["per_stem"].items():
        g = float(dec.get("gain_db", 0.0))
        if abs(g) > 1e-9:
            init_state = A.apply_static_gain(init_state, {"track": name, "gain_db": g})
            init_gain_db[name] = g
    init_audio = norm_fn(init_state)

    # ---- Calibration pool (dry + baselines + init + random) ----
    # In excerpt mode baseline_audio is built on the **full song**, so it must not be
    # mixed into the calib pool of the search domain (the excerpt). The PQ/SB
    # distributions shift systematically with audio length, so mixing them biases the
    # z-scores and pins which side binds in min(z_PQ, z_SB) (= the agreement reward
    # effectively degenerates into a single metric). Random mode already has
    # baseline_audio={}, so this branch is a guard for non-random mode only.
    rng = np.random.default_rng(seed)
    calib_pq: List[float] = []
    calib_sb: List[float] = []
    calib_pool = ([] if full_stems is not None else list(baseline_audio.values()))
    if full_stems is not None and baseline_audio:
        print(f"[excerpt] excluded {len(baseline_audio)} full-song baselines from "
              f"calibration (domain mismatch)", flush=True)
    for au in calib_pool + [init_audio]:
        pq, sb = score_fn(au)
        calib_pq.append(pq); calib_sb.append(sb)
    tracks = list(main_stems.keys())
    for _ in range(args.n_random_calib):
        # **Note**: the calibration pool draws from the same rng and the same
        # proposer. Enabling automation therefore changes the contents of the calib
        # pool and thus the z-score reference (mean/std). **The reward of the llm arm
        # changes too**, not just the random arm. Runs with and without automation
        # cannot be mixed, regardless of arm.
        nm, aargs = _propose_action(rng, tracks, **_rand_kw)
        try:
            st = _apply_action(init_state, nm, aargs)
            au = norm_fn(st)
            pq, sb = score_fn(au)
            calib_pq.append(pq); calib_sb.append(sb)
        except Exception:                                       # noqa: BLE001
            pass
    calib = fit_calibration(calib_pq, calib_sb)
    warn_if_weak_calibration(calib)

    reward_form = getattr(args, "reward_form", "min")
    reward_w = getattr(args, "reward_weight", 0.5)

    def _reward(pq: float, sb: float) -> float:
        return compute_reward_form(pq, sb, calib, form=reward_form, w=reward_w)

    init_pq, init_sb = score_fn(init_audio)
    init_reward = _reward(init_pq, init_sb)

    # ---- eval-gated greedy search ----
    # search_proposer hook (backward compatible): when unspecified (None) the old
    # random path is used as is, so the rng consumption order and the operation order
    # match exactly (bit-invariant).
    # Only when search_proposer is given (e.g. the LLM proposer of size_sweep_run) is
    # the proposal source replaced. The random path also builds `rejected`, but it is
    # unused and harmless.
    search_proposer = getattr(args, "search_proposer", None)
    best_state, best_reward = init_state, init_reward
    best_pq, best_sb = init_pq, init_sb
    n_accepted = 0
    best_chain: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    # ---- warm start (added  --------------------------------------
    # Apply the first N actions of an existing run's best_chain, then start searching.
    # **A way to probe the large-budget regime cheaply.** Running k=10000 from scratch
    # every time takes 10 hours per condition, but continuing for just 500 steps from
    # random's k=1000 endpoint measures "what works beyond the crossover (k~1000)" at
    # the cost of one ordinary wave.
    #
    # The applied actions are not put into best_chain (their step numbers would
    # collide with the new search); only the state and the reward baseline are
    # advanced. search_log covers only the new search. init_reward is also set to the
    # advanced value, so the total delta represents "the gain since the warm start".
    # **It means something different from the total delta of the original run, so do
    # not mix them.**
    # **Fail if only one of the two is given.** Silently running without a warm start
    # would produce results that look like "we measured the large-budget regime" and
    # cannot be told apart afterwards (on  a run actually completed without
    # it ever triggering).
    _wsrun = getattr(args, "warm_start_run", None)
    _wsn = int(getattr(args, "warm_start_steps", 0) or 0)
    if bool(_wsrun) != bool(_wsn > 0):
        raise SystemExit(
            f"--warm-start-run and --warm-start-steps must be given together "
            f"(run={_wsrun!r} steps={_wsn})")
    if _wsrun and _wsn > 0:
        if not song_id:
            raise SystemExit("warm start needs song_id (the caller failed to pass it)")
        _p = (ROOT / "outputs" / "runs" / _wsrun / "songs"
              / f"{safe_song_id(song_id)}.json")
        if not _p.is_file():
            raise SystemExit(f"warm start source not found: {_p}")
        _j = json.loads(_p.read_text("utf-8"))
        _pr = _j["seeds"][0].get("proposed") or _j["seeds"][0].get("proposed_rand") or {}
        _chain = [c for c in (_pr.get("best_chain") or [])
                  if int(c.get("step", 0)) < _wsn]
        for c in _chain:
            best_state = _apply_action(best_state, c["action"], c["args"])
        _au = norm_fn(best_state)
        best_pq, best_sb = (search_score_fn or score_fn)(_au)
        best_reward = _reward(best_pq, best_sb)
        init_reward = best_reward
        n_accepted = 0
        # **Always print the song name.** 22 processes write to the same log, so the
        # line order does not tell you which song a value belongs to (during the
        #  verification it became impossible to match them up).
        print(f"[warm] song={song_id} | resuming from run {_wsrun} at its first "
              f"{_wsn} actions: applied {len(_chain)} actions pq={best_pq:.4f} "
              f"reward={best_reward:+.4f}", flush=True)
    # Log of every proposal (saved to JSON only with --log-proposals. review
    #  item 4: used for diversity analysis over all proposals including
    # rejected ones, and for computing budget curves via the prefix property of greedy
    # search (the state at budget k = the first k steps of a 28-step run)).
    search_log: List[Dict[str, Any]] = []
    # ---- Intermediate saving (added , approved by the design) ----------
    # The song JSON is only written on completion, so if the job dies at the time
    # limit the trajectory of every step is lost (on  we came close to
    # losing 7 hours across k=10000 x 44 processes). Only when **both**
    # AUTOMIX_PARTIAL_DIR and AUTOMIX_PARTIAL_EVERY (in steps) are set do we write
    # search_log out incrementally every N steps.
    # Acceptance requires strict improvement, so by the prefix property an
    # intermediate log can be analysed directly as "the result at budget k".
    # Disabled by default (behaviour unchanged).
    _pdir = os.environ.get("AUTOMIX_PARTIAL_DIR", "")
    _pev = int(os.environ.get("AUTOMIX_PARTIAL_EVERY", "0") or 0)

    def _partial_save(step_now: int) -> None:
        if not (_pdir and _pev and step_now % _pev == 0):
            return
        try:
            # Split into subdirectories by run name. Different conditions of the same
            # wave (different runs) process the same songs, so with only the song name
            # **the conditions overwrite each other** (this actually happened on the
            # first submission on : 44 saves became 22 files).
            _rn = str(getattr(args, "run_name", "") or "run")
            d = Path(_pdir) / _rn
            d.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                d / f"{safe_song_id(song_id or 'unknown')}.json",
                {"song_id": song_id, "run_name": _rn, "steps_done": step_now,
                 "partial": True, "search_log": search_log})
        except Exception as ex:                                 # noqa: BLE001
            # Do not kill the search itself because an intermediate save failed
            # (disk full, etc.). Report on a single line.
            print(f"[partial] WARN save failed step={step_now}: {ex!r}", flush=True)
    # --- Progress display (added  ---------------------------------
    # **Does not change the search behaviour at all. Only prints.**
    # Back when n_search=28 a song finished in 10 minutes so this was unnecessary, but
    # at 10000 steps one song takes days. This function only writes JSON on song
    # completion, so there was no way to see progress and all one could do was "run it
    # and pray".
    # Every PROGRESS_EVERY steps, print the reward, the acceptance count, and the
    # breakdown of action types.
    #   - whether the reward is still improving (so we can decide to stop on saturation)
    #   - wall-clock time per step (to check whether extrapolating from 28 steps is right)
    #   - diversity of action types (at 28 steps Qwen converged onto gain and EQ and
    #     never once proposed stereo width; that is why it lost to random)
    # 0 disables it. Configurable via the environment variable.
    import collections as _collections
    import time as _time
    _prog_every = int(os.environ.get("PROGRESS_EVERY", "100") or 0)
    _t_start = _time.time()
    _kind_count: "_collections.Counter[str]" = _collections.Counter()
    for step in range(args.n_search):
        if _prog_every and step and step % _prog_every == 0:
            _el = _time.time() - _t_start
            _top = ", ".join("%s:%d" % (k, v)
                             for k, v in _kind_count.most_common(5))
            # The song name can be read off the preceding "[all] (i/n) split song_id"
            # line (we submit one song per job). Here we print only the seed and the
            # step count.
            print("[prog] seed=%d step=%d/%d accepted=%d "
                  "reward=%.4f pq=%.4f sb=%.4f "
                  "%.2f s/step elapsed=%.1fh remaining~%.1fh actions=[%s]"
                  % (seed, step, args.n_search, n_accepted,
                     best_reward, best_pq, best_sb,
                     _el / step, _el / 3600.0,
                     (_el / step) * (args.n_search - step) / 3600.0, _top),
                  flush=True)
        # CRN (common random numbers): draw a random proposal every step so that the
        # rng consumption is constant regardless of the proposer type. This makes the
        # random tail of a hybrid proposer follow the same rng sequence as a
        # pure-random run (same seed, same calib consumption), which maximizes the
        # power of the paired comparison (review .
        #   - pure-random (search_proposer None): use this proposal as is -> bit-identical
        #     to before
        #   - pure-LLM / the LLM steps of hybrid: discard rand_proposal and use the LLM
        #     proposal. The rng is not used after the search, so the final result is
        #     unchanged (reproducibility preserved).
        #   - the random steps of hybrid (step>=switch): return rand_proposal -> exactly
        #     the same random action sequence as pure-random (seed_idx is symmetrized
        #     automatically too).
        #
        # **Adding automation makes CRN compatibility hold only within a set of runs**:
        # the rng consumption order of _propose_action changes, so the proposal
        # sequence of a run with AUTOMIX_RANDOM_AUTOMATION enabled diverges from
        # existing runs from the very first step. Step-wise paired comparison across
        # enabled/disabled (variance cancellation of the difference) does not hold.
        # CRN between identical settings (the 4 reward arms / llm vs random) works as
        # before.
        rand_nm, rand_args = _propose_action(rng, tracks, **_rand_kw)
        if search_proposer is not None:
            nm, aargs = search_proposer(
                best_state, best_pq, best_sb, best_reward, calib,
                best_chain, rejected, step, args.n_search, tracks, sr,
                rand_proposal=(rand_nm, rand_args))
        else:
            nm, aargs = rand_nm, rand_args
        if nm is None:
            search_log.append({"step": step, "action": None, "note": "no_proposal"})
            continue
        try:
            cand = _apply_action(best_state, nm, aargs)
        except Exception:                                       # noqa: BLE001
            rejected.append({"action": nm, "args": aargs, "note": "apply failed"})
            search_log.append({"step": step, "action": nm, "args": aargs,
                               "note": "apply_failed"})
            continue
        au = norm_fn(cand)
        # Scoring dedicated to the search loop. The default (None) is score_fn itself,
        # bit-identical to existing runs. When AUTOMIX_NO_SB_IN_SEARCH=1 the caller
        # passes a function that "measures PQ only and returns nan for sb"
        # (reward_form=pq_only only).
        # The final full-song scoring (the score_fn call below) is left untouched, so
        # **the SongBench value used for reporting is measured exactly once per song**.
        pq, sb = (search_score_fn or score_fn)(au)
        r = _reward(pq, sb)
        accepted = r > best_reward
        _kind_count[str(nm).replace("apply_static_", "").replace("apply_", "")] += 1
        search_log.append({"step": step, "action": nm, "args": aargs,
                           "pq": pq, "sb": sb, "reward": r,
                           "accepted": bool(accepted)})
        _partial_save(step)
        if accepted:
            delta = r - best_reward
            best_state, best_reward, best_pq, best_sb = cand, r, pq, sb
            n_accepted += 1
            best_chain.append({"action": nm, "args": aargs, "step": step,
                               "pq": pq, "sb": sb, "reward": r, "delta": delta})
        else:
            # Attach step and delta (added .
            # delta = how far below the current best the proposal fell (a negative
            # value). Without it the prompt cannot distinguish "a near-miss rejection"
            # from "a badly-off rejection". There was an asymmetry where accepted
            # actions carried a delta but rejected ones did not. step is used to decide
            # "which of the two the immediately preceding action was".
            # **The default prompt string does not change** (whether it is shown is
            # selected by last_outcome / rejected_delta in AUTOMIX_PROMPT_HISTORY).
            rejected.append({"action": nm, "args": aargs, "reward": r,
                             "step": step, "delta": r - best_reward})
    # ---- Final mix ----
    # Default (full_stems=None): render in the search domain and return, as before.
    #   -> bit-identical to existing runs (the body of this branch is the original code).
    # Excerpt mode (full_stems given): transplant the chain obtained by the search
    #   directly onto the full-song stems and render + score the full song **exactly
    #   once**. This is the core of dropping O(K^2) to O(K); the reported values
    #   (pq/sb) stay full-song, so they remain comparable with the existing tables.
    #   The excerpt values are recorded alongside in *_excerpt.
    # Excerpt mode + --excerpt-only: **do not** transplant, because if the chain
    #   carries absolute time (gain automation, etc.) the premise of the transplant
    #   does not hold. The reported values become the PQ/SB of the 12-second excerpt.
    #   See the discussion of EXCERPT_ONLY_ENV for details.
    _excerpt_only = getattr(args, "excerpt_only", None)
    if _excerpt_only is None:
        _excerpt_only = excerpt_only_default()
    initial_rec: Dict[str, Any] = {"pq": init_pq, "sb": init_sb,
                                   "reward": init_reward}
    if full_stems is None:
        proposed_audio = norm_fn(best_state)
        proposed: Dict[str, Any] = {
            "pq": best_pq, "sb": best_sb, "reward": best_reward,
            "n_accepted": n_accepted, "best_chain": best_chain}
    elif _excerpt_only:
        # Return the excerpt itself as the final deliverable. Neither full-song
        # rendering nor full-song scoring is performed.
        # pq/sb/reward are **the excerpt values**. The same values are also put into
        # *_excerpt so that an aggregation script sees excerpt values whichever key it
        # reads (this structurally prevents the accident of misreading a full-song
        # value).
        _reason = time_absolute_reason(best_state)
        proposed_audio = norm_fn(best_state)
        proposed = {
            "pq": best_pq, "sb": best_sb, "reward": best_reward,
            "n_accepted": n_accepted, "best_chain": best_chain,
            "search_domain": "excerpt",
            # **Explicit statement that the reporting domain is the excerpt.** The
            # pq/sb of a run where this is 'excerpt' must not be placed in the same
            # column as the pq/sb of a full-song run.
            "report_domain": "excerpt",
            "no_full_transplant": True,
            "no_full_transplant_reason": (
                _reason or "--excerpt-only given (the chain is time-invariant, but the "
                           "transplant was skipped)"),
            "pq_excerpt": best_pq, "sb_excerpt": best_sb,
            "reward_excerpt": best_reward,
            "reward_calib_domain": "excerpt",
            "init_gain_db": dict(init_gain_db),
        }
        initial_rec["domain"] = "excerpt"
    else:
        full_state = swap_state_stems(best_state, full_stems)
        proposed_audio = (full_norm_fn or norm_fn)(full_state)
        full_pq, full_sb = score_fn(proposed_audio)
        proposed = {
            "pq": full_pq, "sb": full_sb, "reward": _reward(full_pq, full_sb),
            "n_accepted": n_accepted, "best_chain": best_chain,
            # --- Excerpt-mode metadata (absent in the default mode) ---
            "search_domain": "excerpt",
            "pq_excerpt": best_pq, "sb_excerpt": best_sb,
            "reward_excerpt": best_reward,
            # reward here is "the full-song pq/sb z-scored with **the calib fitted on
            # the excerpt**". Its domain differs from the search objective itself
            # (= reward_excerpt), so we state it explicitly.
            "reward_calib_domain": "excerpt",
            "init_gain_db": dict(init_gain_db),
        }
        initial_rec["domain"] = "excerpt"
    if getattr(args, "log_proposals", False):
        proposed["search_log"] = search_log
        proposed["init_reward"] = init_reward
    return {
        "proposed_audio": proposed_audio,
        "proposed": proposed,
        "initial": initial_rec,
        "calib": calib,
        "reward_form": reward_form,
        "reward_weight": reward_w if reward_form == "weighted" else None,
    }


def _run_one_song_random(row: Dict[str, str], song_index: int, run_dir: Path,
                         args, scorers: Dict[str, Any]) -> Dict[str, Any]:
    """Randomized-stem recovery experiment (1 song, multiple seeds).

    For each seed, apply a linear gain U(LOW,HIGH) to main_stems to break the
    professional balance, and score these 2 conditions with PQ / SongBench:
      - random_dry    = sum randomized_stems in a plain state -> norm(-14)
      - proposed_rand = run the proposal pipeline (KB initial mix + eval-gated search)
                        with randomized_stems as input
    The mixes are saved under mixes/ (random_dry__ / proposed_rand__), and the gain
    multipliers, seeds, and per-condition scores are recorded in the per-song JSON.
    Uses a GPU, so it is out of scope for CPU unit tests (only the pure logic is
    tested).
    """
    import soundfile as sf
    from mix_orchestrator.dsp.mix_state import MixState

    sid = row["song_id"]
    mixes_dir = run_dir / "mixes"
    mixes_dir.mkdir(parents=True, exist_ok=True)
    pq_scorer, sb_ear = scorers["pq"], scorers["sb"]
    sr_target_lufs = args.target_lufs
    gain_db_range = getattr(args, "random_gain_db_range", None)
    if gain_db_range is not None:
        if getattr(args, "random_init_gain_range", None) is not None:
            raise ValueError(
                "--random-gain-db-range and --random-init-gain-range are mutually "
                "exclusive")
        low, high = float(gain_db_range[0]), float(gain_db_range[1])
        gain_mode = "db"
    else:
        low, high = float(args.random_init_gain_range[0]), \
            float(args.random_init_gain_range[1])
        gain_mode = "linear"

    t_song0 = time.perf_counter()
    stems, sr = load_one(args.dataset, song_id=sid,
                         duration_sec=args.duration_sec, seed=args.seed,
                         split=row["split"])
    main_stems = {k: v for k, v in stems.items() if k != "mixture"}

    # The incremental-rendering cache is **dedicated to the search domain**, and
    # randomized_stems changes per seed, so it is rebuilt inside the seed loop.
    # _norm (full song, uncached) is for random_dry / baselines / the final mix.
    rc_search = None
    _norm = make_norm_fn(sr, sr_target_lufs, cache=None)

    def _save(method: str, suffix: str, audio: np.ndarray) -> None:
        # kad-compatible naming: <method>__<safe_song_id>__<suffix>.wav. The
        # convention that '__' is not allowed inside method is kept, and the seed
        # identifier is attached on the song_id side instead.
        sf.write(str(mixes_dir / mix_filename(method, f"{sid}__{suffix}")),
                 audio.T, sr, subtype="FLOAT")

    def _score(audio: np.ndarray) -> Tuple[float, float]:
        return float(pq_scorer.score(audio, sr)), float(sb_ear.score(audio, sr))

    # Scoring dedicated to the search loop. When AUTOMIX_NO_SB_IN_SEARCH=1, SongBench
    # is not called and sb=nan is returned. **Only for reward_form=pq_only**: with
    # min/weighted the reward would become nan and break the search, so we reject that
    # explicitly below.
    # The final full-song scoring stays _score, so the SongBench value used for
    # reporting is still measured once per song.
    _rf = getattr(args, "reward_form", "min")
    _no_sb = _no_sb_in_search(default=(_rf == "pq_only"))
    if _no_sb and _rf != "pq_only":
        raise SystemExit(
            f"AUTOMIX_NO_SB_IN_SEARCH=1 is only for --reward-form pq_only "
            f"(given: {getattr(args, 'reward_form', None)!r}). "
            "With min/weighted/sb_only the reward becomes nan and the search breaks.")

    def _score_pq_only(audio: np.ndarray) -> Tuple[float, float]:
        return float(pq_scorer.score(audio, sr)), float("nan")

    _search_score = _score_pq_only if _no_sb else None

    # Search domain. The window is decided from the dry stems, so it is one window per
    # song regardless of the corruption seed.
    excerpt_info = _resolve_excerpt_for_song(main_stems, sr, sid, row["split"], args)

    # --random-base-seed (added : an additive shift to the base seed. With
    # the default 0, base_seed=args.seed, identical to before (the seed sequence is
    # bit-identical).
    _rbs = int(getattr(args, "random_base_seed", 0) or 0)

    seed_results: List[Dict[str, Any]] = []
    for seed_idx in range(int(args.random_seeds)):
        seed = random_seed_for(song_index, seed_idx,
                               base_seed=int(args.seed) + _rbs)
        gain_rng = np.random.default_rng(seed)
        gains = sample_random_gains(
            list(main_stems.keys()), gain_rng, low, high,
            gain_db_range=(low, high) if gain_mode == "db" else None)
        randomized_stems = apply_stem_gains(main_stems, gains)
        # Normalize the absolute level after corruption with a common scalar (the
        # relative balance is unchanged). Prevents artifacts from the absolute-dBFS
        # silence gates in MEGAMI and others. Only when specified.
        norm_lufs = getattr(args, "normalize_corrupted_lufs", None)
        if norm_lufs is not None:
            from mix_orchestrator.dsp.loudness_norm import common_gain_to_lufs
            randomized_stems, common_gain = common_gain_to_lufs(
                randomized_stems, sr, target_lufs=float(norm_lufs))
        else:
            common_gain = None

        suffix = f"seed{seed_idx}"
        # The actual stems change per seed, so the cache is rebuilt too (so we do not
        # keep holding the previous seed's 78 MiB-class buffers). None without
        # --render-cache.
        # **Search domain only.** random_dry / baselines / the final full-song mix are
        # each rendered exactly once, so a cache brings no benefit and only leaves the
        # harm of holding one prefix per accepted action during the final rendering.
        rc_search = make_render_cache(args)

        # random_dry = sum the corrupted stems in a plain state -> norm
        random_dry_audio = _norm(MixState.initial_from_stems(randomized_stems, sr))
        rd_pq, rd_sb = _score(random_dry_audio)
        _save("random_dry", suffix, random_dry_audio)

        # Excerpt mode: run the search on the excerpt stems and return the chain
        # transplanted onto the full song.
        # random_dry is also scored in the excerpt domain (without a starting point
        # comparable under the same calib as the search reward, reward_excerpt cannot
        # be read on its own).
        if excerpt_info is None:
            search_stems, full_for_render = randomized_stems, None
            _norm_search = (_norm if rc_search is None
                            else make_norm_fn(sr, sr_target_lufs, cache=rc_search))
            _norm_full = None
            rd_exc = None
        else:
            search_stems = slice_stems(randomized_stems,
                                       excerpt_info["slice_start_sample"],
                                       excerpt_info["slice_n_samples"])
            full_for_render = randomized_stems
            _norm_inner = (_norm if rc_search is None
                           else make_norm_fn(sr, sr_target_lufs, cache=rc_search))
            _norm_search = make_excerpt_norm_fn(
                _norm_inner, excerpt_info["preroll_samples"], cache=rc_search)
            _norm_full = _norm
            rd_exc = _score(_norm_search(
                MixState.initial_from_stems(search_stems, sr)))

        # proposed_rand = the proposal pipeline with the corrupted stems as input
        pipe = _propose_pipeline(search_stems, sr, args,
                                 baseline_audio={}, norm_fn=_norm_search,
                                 score_fn=_score, seed=seed,
                                 full_stems=full_for_render,
                                 full_norm_fn=_norm_full,
                                 search_score_fn=_search_score,
                                 song_id=sid)
        _save("proposed_rand", suffix, pipe["proposed_audio"])
        calib = pipe["calib"]
        # Re-evaluate the reward of random_dry with the same per-song calib as
        # proposed (so both conditions can be compared on the same z reference).
        rd_reward = compute_reward_form(
            rd_pq, rd_sb, calib, form=pipe["reward_form"],
            w=getattr(args, "reward_weight", 0.5))
        random_dry_rec: Dict[str, Any] = {"pq": rd_pq, "sb": rd_sb,
                                          "reward": rd_reward}
        if rd_exc is not None:
            # random_dry in the excerpt domain. It is in the same domain as the calib,
            # so it serves as a starting point directly comparable with
            # proposed_rand.reward_excerpt.
            random_dry_rec.update({
                "pq_excerpt": rd_exc[0], "sb_excerpt": rd_exc[1],
                "reward_excerpt": compute_reward_form(
                    rd_exc[0], rd_exc[1], calib, form=pipe["reward_form"],
                    w=getattr(args, "reward_weight", 0.5)),
            })

        # Baselines on randomized stems: a fair comparison where every method restores
        # from the same corrupted input. MEGAMI (learned) / P2 (rule-based) / P6 /
        # equal_lufs are re-mixed from randomized_stems and their reward is evaluated
        # with the same per-song calib as proposed_rand. 'dry' is excluded because it
        # is identical to random_dry.
        baselines_rand: Dict[str, Dict[str, float]] = {}
        if not getattr(args, "skip_baselines", False):
            try:
                bl = _build_baselines(randomized_stems, randomized_stems, sr, _norm)
            except Exception as ex:                              # noqa: BLE001
                print(f"[all] WARN random baselines skipped {sid} s{seed_idx}: {ex!r}")
                bl = {}
            for m, au in bl.items():
                if m == "dry":
                    continue
                _save(m, suffix, au)
                bpq, bsb = _score(au)
                breward = compute_reward_form(
                    bpq, bsb, calib, form=pipe["reward_form"],
                    w=getattr(args, "reward_weight", 0.5))
                baselines_rand[m] = {"pq": bpq, "sb": bsb, "reward": breward}

        seed_results.append({
            "seed_idx": seed_idx,
            "seed": seed,
            "gain_range": [low, high],
            "gain_mode": gain_mode,
            "common_gain": common_gain,
            "stem_gains": gains,
            "random_dry": random_dry_rec,
            "baselines_rand": baselines_rand,
            # How well incremental rendering worked (telemetry that has no effect
            # whatsoever on the search behaviour). Without --render-cache **the key
            # itself is not written**, so the JSON of a default run does not change by
            # a single byte.
            **({"render_cache": {"search": rc_search.stats(),
                                 "max_mb": float(args.render_cache_mb)}}
               if rc_search is not None else {}),
            # Save pipe["proposed"] as is (pq/sb/reward/n_accepted/best_chain +
            # search_log/init_reward with --log-proposals. A selective copy would drop
            # new keys —  fix).
            "proposed_rand": dict(pipe["proposed"]),
            "initial": pipe["initial"],
            "calibration": {
                "pq_mean": calib.pq_mean, "pq_std": calib.pq_std,
                "sb_mean": calib.sb_mean, "sb_std": calib.sb_std,
                "n_samples": calib.n_samples,
                "sb_floored": calib.sb_floored, "pq_floored": calib.pq_floored,
            },
        })

    return {
        "status": "done",
        "mode": "random",
        "song_id": sid, "song_index": song_index,
        "split": row["split"], "dataset": args.dataset, "sr": sr,
        "random_init_gain_range": [low, high],
        "gain_mode": gain_mode,
        "normalize_corrupted_lufs": getattr(args, "normalize_corrupted_lufs", None),
        "random_seeds": int(args.random_seeds),
        # Write the key only when --random-base-seed is given (the JSON of a default
        # run does not change by a single byte; same policy as render_cache). Resume
        # consistency is checked by assert_corruption_consistent treating
        # "missing = 0".
        **({"random_base_seed": _rbs} if _rbs else {}),
        "reward_form": getattr(args, "reward_form", "min"),
        # --- Reproducibility metadata (added . New writes only; existing
        # JSONs are unchanged) ---
        # reward_weight does not contribute to the reward outside of weighted, so it
        # is set to None to avoid misreading "the unused default 0.5" as the effective
        # value.
        "reward_weight": (float(getattr(args, "reward_weight", 0.5))
                          if getattr(args, "reward_form", "min") == "weighted"
                          else None),
        "n_search": int(args.n_search),
        "sb_chunk_sec": sb_chunk_sec_effective(),
        # --- Excerpt-search metadata (written only when --search-excerpt-sec>0; the
        # JSON of a default run does not change by a single byte) ---
        **({"search_excerpt_sec": float(args.search_excerpt_sec),
            "search_excerpt": excerpt_info,
            # baselines_rand / random_dry.pq/sb stay full-song. Their rewards are
            # z-scored with the calib fitted on the excerpt, so the domain is stated
            # explicitly.
            "reward_calib_domain": "excerpt",
            # In an --excerpt-only run, proposed_rand.pq/sb become **the excerpt
            # values**. random_dry / baselines_rand stay full-song, so the two must not
            # be compared in the same column. Write it at the song level too so they
            # can be told apart.
            "excerpt_only": bool(
                getattr(args, "excerpt_only", None)
                if getattr(args, "excerpt_only", None) is not None
                else excerpt_only_default()),
            "excerpt_beats": bool(getattr(args, "excerpt_beats", False)),
            } if excerpt_info else {}),
        "seeds": seed_results,
        "elapsed_sec": round(time.perf_counter() - t_song0, 2),
    }


def _build_baselines(main_stems, stems, sr, norm_fn) -> Dict[str, np.ndarray]:
    """Return the normalized audio for dry / equal_lufs / P2 / P6 / MEGAMI.

    Baselines that fail are omitted (the eval side treats a missing method as
    nan/skip).
    """
    import asyncio
    from mix_orchestrator.dsp.mix_state import MixState
    from mix_orchestrator.dsp.loudness_norm import equal_lufs_mix
    out: Dict[str, np.ndarray] = {}
    # dry = sum the stems in a plain state (render the initial state)
    out["dry"] = norm_fn(MixState.initial_from_stems(main_stems, sr))
    # equal_lufs = naive baseline that just matches every stem to the same LUFS and sums
    try:
        out["equal_lufs"] = norm_fn(equal_lufs_mix(main_stems, sr))
    except Exception as ex:                                     # noqa: BLE001
        print(f"[all] WARN equal_lufs skipped: {ex!r}")
    # P2 = analysis_driven
    try:
        from mix_orchestrator.strategies.analysis_driven import build_analysis_driven_mix
        p2_raw, _ = asyncio.run(build_analysis_driven_mix(main_stems, sr))
        out["P2"] = norm_fn(p2_raw)
    except Exception as ex:                                     # noqa: BLE001
        print(f"[all] WARN P2 skipped: {ex!r}")
    # P6 = profile_matched (requires the target library)
    try:
        out["P6"] = _build_p6(stems, sr, norm_fn)
    except Exception as ex:                                     # noqa: BLE001
        print(f"[all] WARN P6 skipped: {ex!r}")
    # MEGAMI baseline (prior method 1)
    try:
        out["MEGAMI"] = _build_megami(main_stems, sr, norm_fn)
    except Exception as ex:                                     # noqa: BLE001
        print(f"[all] WARN MEGAMI skipped: {ex!r}")
    # FxNorm baseline (prior method 2, Martinez-Ramirez et al. ISMIR2022)
    try:
        out["FxNorm"] = _build_fxnorm(main_stems, sr, norm_fn)
    except Exception as ex:                                     # noqa: BLE001
        print(f"[all] WARN FxNorm skipped: {ex!r}")
    return out


# The FxNorm model is heavy (build ~14s), so it is constructed once and reused for
# every song.
_FXNORM_INSTANCE = None


def _build_fxnorm(main_stems, sr, norm_fn) -> np.ndarray:
    """Mix main_stems with FxNorm-automix (prior method 2) and return normalized audio."""
    global _FXNORM_INSTANCE
    from mix_orchestrator.baselines.fxnorm_baseline import FXNormBaseline
    if _FXNORM_INSTANCE is None:
        _FXNORM_INSTANCE = FXNormBaseline()
    mix, _meta = _FXNORM_INSTANCE.mix(main_stems, sr)   # (audio, meta)
    return norm_fn(mix)


def _build_p6(stems, sr, norm_fn) -> np.ndarray:
    import glob
    from mix_orchestrator.agent.experience_library import ExperienceLibrary
    from mix_orchestrator.strategies.target_profile import build_profile_matched_mix
    from mix_orchestrator.strategies.analysis_driven import _role_of
    lib = ExperienceLibrary()
    for f in glob.glob(str(ROOT / "configs/agent/target_profiles*.jsonl")):
        lib.cases.extend(ExperienceLibrary.load(f).cases)
    main = {k: v for k, v in stems.items() if k != "mixture"}
    roles = sorted({_role_of(n) for n in main})
    mix, _meta = build_profile_matched_mix(stems, sr, lib, genre="default",
                                           roles=roles, k=1)
    if mix is None:
        raise RuntimeError("profile_matched mix returned None")
    return norm_fn(mix)


def _build_megami(main_stems, sr, norm_fn) -> np.ndarray:
    from mix_orchestrator.baselines.megami_baseline import MegamiBaseline
    mb = MegamiBaseline()
    mix, _meta = mb.mix(main_stems, sr)   # ExternalBaseline.mix -> (audio, meta)
    return norm_fn(mix)


def build_full_plan_song_index(splits: List[str], dataset: str = "musdb18",
                               enumerate_fn: Optional[Callable[..., List[Optional[str]]]] = None
                               ) -> Dict[str, int]:
    """Return the song_id -> song_index mapping based on the full plan, before
    offset/limit is applied.

    The corruption seed in random mode is derived from song_index, so this index must
    be unique regardless of chunking (offset/limit) or pilot selection.
    (With a chunk-local index the seeds would collide with songs in other chunks —
    review 
    """
    full_plan = build_song_plan(splits, dataset=dataset, enumerate_fn=enumerate_fn,
                                offset=0, limit=None)
    return {r["song_id"]: i for i, r in enumerate(full_plan)}


def _write_json_atomic(path: Path, obj: Any) -> None:
    # Include the pid in the tmp name so that concurrent processes (backfill chunks,
    # etc.) writing to the same file do not collide on the tmp file
    # (review fix .
    # Note that concurrent writes to the same target are last-writer-wins (full
    # replacement), not a merge — concurrency is only acceptable when both write
    # identical content.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    splits = parse_splits(args.splits)
    run_dir = ROOT / "outputs" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Resume consistency of the search domain (full song / N-second excerpt). In a
    # default run both want and have are 0.0, so this is a no-op.
    assert_search_domain_consistent(run_dir, args)

    plan = build_song_plan(splits, dataset=args.dataset,
                           offset=args.offset, limit=args.limit)
    pending = filter_pending(plan, run_dir, resume=args.resume)
    random_mode = is_random_mode(args)
    # To keep the random-mode seeds stable regardless of resume/chunking
    # (offset/limit), song_index is built from the **full plan before offset/limit is
    # applied**. The old implementation enumerated the plan after the window was cut,
    # which had the defect that song_index became chunk-local under --offset chunked
    # execution (agree_random_dbfs_v1 ran in 2 chunks and the corruption seeds
    # collided across chunks — review . Aggregation of completed runs uses
    # the stored values, so it is unaffected. Runs without offset (size sweeps, etc.)
    # get the same index as the old implementation.
    song_index_of = build_full_plan_song_index(splits, dataset=args.dataset)
    print(f"[all] splits={splits} plan={len(plan)} pending={len(pending)} "
          f"(offset={args.offset} limit={args.limit} resume={args.resume} "
          f"random={random_mode})")
    if random_mode:
        _db = getattr(args, "random_gain_db_range", None)
        if _db is not None:
            _gain_desc = f"gain_db_range={tuple(_db)} (dBFS)"
        else:
            _gain_desc = f"gain_range={tuple(args.random_init_gain_range)} (linear)"
        print(f"[all] RANDOM mode: {_gain_desc} "
              f"seeds={args.random_seeds} run={args.run_name}")
    for r in pending:
        print(f"    - {r['split']:5s} {r['song_id']}")

    if args.dry_run:
        print("[all] --dry-run: exiting without scoring (no GPU needed)")
        _write_json_atomic(run_dir / "plan.json",
                           {"splits": splits, "plan": plan, "pending": pending,
                            "random_mode": random_mode})
        return 0

    scorers = build_scorers(getattr(args, "scorer", "local"),
                            getattr(args, "scorer_addr", None),
                            int(getattr(args, "scorer_client_index", 0) or 0))

    index_path = run_dir / "index.json"
    done_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(pending):
        print(f"[all] ({i+1}/{len(pending)}) {row['split']} {row['song_id']}",
              flush=True)
        try:
            if random_mode:
                res = _run_one_song_random(
                    row, song_index_of[row["song_id"]], run_dir, args, scorers)
            else:
                res = _run_one_song(row, run_dir, args, scorers)
        except Exception as ex:                                 # noqa: BLE001
            res = {"status": "error", "song_id": row["song_id"],
                   "split": row["split"], "error": repr(ex)}
            print(f"[all] ERROR {row['song_id']}: {ex!r}", flush=True)
        _write_json_atomic(song_result_path(run_dir, row["song_id"]), res)
        done_rows.append({"song_id": row["song_id"], "split": row["split"],
                          "status": res["status"]})
        _write_json_atomic(index_path, {"run_name": args.run_name,
                                        "splits": splits, "rows": done_rows,
                                        "random_mode": random_mode})
    print(f"[all] done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
