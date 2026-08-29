"""Memoization of the LLM proposer's per-step context (per-stem LUFS).

Background (measured , 220 s track / CPU)
---------------------------------------------------
The LLM proposer calls ``_per_stem_lufs`` once per step. That call renders the
4 stems **individually and runs pyloudnorm 4 times**, which is heavier than
rendering a candidate::

    depth  0:  _per_stem_lufs  3.752 s   vs  render  0.463 s   (8.1x)
    depth 20:                 13.223 s   vs          9.937 s
    depth 50:                 21.219 s   vs         18.007 s

Its input ``best_state``, on the other hand, **only changes on acceptance**.
The acceptance rate is about 20% for k<=100 and about 9% at k=5000, so
recomputing only on acceptance already removes 80-90% of the work. On top of
that, a single acceptance changes **only the chain of one track**
(``MixState.add_static`` / ``add_transient`` / ``add_section`` build
``{**old, track: new_chain}``, so the chain tuples of the other tracks are
carried over as **the very same objects**), so with per-stem memoization even
an accepted step re-renders only 1 of the 4 stems.

    expected renders / step:  4  ->  acceptance rate (about 0.09-0.20)   => about 95% reduction

Grounds for bit-exactness (the top-priority requirement)
--------------------------------------------------------
1. ``MixState`` is ``@dataclass(frozen=True)``. The ndarrays in
   ``state.stems`` are never rewritten during the search (every action in
   ``tools/action_tools.py`` uses only ``add_static`` / ``add_section`` /
   ``add_transient`` / ``add_master`` and never touches ``stems``).
2. ``_render_single_stem(state, name)`` reads only

       state.stems[name] / static_processors[name] / transient_processors[name]
       / section_processors[name] / sample_rate / section_boundaries

   (``master_chain`` is explicitly forced down to ``()``). Every effect is a
   ``@dataclass(frozen=True)``, i.e. hashable and compared by value, so the
   above can be used directly as a dict key.
3. render (pedalboard / numba kernels with fastmath=False) and pyloudnorm are
   deterministic. A memo hit is therefore "the same pure function on the same
   input", and its return value is **bit-identical** to the uncached
   implementation.
4. Identity of stems is checked on every call as "identity of the dict object
   + identity of each ndarray object"; if it differs the cache is dropped.
   This structurally rules out false hits across tracks (and no id() reuse
   accident can happen either: the snapshot holds strong references).

Verification: ``experiments/perf/verify_stem_lufs_cache.py`` (CPU only).

Memoization alone leaves an O(K^2) term (added 
----------------------------------------------------------
The memo only helps on "steps with no acceptance". On a step with an
acceptance the chain of that one track grows, so its key changes and **that
stem is re-rendered over the full length of its chain**. The
A(K)=0.394*K^0.866 acceptances are spread over n stems, so the solo render
cost of the whole trajectory is

    Σ_stem Σ_{i=1..A/n} i  =  n * (A/n)^2 / 2  =  A^2 / (2n)   [effect applications]

At K=10000 (A=1147, n=4) that is about 164,000 applications. On a 220 s track
one application takes about 0.32 s, i.e. **about 15 hours per track**. Killing
the O(K^2) on the render side with incremental rendering is meaningless as
long as a term of the same order remains here.

As a countermeasure, passing ``render_cache=True`` keeps **a separate
``RenderCache`` per stem name** and makes solo render incremental as well
(1 acceptance = 1 effect application). The output is bit-identical (the
invariants of ``RenderCache`` carry over unchanged). The reason for a separate
instance per stem is that a solo state holds only one stem, so sharing the
cache used for the full mix would rebind on every call and end up slower
instead.

Environment variables
---------------------
``AUTOMIX_STEM_LUFS_CACHE=0`` disables it (for bisecting regressions). The
default is enabled.
``AUTOMIX_STEM_LUFS_RENDER_CACHE_MB`` sets the incremental-rendering budget
per stem in [MiB] (default 512; the required working set is only 2-3 buffers,
so that is plenty).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

CACHE_ENV = "AUTOMIX_STEM_LUFS_CACHE"
RENDER_CACHE_MB_ENV = "AUTOMIX_STEM_LUFS_RENDER_CACHE_MB"

#: Incremental-rendering budget per stem [MiB]. The prefix working set only
#: needs 1-2 buffers ("the immediately preceding prefix"), so even for 220 s
#: stereo (78 MiB per buffer) 512 MiB is enough that no eviction occurs.
_DEFAULT_RENDER_CACHE_MB = 512.0

_FALSEY = ("0", "false", "no", "off")

# Sentinel used as the default for dict.get. A dedicated object rather than
# None, so that a stored value of NaN is still distinguishable from a "hit".
_MISS = object()


def cache_enabled_by_env() -> bool:
    """Read ``AUTOMIX_STEM_LUFS_CACHE``. Enabled (True) when unset."""
    v = os.environ.get(CACHE_ENV)
    if v is None:
        return True
    return v.strip().lower() not in _FALSEY


def _one_stem_lufs(state, name: str, sr: int, render_cache=None) -> float:
    """Solo render a single stem and return its integrated LUFS (identical to
    the reference implementation).

    ``render_cache`` is a ``RenderCache`` **dedicated to this stem** (or None).
    Passing it keeps the output bit-identical (invariant of incremental
    rendering).
    """
    from ..strategies.knowledge_base_mix import _render_single_stem
    from .loudness_norm import _measure_lufs
    try:
        single = _render_single_stem(state, name, cache=render_cache)
        lufs = _measure_lufs(single, sr)
    except Exception:                                             # noqa: BLE001
        lufs = float("nan")
    return float(lufs)


def compute_per_stem_lufs(state, sr: int) -> Dict[str, Dict[str, float]]:
    """Reference implementation without any cache.

    Corresponds one-to-one to the original code in
    ``experiments/agreement_loop_qwen._per_stem_lufs`` (the dict insertion
    order is also kept as the order of ``state.stems``).
    """
    from ..strategies.knowledge_base_mix import classify_role
    out: Dict[str, Dict[str, float]] = {}
    for name in state.stems:
        lufs = _one_stem_lufs(state, name, sr)
        out[name] = {"role": classify_role(name), "lufs": lufs}
    return out


class StemLufsCache:
    """Memo for per-stem LUFS. Use **one instance per track**.

    When the track changes (i.e. the contents of ``state.stems`` become
    different objects) it resets automatically, so reusing an instance never
    produces a false hit. Still, since the snapshot holds strong references to
    the stem ndarrays, recreating it per track gives more predictable memory
    behaviour.

    Attributes:
        hits / misses: per-stem hit / miss counts (they grow by the number of
            stems per step).
        state_hits: number of whole-result shortcuts taken because the
            ``state`` object was identical.
        resets: number of times the cache was dropped because stems were
            swapped out (roughly the number of tracks).
    """

    def __init__(self, enabled: Optional[bool] = None,
                 max_entries: int = 200_000,
                 render_cache: bool = False,
                 render_cache_mb: Optional[float] = None):
        self.enabled = cache_enabled_by_env() if enabled is None else bool(enabled)
        self.max_entries = int(max_entries)
        # Whether to make the solo render of accepted steps incremental
        # (default False = as before, reapply the full chain every time).
        # Enabling it keeps the output bit-identical.
        self.render_cache = bool(render_cache)
        if render_cache_mb is None:
            try:
                render_cache_mb = float(os.environ.get(
                    RENDER_CACHE_MB_ENV, _DEFAULT_RENDER_CACHE_MB))
            except ValueError:
                render_cache_mb = _DEFAULT_RENDER_CACHE_MB
        self.render_cache_mb = float(render_cache_mb)
        #: stem name -> RenderCache dedicated to that stem. Sharing one cache
        #: would self-destruct through constant rebinding.
        self._render_caches: Dict[str, Any] = {}
        self._entries: Dict[Tuple[Any, ...], float] = {}
        # stems of the track currently bound (strong references). Prevents
        # false hits caused by id() reuse.
        self._stems_owner: Optional[Dict[str, Any]] = None
        self._stems_snapshot: Optional[Dict[str, Any]] = None
        # Whole-result shortcut on the most recent state (for back-to-back
        # calls).
        self._last_state: Any = None
        self._last_sr: Optional[int] = None
        self._last_lufs: Optional[Dict[str, float]] = None
        self.hits = 0
        self.misses = 0
        self.state_hits = 0
        self.resets = 0
        self.evictions = 0

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def per_stem_lufs(self, state, sr: int) -> Dict[str, Dict[str, float]]:
        """Return a result bit-identical to ``compute_per_stem_lufs(state, sr)``.

        The return value is **a fresh dict** every time (the caller may mutate
        it without dirtying the cache).
        """
        if not self.enabled:
            # Incremental rendering works independently of the memo, so even
            # with the memo off, render_cache=True still makes the solo render
            # side incremental (bit-identical).
            if not self.render_cache:
                return compute_per_stem_lufs(state, sr)
            from ..strategies.knowledge_base_mix import classify_role as _role
            self._bind(state)
            return {nm: {"role": _role(nm),
                         "lufs": _one_stem_lufs(state, nm, sr,
                                                self._render_cache_for(nm))}
                    for nm in state.stems}

        from ..strategies.knowledge_base_mix import classify_role

        # Whole-result shortcut: the same state object implies the same values.
        if (self._last_state is state and self._last_sr == sr
                and self._last_lufs is not None):
            self.state_hits += 1
            return {nm: {"role": classify_role(nm), "lufs": lu}
                    for nm, lu in self._last_lufs.items()}

        self._bind(state)
        lufs_by_name: Dict[str, float] = {}
        out: Dict[str, Dict[str, float]] = {}
        for name in state.stems:
            key = self._key(state, name, sr)
            val = self._entries.get(key, _MISS)
            if val is _MISS:
                val = _one_stem_lufs(state, name, sr,
                                     self._render_cache_for(name))
                if len(self._entries) >= self.max_entries:
                    self._entries.clear()
                    self.evictions += 1
                self._entries[key] = val
                self.misses += 1
            else:
                self.hits += 1
            lufs_by_name[name] = val
            out[name] = {"role": classify_role(name), "lufs": val}
        self._last_state = state
        self._last_sr = sr
        self._last_lufs = lufs_by_name
        return out

    def stats(self) -> Dict[str, Any]:
        """Aggregates to record in the run JSON (no effect whatsoever on the
        behaviour of the search)."""
        total = self.hits + self.misses
        out: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "stem_renders": int(self.misses),
            "stem_hits": int(self.hits),
            "stem_lookups": int(total),
            "hit_rate": (round(self.hits / total, 4) if total else None),
            "state_shortcuts": int(self.state_hits),
            "resets": int(self.resets),
            "evictions": int(self.evictions),
            "render_cache": bool(self.render_cache),
        }
        if self._render_caches:
            # Residual cost of the solo renders. If the incremental path is
            # working, stem_effect_applies is approximately the number of
            # acceptances (= stem_renders).
            out["stem_effect_applies"] = int(
                sum(c.n_apply for c in self._render_caches.values()))
            out["stem_render_cache_mib"] = round(
                sum(c.nbytes for c in self._render_caches.values()) / 2 ** 20, 1)
            out["stem_render_cache_rebinds"] = int(
                sum(c.n_rebind for c in self._render_caches.values()))
        return out

    def clear(self) -> None:
        self._entries.clear()
        self._stems_owner = None
        self._stems_snapshot = None
        self._last_state = None
        self._last_sr = None
        self._last_lufs = None
        self._render_caches.clear()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _render_cache_for(self, name: str):
        """RenderCache dedicated to ``name`` (None when disabled).

        The key point is a separate instance per stem. Reusing a single cache
        across stems makes the stem set of the solo state change every time,
        so ``RenderCache._bind`` clears on every call and the result is
        **slower than running with no cache at all**.
        """
        if not self.render_cache:
            return None
        c = self._render_caches.get(name)
        if c is None:
            from .renderer import RenderCache
            c = RenderCache(max_bytes=int(self.render_cache_mb * 1024 * 1024))
            self._render_caches[name] = c
        return c

    def _bind(self, state) -> None:
        """Bind to the current ``state.stems``. Drop the cache on a new track."""
        stems = state.stems
        snap = self._stems_snapshot
        if (self._stems_owner is stems and snap is not None
                and len(snap) == len(stems)
                and all(stems.get(k) is v for k, v in snap.items())):
            return
        self._entries.clear()
        self._last_state = None
        self._last_sr = None
        self._last_lufs = None
        # The track changed, so drop the solo RenderCaches too (so we stop
        # holding on to the stems of the previous track. RenderCache detects
        # rebinding by itself, but dropping here gives more predictable memory
        # behaviour).
        self._render_caches.clear()
        self._stems_owner = stems
        self._stems_snapshot = dict(stems)   # strong refs (guards against id() reuse accidents)
        self.resets += 1

    @staticmethod
    def _key(state, name: str, sr: int) -> Tuple[Any, ...]:
        """Key enumerating exactly the values that ``_render_single_stem(state,
        name)`` + ``_measure_lufs(_, sr)`` depend on -- no more, no less.

        ``master_chain`` is **deliberately excluded**, because
        ``_render_single_stem`` forces it down to ``()`` (so that master-scope
        actions do not invalidate the cache). Identity of stems is guaranteed
        on the ``_bind`` side, so it is not part of the key.
        """
        return (
            name,
            int(sr),
            int(state.sample_rate),
            state.static_processors.get(name, ()),
            state.transient_processors.get(name, ()),
            state.section_processors.get(name, ()),
            state.section_boundaries,
        )
