"""Immutable, content-addressable mix state.

A MixState is a snapshot of stems + every processor pending on them.
`with_action()` returns a NEW state — the previous state is preserved
for rollback. `state_id` is deterministic (content hash) so identical
mixes share cache entries.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .effects import (
    Effect, StaticEffect, SectionEffect, MasterEffect, TransientEffect,
)


@dataclass(frozen=True)
class MixState:
    """Immutable snapshot of a mix in progress.

    Mutation is done via `with_action()` which returns a new MixState
    referencing the parent's state_id.
    """
    # Audio data (stems, NOT processed). float32 in [-1, 1].
    # Shape: (channels, samples). Mono stems are (1, N), stereo (2, N).
    stems: Dict[str, np.ndarray]
    sample_rate: int

    # Section boundaries — sorted by time, label = "intro"/"verse_1"/...
    section_boundaries: Tuple[Tuple[float, str], ...] = ()

    # Processors keyed by track name -> list of effects (applied in order).
    static_processors: Dict[str, Tuple[StaticEffect, ...]] = field(default_factory=dict)
    section_processors: Dict[str, Tuple[SectionEffect, ...]] = field(default_factory=dict)
    transient_processors: Dict[str, Tuple[TransientEffect, ...]] = field(default_factory=dict)
    master_chain: Tuple[MasterEffect, ...] = ()

    # Lineage
    parent_state_id: Optional[str] = None
    state_id: str = ""
    created_at: float = field(default_factory=time.time)
    action_summary: Optional[str] = None

    def __post_init__(self):
        # Compute deterministic state_id if not provided
        if not self.state_id:
            object.__setattr__(self, "state_id", self._content_hash()[:12])

    # ---------- Construction ----------

    @classmethod
    def initial_from_stems(
        cls,
        stems: Dict[str, np.ndarray],
        sample_rate: int,
    ) -> "MixState":
        normed = {k: _ensure_2d(v).astype(np.float32) for k, v in stems.items()}
        return cls(stems=normed, sample_rate=sample_rate, action_summary="initial")

    def with_sections(self, boundaries: List[Tuple[float, str]]) -> "MixState":
        return self._evolve(section_boundaries=tuple(boundaries),
                            action_summary="set_sections")

    # ---------- Mutators (return new state) ----------

    def add_static(self, track: str, effect: StaticEffect, summary: str) -> "MixState":
        if track not in self.stems:
            raise KeyError(f"track {track!r} not in stems {list(self.stems)}")
        new_chain = self.static_processors.get(track, ()) + (effect,)
        new_static = {**self.static_processors, track: new_chain}
        return self._evolve(static_processors=new_static, action_summary=summary)

    def add_section(self, track: str, effect: SectionEffect, summary: str) -> "MixState":
        if track not in self.stems:
            raise KeyError(f"track {track!r} not in stems {list(self.stems)}")
        new_chain = self.section_processors.get(track, ()) + (effect,)
        new_section = {**self.section_processors, track: new_chain}
        return self._evolve(section_processors=new_section, action_summary=summary)

    def add_transient(self, track: str, effect: TransientEffect, summary: str) -> "MixState":
        if track not in self.stems:
            raise KeyError(f"track {track!r} not in stems {list(self.stems)}")
        new_chain = self.transient_processors.get(track, ()) + (effect,)
        new_trans = {**self.transient_processors, track: new_chain}
        return self._evolve(transient_processors=new_trans, action_summary=summary)

    def add_master(self, effect: MasterEffect, summary: str) -> "MixState":
        return self._evolve(master_chain=self.master_chain + (effect,),
                            action_summary=summary)

    def _evolve(self, **changes) -> "MixState":
        # Reset state_id & set parent so __post_init__ recomputes
        changes.setdefault("parent_state_id", self.state_id)
        changes["state_id"] = ""
        changes["created_at"] = time.time()
        return replace(self, **changes)

    # ---------- Identity / serialization ----------

    def _content_hash(self) -> str:
        h = hashlib.sha256()
        # Hash structure (not raw audio — stems are not modified, only referenced)
        h.update(str(sorted(self.stems.keys())).encode())
        h.update(str(self.sample_rate).encode())
        h.update(str(self.section_boundaries).encode())
        h.update(_describe(self.static_processors).encode())
        h.update(_describe(self.section_processors).encode())
        h.update(_describe(self.transient_processors).encode())
        h.update(str(self.master_chain).encode())
        h.update((self.parent_state_id or "").encode())
        return h.hexdigest()

    def describe(self) -> str:
        lines = [f"MixState({self.state_id}, parent={self.parent_state_id})",
                 f"  tracks: {list(self.stems)}",
                 f"  sections: {len(self.section_boundaries)}"]
        for trk, eff in self.static_processors.items():
            lines.append(f"  static[{trk}]: {len(eff)} effects")
        for trk, eff in self.section_processors.items():
            lines.append(f"  section[{trk}]: {len(eff)} effects")
        if self.master_chain:
            lines.append(f"  master: {len(self.master_chain)} effects")
        return "\n".join(lines)

    def diff(self, other: "MixState") -> str:
        return f"{self.action_summary} → {other.action_summary}"


# ---------- helpers ----------


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        # Accept (samples, channels) or (channels, samples); prefer (C, N).
        return x if x.shape[0] <= 2 else x.T
    raise ValueError(f"audio must be 1-D or 2-D, got shape {x.shape}")


def _describe(d: Dict[str, Tuple[Effect, ...]]) -> str:
    parts = []
    for k in sorted(d.keys()):
        parts.append(f"{k}=[" + ",".join(repr(e) for e in d[k]) + "]")
    return "{" + ";".join(parts) + "}"
