"""Experience library: keeps mixing moves that scored well in the past as
structured cases and retrieves the ones similar to the current context,
RAG-style (user proposal, .

Each case = {query(retrieval key: computable from the current state, i.e. it
never peeks at the answer), method(the move to apply), outcome(its effect)}.
Retrieval is a structured filter (genre/role) x cosine over the diagnosis
vector, so no heavy embedding model is needed and it stays interpretable.
numpy is the only dependency.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MixingCase:
    case_id: str
    query: Dict[str, Any]
    method: Dict[str, Any]
    outcome: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MixingCase":
        return MixingCase(case_id=d["case_id"], query=d.get("query", {}),
                          method=d.get("method", {}), outcome=d.get("outcome", {}))


def _role_jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _diag_cosine(qa: Dict[str, float], qb: Dict[str, float]) -> Optional[float]:
    keys = [k for k in qa if k in qb and isinstance(qa[k], (int, float))
            and isinstance(qb[k], (int, float))]
    if not keys:
        return None
    va = np.array([qa[k] for k in keys], dtype=float)
    vb = np.array([qb[k] for k in keys], dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class ExperienceLibrary:
    def __init__(self, cases: Optional[List[MixingCase]] = None):
        self.cases: List[MixingCase] = cases or []

    @staticmethod
    def load(path) -> "ExperienceLibrary":
        p = Path(path)
        cases = []
        if p.exists():
            for ln in p.read_text("utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    try:
                        cases.append(MixingCase.from_dict(json.loads(ln)))
                    except Exception:                          # noqa: BLE001
                        pass
        return ExperienceLibrary(cases)

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(c.to_dict(), ensure_ascii=False)
                               for c in self.cases) + "\n", encoding="utf-8")

    def add_case(self, case: MixingCase) -> None:
        self.cases.append(case)

    def _similarity(self, case: MixingCase, *, genre, stem_roles, diagnosis, section_type) -> float:
        q = case.query
        genre_match = 1.0 if (genre and q.get("genre") == genre) else (0.3 if genre else 0.5)
        role_sim = _role_jaccard(stem_roles, q.get("stem_roles", []))
        sect_match = 1.0 if (section_type is None or q.get("section_type") in (None, section_type)) else 0.5
        structured = 0.45 * genre_match + 0.45 * role_sim + 0.10 * sect_match
        diag = None
        if diagnosis is not None and isinstance(q.get("diagnosis"), dict):
            c = _diag_cosine(diagnosis, q["diagnosis"])
            if c is not None:
                diag = 0.5 * (c + 1.0)
        if diag is None:
            return structured
        return 0.5 * structured + 0.5 * diag

    def retrieve(self, *, genre=None, stem_roles=None, diagnosis=None,
                 section_type=None, k: int = 5, min_reward_delta: float = 0.0):
        stem_roles = stem_roles or []
        scored = []
        for c in self.cases:
            if float(c.outcome.get("reward_delta", 0.0)) < min_reward_delta:
                continue
            sim = self._similarity(c, genre=genre, stem_roles=stem_roles,
                                   diagnosis=diagnosis, section_type=section_type)
            conf = math.tanh(float(c.outcome.get("reward_delta", 0.0))) * 0.1 \
                + math.log1p(float(c.outcome.get("n_observations", 1))) * 0.02
            scored.append((sim + conf, sim, c))
        scored.sort(key=lambda x: -x[0])
        return [{"case": c.to_dict(), "similarity": round(sim, 3)}
                for _, sim, c in scored[:k]]

    def __len__(self) -> int:
        return len(self.cases)
