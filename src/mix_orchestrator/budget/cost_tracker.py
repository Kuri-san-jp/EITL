"""Hard budget cap on per-run and per-session API spend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostTracker:
    max_usd_per_run: float = 2.0
    max_usd_total:   float = 50.0
    spent_this_run:  float = 0.0
    spent_total:     float = 0.0
    by_source:       Dict[str, float] = field(default_factory=dict)

    def add(self, source: str, usd: float) -> None:
        self.spent_this_run += usd
        self.spent_total += usd
        self.by_source[source] = self.by_source.get(source, 0.0) + usd
        if self.spent_this_run > self.max_usd_per_run:
            raise BudgetExceeded(f"per-run cap ${self.max_usd_per_run} exceeded "
                                 f"(spent ${self.spent_this_run:.3f})")
        if self.spent_total > self.max_usd_total:
            raise BudgetExceeded(f"total cap ${self.max_usd_total} exceeded "
                                 f"(spent ${self.spent_total:.3f})")

    def reset_run(self) -> None:
        self.spent_this_run = 0.0
