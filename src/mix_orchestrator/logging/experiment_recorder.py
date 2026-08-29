"""JSON-on-disk recorder used in tandem with W&B (and as the offline fallback)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ExperimentRecorder:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"

    def event(self, kind: str, payload: Dict[str, Any]) -> None:
        rec = {"kind": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def write_artifact(self, name: str, content: str) -> None:
        (self.root / name).write_text(content, encoding="utf-8")
