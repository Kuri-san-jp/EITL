"""W&B logger with a graceful no-op when disabled or unavailable."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


class WandbLogger:
    def __init__(self,
                 project: str = "mix-orchestrator",
                 entity: Optional[str] = None,
                 run_name: Optional[str] = None,
                 enabled: bool = True,
                 mode: str = "online",
                 config: Optional[Dict[str, Any]] = None):
        self.enabled = enabled
        self._run = None
        if not enabled or mode == "disabled":
            return
        try:
            import wandb
            self._run = wandb.init(
                project=project, entity=entity, name=run_name,
                config=config or {}, mode=mode,
                dir=os.environ.get("MIXORCH_RUNS_DIR", "outputs/runs"),
            )
        except Exception as ex:                                       # noqa: BLE001
            print(f"[wandb] disabled: {ex}")
            self.enabled = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            wandb.log(data, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
