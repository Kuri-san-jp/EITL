"""Common interface for any LLM backend (Anthropic, OpenAI, mock, local)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Structured output from a backend turn.

    `tool_calls` is the LLM's planned actions (already structured).
    `raw_text` is the textual portion (diagnosis / reasoning).
    `parsed_json` is the JSON the system prompt asked for.
    `cost_usd` and `tokens` for budget tracking.
    """
    parsed_json: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]      # [{"name":..., "arguments":{...}}, ...]
    raw_text: str = ""
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    async def complete(self,
                       system: str,
                       user: str,
                       tool_specs: Optional[List[Dict[str, Any]]] = None,
                       ) -> LLMResponse:
        """One turn of the orchestrator loop."""
        raise NotImplementedError
