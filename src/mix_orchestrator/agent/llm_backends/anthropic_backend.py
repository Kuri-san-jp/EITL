"""Anthropic Claude backend (default model: claude-opus-4-7).

Uses the Anthropic Messages API with native tool-use. Prompt caching is
enabled on the system prompt + tool specs to keep iteration cost bounded.

Two auth modes (auto-selected):

1. **First-party API** (default): needs ``ANTHROPIC_API_KEY``.
2. **Vertex AI**: set ``ANTHROPIC_VERTEX_PROJECT_ID`` (and optionally
   ``CLOUD_ML_REGION``, default ``global``). Auth comes from Application
   Default Credentials, so billing goes through GCP exactly like the Gemini
   backend. Vertex model ids carry a version suffix
   (e.g. ``claude-haiku-4-5@20251001``); pass ``ANTHROPIC_VERTEX_MODEL`` or
   ``--llm-model`` in that form. Requires the Anthropic models to be enabled
   for the GCP project in Vertex Model Garden -- otherwise every call fails
   with ``404 Publisher model ... not found or your project does not have
   access`` (verified  for project rd-1-482007).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import LLMBackend, LLMResponse

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Rough public price estimate ($/MTok). Update as needed.
_PRICE_PER_MTOK = {
    "claude-opus-4-7":   {"in": 15.0, "out": 75.0},
    "claude-sonnet-4-6": {"in":  3.0, "out": 15.0},
    "claude-haiku-4-5":  {"in":  1.0, "out":  5.0},
}


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self,
                 model: str = "claude-opus-4-7",
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 api_key: Optional[str] = None,
                 vertex_project: Optional[str] = None,
                 vertex_region: Optional[str] = None):
        if not HAS_ANTHROPIC:
            raise RuntimeError("`pip install anthropic` first")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        project = vertex_project or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        if project:
            # Vertex AI mode: ADC auth (google-auth). Billing goes through GCP.
            region = (vertex_region or os.environ.get("CLOUD_ML_REGION")
                      or "global")
            self.via_vertex = True
            self.vertex_region = region
            self.client = anthropic.AsyncAnthropicVertex(
                project_id=project, region=region)
        else:
            self.via_vertex = False
            self.vertex_region = None
            self.client = anthropic.AsyncAnthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    async def complete(self,
                       system: str,
                       user: str,
                       tool_specs: Optional[List[Dict[str, Any]]] = None,
                       ) -> LLMResponse:
        # claude-haiku-4-5 and newer models (claude-sonnet-5 etc.) do not
        # accept the temperature parameter (API returns BadRequestError).
        # On Vertex the id looks like ``claude-haiku-4-5@20251001``, so match
        # on the prefix.
        _no_temp_prefixes = ("claude-haiku-4-5", "claude-sonnet-5")
        _no_temp = self.model.startswith(_no_temp_prefixes)
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # Cache the system prompt so multi-iter runs cost ~5x less
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if not _no_temp:
            kwargs["temperature"] = self.temperature
        if tool_specs:
            kwargs["tools"] = tool_specs

        try:
            resp = await self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as ex:
            # Fallback for an unknown new model that rejects temperature
            # (this changes the decoding conditions, so disclose it together
            # with the served meta).
            if "temperature" in str(ex) and "temperature" in kwargs:
                kwargs.pop("temperature")
                resp = await self.client.messages.create(**kwargs)
            else:
                raise

        # Extract text + tool calls
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in resp.content:
            t = getattr(block, "type", None)
            if t == "text":
                text_parts.append(block.text)
            elif t == "tool_use":
                tool_calls.append({"name": block.name, "arguments": dict(block.input)})
        raw_text = "\n".join(text_parts).strip()

        # Try to extract JSON from raw_text (system prompt asked for JSON)
        parsed = _extract_json(raw_text) or {}

        usage = resp.usage
        tokens_in = getattr(usage, "input_tokens", 0)
        tokens_out = getattr(usage, "output_tokens", 0)
        # cache tokens are billed separately; price approximation below
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read   = getattr(usage, "cache_read_input_tokens", 0)     or 0

        price = _PRICE_PER_MTOK.get(self.model, {"in": 5.0, "out": 20.0})
        cost = (
            (tokens_in    * price["in"]  +
             tokens_out   * price["out"] +
             cache_create * price["in"] * 1.25 +
             cache_read   * price["in"] * 0.1) / 1e6
        )

        return LLMResponse(
            parsed_json=parsed,
            tool_calls=tool_calls,
            raw_text=raw_text,
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            meta={"stop_reason": resp.stop_reason,
                  # The model the API actually served (to detect a repeat of
                  # the 2026-06 incident where a different model was served
                  # for haiku; cross-check against the requested self.model)
                  "model": getattr(resp, "model", None),
                  "cache_create": cache_create,
                  "cache_read": cache_read},
        )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON pick from free-form text."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))
    # Find first { ... } block
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
