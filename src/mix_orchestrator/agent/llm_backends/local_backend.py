"""Local OpenAI-compatible LLM backend.

Targets any server speaking the OpenAI Chat Completions API on
http://<host>:<port>/v1 — including LM Studio, Ollama (with
`OLLAMA_HOST` exposing /v1), vLLM, llama.cpp's `server`, and
text-generation-inference.

Anthropic-style tool specs (input_schema based) are converted to the
OpenAI function-tool format on the fly so the same `tools/schemas.py`
catalog works across both backends.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import LLMBackend, LLMResponse


class LocalLLMBackend(LLMBackend):
    name = "local"

    def __init__(self,
                 base_url: str = "http://127.0.0.1:1234/v1",
                 model: str = "default",
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 api_key: str = (os.environ.get("LOCAL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed"),
                 request_timeout: float = 120.0,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 seed_base: Optional[int] = None,
                 presence_penalty: Optional[float] = None,
                 frequency_penalty: Optional[float] = None,
                 repetition_penalty: Optional[float] = None):
        """OpenAI-compatible local server backend.

        The defaults are unchanged: **greedy (temperature=0.0, nothing else sent)**.
        *All* the extra sampling arguments default to ``None``, and a ``None``
        entry is **never put on the request** (the server-side default applies).
        So passing nothing reproduces earlier runs bit-for-bit.

        ``seed_base``:
            If not ``None``, every request carries
            ``seed = (seed_base ^ blake2b(system + "\\x00" + user)) mod 2**31``.
            The key property is that **the seed is a pure function of the prompt**:

            * The prompt starts with ``Step {step}``, which changes per step, so
              the seed changes per step too -> infinite repetition of the same
              proposal is broken.
            * On an invalid-action retry the prompt gets ``[system] ...`` appended,
              so the seed changes within the same step -> the retry yields a
              different candidate.
            * Conversely, "same history = same prompt" implies the same seed, so a
              run with budget k and a run with budget K(>k) match exactly for the
              first k moves (prefix integrity holds under
              ``--no-prompt-show-budget``).

            Python's built-in ``hash()`` is randomized per process for strings, so
            it cannot be used here. blake2b is used instead.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as ex:
            raise RuntimeError("`pip install openai` is required for the local backend") from ex
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        self.client = AsyncOpenAI(base_url=self.base_url,
                                  api_key=api_key,
                                  timeout=request_timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.seed_base = seed_base
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty

    # ---- telemetry: the sampling settings actually sent (required to reproduce a run) ----
    def sampling_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed_base": self.seed_base,
            "seed_mode": ("prompt_hash" if self.seed_base is not None else None),
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }

    def _seed_for(self, system: str, user: str) -> int:
        h = hashlib.blake2b((system + "\x00" + user).encode("utf-8"),
                            digest_size=8).digest()
        return (int(self.seed_base) ^ int.from_bytes(h, "big")) % (2 ** 31)

    async def complete(self,
                       system: str,
                       user: str,
                       tool_specs: Optional[List[Dict[str, Any]]] = None,
                       ) -> LLMResponse:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        # Leave out the None entries = keep the server defaults (bit-identical to earlier runs).
        if self.top_p is not None:
            kwargs["top_p"] = float(self.top_p)
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = float(self.presence_penalty)
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = float(self.frequency_penalty)
        if self.seed_base is not None:
            kwargs["seed"] = self._seed_for(system, user)
        # top_k / repetition_penalty are vLLM extensions absent from the OpenAI API,
        # so they go through extra_body.
        extra: Dict[str, Any] = {}
        if self.top_k is not None:
            extra["top_k"] = int(self.top_k)
        if self.repetition_penalty is not None:
            extra["repetition_penalty"] = float(self.repetition_penalty)
        if extra:
            kwargs["extra_body"] = extra

        if tool_specs:
            kwargs["tools"] = _anthropic_to_openai_tools(tool_specs)
            kwargs["tool_choice"] = "auto"

        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as ex:                                                # noqa: BLE001
            # Some local servers reject `tool_choice="auto"` — retry without
            if tool_specs and "tool_choice" in str(ex):
                kwargs.pop("tool_choice", None)
                resp = await self.client.chat.completions.create(**kwargs)
            else:
                raise

        msg = resp.choices[0].message
        raw_text = (msg.content or "").strip()

        # Native tool_use blocks (OpenAI style)
        tool_calls: List[Dict[str, Any]] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = tc.function
            tool_calls.append({
                "name": fn.name,
                "arguments": _try_json_loads(fn.arguments) or {},
            })

        # Fallback: manually extract the Qwen2.5-style
        # `<tool_call>{"name":..., "arguments":...}</tool_call>` XML-tag form from
        # raw_text (covers cases where vLLM's hermes parser fails on nested or
        # mixed text).
        if not tool_calls and raw_text and "<tool_call>" in raw_text:
            tool_calls = _extract_xml_tool_calls(raw_text)

        parsed = _extract_json(raw_text) or {}
        usage = getattr(resp, "usage", None)
        tokens_in  = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

        return LLMResponse(
            parsed_json=parsed,
            tool_calls=tool_calls,
            raw_text=raw_text,
            cost_usd=0.0,                              # local: no API cost
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            meta={
                "finish_reason": resp.choices[0].finish_reason,
                "model": resp.model,
                "base_url": self.base_url,
            },
        )


# ---------- helpers ----------


def _anthropic_to_openai_tools(anthropic_specs: List[Dict[str, Any]]
                                ) -> List[Dict[str, Any]]:
    """Convert tools/schemas.py output to OpenAI function-tool format."""
    out = []
    for s in anthropic_specs:
        out.append({
            "type": "function",
            "function": {
                "name":        s["name"],
                "description": s.get("description", ""),
                "parameters":  s.get("input_schema", {"type": "object"}),
            },
        })
    return out


def _try_json_loads(x: Any) -> Optional[Dict[str, Any]]:
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return None


_XML_TOOL_CALL_RE = None
def _extract_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract the Qwen2.5-style `<tool_call>{...}</tool_call>` form.

    When the LLM emits the hermes/qwen tool_call format, vLLM's auto-parser can
    get confused and leave native tool_calls empty. As a fallback, extract them
    from raw_text with a regex.
    """
    import re
    global _XML_TOOL_CALL_RE
    if _XML_TOOL_CALL_RE is None:
        _XML_TOOL_CALL_RE = re.compile(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    out: List[Dict[str, Any]] = []
    for m in _XML_TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        if not name:
            continue
        args = payload.get("arguments", {})
        if isinstance(args, str):
            args = _try_json_loads(args) or {}
        out.append({"name": name, "arguments": args})
    return out


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object pick from free-form text."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = "\n".join(l for l in t.splitlines() if not l.startswith("```"))
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(t[start:end])
    except json.JSONDecodeError:
        return None
