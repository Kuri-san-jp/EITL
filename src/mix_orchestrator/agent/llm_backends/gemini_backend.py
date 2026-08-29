"""Google Gemini backend via OpenAI-compatible endpoints.

Two auth modes (auto-selected):

1. API-key mode (GEMINI_API_KEY set):
     https://generativelanguage.googleapis.com/v1beta/openai/
2. Vertex AI mode (no GEMINI_API_KEY; ADC credentials):
     https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi
   Auth = OAuth2 Bearer token from Application Default Credentials
   (GOOGLE_APPLICATION_CREDENTIALS or gcloud ADC). Tokens expire after
   ~1 h, so complete() refreshes them when <10 min of validity remains.
   Model ids on this endpoint need a "google/" prefix (added here).

We inherit LocalLLMBackend.complete() unchanged (it handles OpenAI-style
tool_calls natively) but override __init__ to bypass the /v1 suffix logic
that LocalLLMBackend applies to local-server URLs.

Required env vars:
  API-key mode:  GEMINI_API_KEY
  Vertex mode:   GOOGLE_CLOUD_PROJECT (+ GOOGLE_APPLICATION_CREDENTIALS
                 when ADC is not at the default gcloud path)
Optional env vars (read by build_llm in _common.py):
  GEMINI_MODEL           default: gemini-2.5-flash
  GEMINI_MAX_TOKENS      default: 4096
  GOOGLE_CLOUD_LOCATION  default: global (Vertex mode)
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional

from .base import LLMResponse
from .local_backend import LocalLLMBackend

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_VERTEX_BASE_URL = ("https://aiplatform.googleapis.com/v1/projects/{project}"
                    "/locations/{location}/endpoints/openapi")
# Refresh the OAuth token when less than this many seconds of validity remain.
_TOKEN_REFRESH_MARGIN_SEC = 600.0


class GeminiBackend(LocalLLMBackend):
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        # Thinking-style models (gemini-2.5-pro) can take over 60s for a single
        # response, so allow overriding via env GEMINI_TIMEOUT (default 60s).
        if request_timeout is None:
            request_timeout = float(os.environ.get("GEMINI_TIMEOUT", "60"))
        try:
            from openai import AsyncOpenAI
        except ImportError as ex:
            raise RuntimeError(
                "`pip install openai` is required for the Gemini backend"
            ) from ex

        self._creds = None  # Vertex mode only (google.auth credentials)
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        if resolved_key:
            base_url = _GEMINI_BASE_URL
            resolved_model = model
        else:
            # Vertex AI mode: OAuth via Application Default Credentials.
            try:
                import google.auth
                from google.auth.transport.requests import Request as GARequest
            except ImportError as ex:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set and `google-auth` is not "
                    "importable. Either export GEMINI_API_KEY, or install "
                    "google-auth and provide ADC (GOOGLE_APPLICATION_CREDENTIALS)."
                ) from ex
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            if not project:
                raise ValueError(
                    "Vertex AI mode needs GOOGLE_CLOUD_PROJECT "
                    "(GEMINI_API_KEY is not set)."
                )
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            self._ga_request_cls = GARequest
            self._creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
            base_url = _VERTEX_BASE_URL.format(project=project,
                                               location=location)
            # Vertex OpenAI-compatible endpoint expects "google/<model>".
            resolved_model = (model if model.startswith("google/")
                              else f"google/{model}")
            resolved_key = self._refresh_token()

        # Set attributes directly to bypass LocalLLMBackend.__init__'s
        # /v1 suffix appending, which would corrupt the Gemini URL.
        self.base_url = base_url
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            timeout=request_timeout,
        )
        self.model = resolved_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ---- Vertex token handling -------------------------------------------
    def _refresh_token(self) -> str:
        self._creds.refresh(self._ga_request_cls())
        return self._creds.token

    def _ensure_fresh_token(self) -> None:
        creds = self._creds
        remaining = None
        if creds.expiry is not None:
            # google-auth expiries are naive UTC datetimes.
            remaining = (creds.expiry
                         - datetime.datetime.utcnow()).total_seconds()
        if (not creds.valid or remaining is None
                or remaining < _TOKEN_REFRESH_MARGIN_SEC):
            self._refresh_token()
            # openai client reads api_key per request, so mutation suffices.
            self.client.api_key = creds.token

    async def complete(self,
                       system: str,
                       user: str,
                       tool_specs: Optional[List[Dict[str, Any]]] = None,
                       ) -> LLMResponse:
        if self._creds is not None:
            self._ensure_fresh_token()
        return await super().complete(system, user, tool_specs)
