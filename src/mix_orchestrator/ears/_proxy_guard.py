"""Guard that hard-blocks surrogate computations (proxies) from slipping in.

Strict policy: in experiments (smoke / ablation / sweep / orchestrator runs)
ears are forbidden from using the proxy fallback. If the real evaluation
model cannot be imported / loaded, fail with an error.

How it works:
  1. When the environment variable ``AUTOMIX_FORBID_PROXY=1`` is set,
     ``assert_proxy_allowed("ear_name")`` raises and stops its caller.
  2. Every ear calls this function at the top of its ``_proxy()`` method.
  3. Experiment runners (experiments/run_*.py and the smoke scripts) export
     this env at startup.
  4. pytest unit tests do not set the env, so checking behaviour through the
     proxy fallback on a dev machine still works.

Example call::

    from .._proxy_guard import assert_proxy_allowed

    def _proxy(self, ...):
        assert_proxy_allowed(self.name)
        ...
"""
from __future__ import annotations

import os


_ENV_FLAG = "AUTOMIX_FORBID_PROXY"


class ProxyForbiddenError(RuntimeError):
    """Raised when the proxy path is called while it is forbidden."""


def proxy_forbidden() -> bool:
    """Read the environment variable and report whether the proxy path is forbidden."""
    val = os.environ.get(_ENV_FLAG, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def assert_proxy_allowed(ear_name: str) -> None:
    """Raise if the proxy is forbidden.

    Args:
        ear_name: name of the calling ear (used in the error message)

    Raises:
        ProxyForbiddenError: when ``AUTOMIX_FORBID_PROXY=1``
    """
    if proxy_forbidden():
        raise ProxyForbiddenError(
            f"ear={ear_name!r} tried to fall back to a surrogate computation (proxy), "
            f"but that is forbidden because the environment variable {_ENV_FLAG}=1. "
            "Fix whatever prevents the real model from being imported / loaded "
            "(see docs/install_real_ears.md)."
        )


def forbid_proxy_for_experiment() -> None:
    """Call at the start of an experiment runner. Forbids the proxy process-wide."""
    os.environ[_ENV_FLAG] = "1"
