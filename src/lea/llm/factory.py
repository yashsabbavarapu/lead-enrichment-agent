"""Provider selection. Never crashes the CLI over a missing key or SDK."""

from __future__ import annotations

import logging
import os
from typing import Any

from lea.config import Settings
from lea.llm.stub import StubClient

logger = logging.getLogger(__name__)

__all__ = ["get_llm"]


def get_llm(settings: Settings) -> Any:
    """Return an :class:`~lea.llm.base.LLMClient` for *settings*.

    ``anthropic`` is used when it is both requested and actually usable. Any
    reason it is not — SDK missing, no key, client construction blowing up — is
    logged as one clear line and downgraded to the deterministic stub, because a
    keyless run is a supported mode, not an error.
    """
    provider = (settings.resolved_llm() or "stub").strip().lower()

    if provider == "stub":
        return StubClient(settings)

    if provider != "anthropic":
        logger.warning("Unknown LLM provider %r; falling back to the deterministic stub client.", provider)
        return StubClient(settings)

    key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning(
            "LLM provider 'anthropic' requested but no ANTHROPIC_API_KEY is set; "
            "falling back to the deterministic stub client."
        )
        return StubClient(settings)

    try:
        import anthropic  # noqa: F401  (import is the availability check)
    except ImportError as exc:
        logger.warning(
            "LLM provider 'anthropic' requested but the SDK is not importable (%s); "
            "falling back to the deterministic stub client. Fix with: pip install anthropic",
            exc,
        )
        return StubClient(settings)

    try:
        from lea.llm.anthropic_client import AnthropicClient

        return AnthropicClient(settings)
    except Exception as exc:  # noqa: BLE001 - provider setup must not kill the run
        logger.warning(
            "Could not construct the Anthropic client (%s: %s); "
            "falling back to the deterministic stub client.",
            type(exc).__name__,
            exc,
        )
        return StubClient(settings)
