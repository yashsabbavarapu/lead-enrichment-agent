"""Provider selection. Never crashes the CLI over a missing key or SDK."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from lea.config import OPENAI_COMPAT_PRESETS, Settings
from lea.llm.stub import StubClient

logger = logging.getLogger(__name__)

__all__ = ["get_llm", "list_models"]


def get_llm(settings: Settings) -> Any:
    """Return an :class:`~lea.llm.base.LLMClient` for *settings*.

    A real provider is used when it is both requested and actually usable. Any
    reason it is not — SDK missing, no key, no base URL, client construction
    blowing up — is logged as one clear line and downgraded to the deterministic
    stub, because a keyless run is a supported mode, not an error.
    """
    provider = (settings.resolved_llm() or "stub").strip().lower()

    if provider == "stub":
        return StubClient(settings)

    if provider == "anthropic":
        return _anthropic_or_stub(settings)

    if settings.is_openai_compat(provider):
        return _openai_compat_or_stub(settings, provider)

    logger.warning("Unknown LLM provider %r; falling back to the deterministic stub client.", provider)
    return StubClient(settings)


# ---------------------------------------------------------------------------


def _anthropic_or_stub(settings: Settings) -> Any:
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


def _openai_compat_or_stub(settings: Settings, provider: str) -> Any:
    """Google / Groq / OpenRouter / Cerebras / Ollama, or any custom endpoint."""
    preset = OPENAI_COMPAT_PRESETS.get(provider, {})
    label = preset.get("label", provider)

    try:
        base_url, _model = settings.endpoint_for(provider)
    except Exception as exc:  # noqa: BLE001 - a bad config must not kill the run
        logger.warning(
            "Could not resolve an endpoint for LLM provider %r (%s: %s); "
            "falling back to the deterministic stub client.",
            provider,
            type(exc).__name__,
            exc,
        )
        return StubClient(settings)

    if not base_url or not str(base_url).lower().startswith(("http://", "https://")):
        logger.warning(
            "LLM provider %r has no usable base URL (%r); set LEA_LLM_BASE_URL to an "
            "http(s) endpoint. Falling back to the deterministic stub client.",
            provider,
            base_url,
        )
        return StubClient(settings)

    # Ollama runs locally and legitimately has no key; everyone else needs one.
    needs_key = bool(preset.get("env")) or provider not in OPENAI_COMPAT_PRESETS
    if needs_key and not settings.key_for(provider):
        env_name = preset.get("env") or "LEA_LLM_API_KEY"
        logger.warning(
            "LLM provider %r (%s) requested but no %s is set; "
            "falling back to the deterministic stub client.",
            provider,
            label,
            env_name,
        )
        return StubClient(settings)

    try:
        from lea.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(settings, provider=provider)
    except Exception as exc:  # noqa: BLE001 - provider setup must not kill the run
        logger.warning(
            "Could not construct the %s client (%s: %s); "
            "falling back to the deterministic stub client.",
            label,
            type(exc).__name__,
            exc,
        )
        return StubClient(settings)


def list_models(settings: Settings, provider: Optional[str] = None) -> list[str]:
    """Model ids the configured OpenAI-compatible endpoint actually serves, or [].

    Re-exported here so callers (e.g. `lea doctor`) have a single import site for
    provider plumbing. Never raises; returns [] for Anthropic, the stub, or any
    unreachable endpoint.
    """
    name = (provider or settings.resolved_llm() or "").strip().lower()
    if not settings.is_openai_compat(name):
        return []
    try:
        from lea.llm.openai_compat import list_models as _list_models

        return _list_models(settings, provider=name)
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        logger.debug("list_models(%s) failed: %s: %s", name, type(exc).__name__, exc)
        return []
