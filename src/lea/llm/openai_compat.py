"""One adapter for every OpenAI-compatible endpoint.

Google AI Studio, Groq, OpenRouter, Cerebras and local Ollama all speak the same
`/chat/completions` REST shape, so a single client unlocks all of them — and
with them, a real LLM run at zero cost. We talk to it with `httpx` (already a
dependency) rather than the `openai` package: five providers speaking plain REST
do not justify another SDK.

The hard part is not the request, it is that **structured-output support varies
wildly across free models**. So this client walks a capability ladder and
remembers which rung worked:

    TOOLS       forced function call            (best: schema-constrained)
    JSON_OBJECT response_format=json_object     (good: valid JSON, schema in prompt)
    PROSE       plain completion + extract_json (last resort: tolerant parsing)

Degradation is automatic. A user should never have to know that their model
cannot do function calling — they should just get a slightly less reliable
answer, not a crash. The resolved rung is cached per (base_url, model) for the
process, so the ~10 calls in one company's run pay the probe cost at most once.

Contract is identical to the Anthropic client: `decide` and `extract` never
raise. On total failure they return `AgentAction(tool=FINISH, ...)` and a
best-effort model instance respectively.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from lea.config import Settings
from lea.llm.anthropic_client import (
    MAX_CONTEXT_CHARS,
    best_effort_instance,
    truncate_context,
    _closest_tool,
    _render_tool_menu,
    _tool_spec_name,
)
from lea.llm.base import LLMUsage
from lea.llm.json_utils import coerce_to_model, extract_json, repair_prompt
from lea.models import AgentAction, ToolName

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "OpenAICompatClient",
    "list_models",
    "resolve_model",
    "rank_models",
    "Capability",
    "AuthStyle",
    "AUTH_STYLE_ENV",
    "auth_transport",
    "pinned_auth_style",
    "reset_capability_cache",
    "reset_provider_state",
    "classify_auth_failure",
]

DECIDE_TOOL_NAME = "next_action"

#: Bounded, jittered retry. Free tiers rate-limit hard; a batch run must bend
#: rather than break.
MAX_ATTEMPTS = 4
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 20.0
#: Never honour an absurd Retry-After — better to degrade than hang a run.
MAX_RETRY_AFTER_S = 30.0
#: After this many whole steps burned entirely on auth failures, stop retrying
#: them. A half-provisioned key deserves patience; a dead one must not turn a
#: 12-step run into minutes of backoff.
AUTH_BREAKER_THRESHOLD = 2


class Capability:
    """Rungs of the structured-output ladder, best first."""

    TOOLS = "tools"
    JSON_OBJECT = "json_object"
    PROSE = "prose"

    ORDER = (TOOLS, JSON_OBJECT, PROSE)


#: (base_url, model) -> the best rung known to work. Process-local by design:
#: it is a performance cache, not state worth persisting.
_CAPABILITY_CACHE: dict[tuple[str, str], str] = {}

#: base_url -> consecutive steps lost entirely to auth failures. Feeds the
#: circuit breaker below; cleared by any successful response.
_AUTH_FAILURES: dict[str, int] = {}

#: (base_url, configured_model) -> the model we actually ended up using. Free
#: model lineups are retired on a timescale of weeks, so a shipped default WILL
#: go stale; one lookup per run repairs it instead of failing every call.
_MODEL_RESOLUTION: dict[tuple[str, str], str] = {}

#: base_url -> the auth transport that endpoint actually accepts.
_AUTH_STYLE_CACHE: dict[str, str] = {}


class AuthStyle:
    """How to present the credential. Negotiated like the capability rung.

    Providers disagree, and change their minds: Google's OpenAI-compat endpoint
    rejects `x-goog-api-key` and `?key=` with a 400 *before it ever looks at the
    key*, while its native endpoint accepts both. Rather than encode a belief
    about which is right this month, try them in order and remember what worked.
    """

    BEARER = "bearer"   # Authorization: Bearer <key>   (OpenAI standard)
    HEADER = "header"   # x-goog-api-key: <key>         (Google native)
    QUERY = "query"     # ?key=<key>                    (Google legacy)

    ORDER = (BEARER, HEADER, QUERY)


#: Pin a transport and skip negotiation entirely.
AUTH_STYLE_ENV = "LEA_LLM_AUTH_STYLE"
_AUTH_STYLE_ALIASES = {
    "bearer": AuthStyle.BEARER, "authorization": AuthStyle.BEARER, "auth": AuthStyle.BEARER,
    "header": AuthStyle.HEADER, "x-goog-api-key": AuthStyle.HEADER, "goog": AuthStyle.HEADER,
    "google": AuthStyle.HEADER, "api-key": AuthStyle.HEADER,
    "query": AuthStyle.QUERY, "param": AuthStyle.QUERY, "key": AuthStyle.QUERY, "url": AuthStyle.QUERY,
}


def reset_provider_state() -> None:
    """Forget probed capabilities, auth counters, resolved models and transport."""
    _CAPABILITY_CACHE.clear()
    _AUTH_FAILURES.clear()
    _MODEL_RESOLUTION.clear()
    _AUTH_STYLE_CACHE.clear()


def pinned_auth_style() -> Optional[str]:
    """The operator's explicit choice via ``LEA_LLM_AUTH_STYLE``, if any."""
    raw = (os.environ.get(AUTH_STYLE_ENV) or "").strip().lower()
    if not raw:
        return None
    style = _AUTH_STYLE_ALIASES.get(raw)
    if style is None:
        logger.warning(
            "%s=%r is not one of %s; negotiating the auth transport instead.",
            AUTH_STYLE_ENV, raw, "/".join(AuthStyle.ORDER),
        )
    return style


def auth_transport(base_url: str, api_key: Optional[str], style: Optional[str] = None) -> tuple[dict[str, str], dict[str, str]]:
    """``(headers, query_params)`` carrying the credential exactly once.

    The key is never placed in two transports at the same time: a request that
    carries both a header and a query param makes a 400 impossible to attribute.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    params: dict[str, str] = {}
    if not api_key:
        # Ollama is local and keyless; an empty bearer makes some servers 401.
        return headers, params

    chosen = style or pinned_auth_style() or _AUTH_STYLE_CACHE.get(base_url, AuthStyle.BEARER)
    if chosen == AuthStyle.HEADER:
        headers["x-goog-api-key"] = api_key
    elif chosen == AuthStyle.QUERY:
        params["key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers, params


#: Backwards-compatible alias: the cache reset has always cleared everything
#: process-local, and callers already import this name.
reset_capability_cache = reset_provider_state


# ---------------------------------------------------------------------------
# error classification
# ---------------------------------------------------------------------------

#: Substrings that mean "this model/endpoint cannot do what you asked for",
#: as opposed to "your request was malformed". Providers disagree on wording.
_UNSUPPORTED_MARKERS = (
    "tool", "tools", "function", "function_call", "tool_choice", "tool use",
    "response_format", "json_object", "json mode", "json_schema",
    "not supported", "unsupported", "unrecognized", "unknown field",
    "does not support", "invalid_type", "unexpected keyword", "no body",
)

#: A 403 whose body talks about provisioning, propagation or an API that is not
#: switched on yet is a *timing* problem: the same request succeeds moments
#: later. Observed live on a freshly issued Google AI Studio key, which returned
#: 403 on 3 of 8 identical requests. Retrying is the difference between a full
#: record and an empty one.
_TRANSIENT_AUTH_MARKERS = (
    "caller does not have permission",
    "has not been used in project",
    "before or it is disabled",
    "is disabled",
    "not enabled",
    "service is not enabled",
    "enable it by visiting",
    "propagat",            # "changes may take a few minutes to propagate"
    "wait a few minutes",
    "try again",
    "temporarily",
    "retry",
)

#: A 401/403 that says the credential itself is wrong. Retrying cannot help, so
#: fail fast rather than burning four attempts on every call in the run.
_TERMINAL_AUTH_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "incorrect api key",
    "invalid authentication",
    "invalid_api_key",
    "unauthorized",
    "api key expired",
    "expired",
    "revoked",
    "permission denied for model",
    "does not have access to model",
    "billing",
    "account is not active",
    "suspended",
)


def _looks_unsupported(status: int, body: str) -> bool:
    """A 4xx that means "drop down a rung", not "give up"."""
    if status not in (400, 404, 405, 415, 422, 501):
        return False
    lowered = (body or "").lower()
    return any(marker in lowered for marker in _UNSUPPORTED_MARKERS)


def classify_auth_failure(status: int, body: str) -> str:
    """``"transient"`` or ``"terminal"`` for a 401/403 body.

    Terminal wording wins over transient wording, because "permission denied for
    model X" and "the caller does not have permission" read almost alike but mean
    opposite things. When nothing matches, a 403 is assumed transient and a 401
    terminal: a wasted retry costs a second, a wrongly-failed extraction costs
    the whole record.
    """
    lowered = (body or "").lower()
    if any(marker in lowered for marker in _TERMINAL_AUTH_MARKERS):
        return "terminal"
    if any(marker in lowered for marker in _TRANSIENT_AUTH_MARKERS):
        return "transient"
    return "terminal" if status == 401 else "transient"


# --- model retirement ------------------------------------------------------
#
# Two ids shipped in this project went dead inside one session
# (`gemini-2.0-flash`, then `gemini-2.5-flash`), so the adapter repairs itself
# rather than trusting any hardcoded default.

_MODEL_NOT_FOUND_MARKERS = (
    "is not found", "not found for api version", "no longer available",
    "model not found", "model_not_found", "model_decommissioned",
    "does not exist", "unknown model", "invalid model", "no endpoints found",
    "update your code to use", "please update to", "has been deprecated",
    "is deprecated", "decommissioned", "retired",
)

#: Words that mean "this is about a capability, not a dead model". A body
#: saying "this model does not support tools" must demote the ladder, not go
#: shopping for a different model.
_CAPABILITY_WORDS = ("tool", "function", "response_format", "json_object", "json_schema", "json mode")

#: Strong model-death phrases that outrank the capability words above.
_DEFINITELY_DEAD = ("not found", "no longer available", "does not exist", "decommission", "deprecat", "retired")

#: Google's 404 helpfully names the successor:
#: "... is not found ... please update to models/gemini-3.6-flash".
_REPLACEMENT_RES = (
    re.compile(r"(?:please\s+)?(?:update|switch|migrate|move)\s+(?:your\s+code\s+)?to\s+(?:use\s+)?['\"`]?(?:models/)?([A-Za-z][\w.:-]{2,120})"),
    re.compile(r"use\s+['\"`]?(?:models/)?([A-Za-z][\w.:-]{2,120})['\"`]?\s+instead"),
    re.compile(r"(?:replaced|superseded)\s+by\s+['\"`]?(?:models/)?([A-Za-z][\w.:-]{2,120})"),
    re.compile(r"(?:try|recommend(?:ed)?)\s+['\"`]?(?:models/)?([A-Za-z][\w.:-]{2,120})['\"`]?\s+instead"),
)

#: Non-text modalities and specialist variants. These rank high on a naive
#: "contains flash" match but cannot run a research agent: `gemini-*-flash-image`
#: generates pictures, `-tts` speaks, `-live`/`-native-audio` are realtime audio
#: sessions, `-embedding` returns vectors. Excluding them is the whole reason
#: this ranking is a named function and not a `sorted(..., key=len)`.
_MODEL_EXCLUDE_SEGMENTS = frozenset(
    {
        "image", "images", "imagen", "vision", "draw",
        "tts", "audio", "speech", "transcribe", "voice", "dialog",
        "live", "realtime", "bidi",
        "video", "veo",
        "embedding", "embeddings", "embed", "gecko",
        "robotics", "computer", "cua",
        "guard", "safety", "moderation", "rerank",
    }
)

_VERSION_RE = re.compile(r"(?:^|[-_/])(\d+(?:\.\d+)?)(?=[-_/]|$)")


def _segments(model_id: str) -> list[str]:
    return [s for s in re.split(r"[-_./:]+", (model_id or "").lower()) if s]


def _model_version(model_id: str) -> float:
    """The first version-looking number in an id: `gemini-3.5-flash` -> 3.5."""
    m = _VERSION_RE.search(model_id or "")
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:  # pragma: no cover - regex guarantees a number
        return 0.0


def _is_excluded_model(model_id: str) -> bool:
    return bool(_MODEL_EXCLUDE_SEGMENTS.intersection(_segments(model_id)))


def _google_flash_tier(model_id: str) -> Optional[int]:
    """Preference among Gemini ids: plain Flash, then Lite, then preview.

    Flash is the free tier, which is the whole point. ``None`` means "not a
    usable Gemini text-flash model".
    """
    segs = _segments(model_id)
    if "gemini" not in segs or "flash" not in segs or _is_excluded_model(model_id):
        return None
    lite = "lite" in segs
    preview = "preview" in segs or "exp" in segs or "experimental" in segs
    if lite and preview:
        return 3
    if lite:
        return 1
    if preview:
        return 2
    return 0


def _generic_sort_key(model_id: str, configured: str) -> tuple:
    """Provider-agnostic fallback: closest to what the user asked for, newest first."""
    segs = set(_segments(model_id))
    wanted = set(_segments(configured)) if configured else set()
    preview = 1 if segs & {"preview", "beta", "exp", "experimental", "alpha", "nightly"} else 0
    return (-len(segs & wanted), preview, -_model_version(model_id), len(model_id), model_id)


def rank_models(candidates: list[str], provider: str = "", configured: str = "") -> list[str]:
    """Order model ids best-first for *provider*, dropping unusable ones.

    Google gets the Flash-first ranking above (free tier, text-capable); every
    other provider gets `_generic_sort_key`, because Groq, OpenRouter and
    Cerebras retire ids too and deserve the same self-healing path.
    """
    usable = [c for c in dict.fromkeys(candidates or []) if c and not _is_excluded_model(c)]
    if not usable:
        return []

    head: list[str] = []
    if configured and configured in usable:
        head.append(configured)  # what the user asked for always wins
    rest = [c for c in usable if c != configured]

    is_gemini = provider.strip().lower() == "google" or any("gemini" in _segments(c) for c in rest)
    if is_gemini:
        flash = [c for c in rest if _google_flash_tier(c) is not None]
        if flash:
            flash.sort(key=lambda m: (_google_flash_tier(m), -_model_version(m), m))
            others = [c for c in rest if c not in set(flash)]
            others.sort(key=lambda m: _generic_sort_key(m, configured))
            return head + flash + others

    rest.sort(key=lambda m: _generic_sort_key(m, configured))
    return head + rest


def _looks_model_not_found(status: int, body: str, current_model: str = "") -> bool:
    """A 4xx that means "this model id is dead", not "this request was wrong"."""
    if status not in (400, 404):
        return False
    lowered = (body or "").lower()
    if not any(marker in lowered for marker in _MODEL_NOT_FOUND_MARKERS):
        return False
    # "this model does not support tools" is a capability gap. Only let the
    # capability words win when nothing says the model itself is gone.
    if any(word in lowered for word in _CAPABILITY_WORDS):
        if not any(dead in lowered for dead in _DEFINITELY_DEAD):
            return False
    return "model" in lowered or bool(current_model and current_model.lower() in lowered)


def _replacement_from_error(body: str, exclude: str = "") -> Optional[str]:
    """The successor id the provider named in its own error, if any.

    Highest-signal source there is, and free to parse: Google's 404 says
    "please update to models/gemini-3.6-flash".
    """
    text = _error_message(body, limit=1000) or (body or "")
    for pattern in _REPLACEMENT_RES:
        m = pattern.search(text)
        if not m:
            continue
        candidate = m.group(1).strip().strip("'\"`.,;:)")
        if not candidate or candidate.lower() == (exclude or "").lower():
            continue
        if _is_excluded_model(candidate):
            logger.debug("Ignoring suggested replacement %r: not a text model", candidate)
            continue
        return candidate
    return None


#: Errors that mean "you presented the credential the wrong way", as distinct
#: from "your credential is bad". Google's compat endpoint 400s with
#: "Missing or invalid Authorization header" for x-goog-api-key / ?key=, and
#: 401s with "Expected OAuth 2 access token" when it wants a different
#: credential *type* — both are transport complaints, not key complaints.
_WRONG_TRANSPORT_MARKERS = (
    "missing or invalid authorization header",
    "authorization header",
    "expected oauth 2 access token",
    "expected oauth2 access token",
    "expects a bearer token",
    "missing authentication",
    "no credentials",
    "credential type",
    "authentication method",
    "api key not provided",
    "missing api key",
    "must be authenticated",
)


def _looks_wrong_auth_transport(status: int, body: str) -> bool:
    """True when switching transports could plausibly help.

    Deliberately yields to `_TERMINAL_AUTH_MARKERS`: cycling three transports
    against a revoked key just triples the cost of an inevitable failure.
    """
    if status not in (400, 401, 403):
        return False
    lowered = (body or "").lower()
    if not any(marker in lowered for marker in _WRONG_TRANSPORT_MARKERS):
        return False
    # A dead key is a dead key however you present it.
    return not any(marker in lowered for marker in _TERMINAL_AUTH_MARKERS)


class _Unsupported(Exception):
    """The endpoint rejected this rung of the ladder."""


class _ModelUnavailable(Exception):
    """The configured model is gone and nothing usable could replace it."""


class _AuthFailure(Exception):
    """Authentication failed. Carries whether it looked transient or terminal."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _error_message(body: str, limit: int = 200) -> str:
    """The provider's `error.message`, collapsed to one short line.

    Pasting a raw JSON payload into the agent trace is what made a 403 read like
    a stack dump in the middle of the rendered run.
    """
    if not body:
        return ""
    text: Any = body
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        data = None
    # Google's OpenAI-compat layer wraps its error in a one-element ARRAY —
    # `[{"error": {...}}]` — so an object-only parse leaks raw JSON to the trace.
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            text = err.get("message") or err.get("status") or err.get("type") or body
        elif isinstance(err, str):
            text = err
        elif "message" in data:
            text = data["message"]
    flat = " ".join(str(text).split())
    return flat[: limit - 1] + "…" if len(flat) > limit else flat


def _note_auth_failure(base_url: str) -> int:
    _AUTH_FAILURES[base_url] = _AUTH_FAILURES.get(base_url, 0) + 1
    return _AUTH_FAILURES[base_url]


def _auth_breaker_open(base_url: str) -> bool:
    return _AUTH_FAILURES.get(base_url, 0) >= AUTH_BREAKER_THRESHOLD


#: Google sends no `Retry-After` header on a 429 — the delay is in the JSON
#: body, as `RetryInfo.retryDelay` and in prose ("Please retry in 30.98s").
_RETRY_INFO_RE = re.compile(r'"retryDelay"\s*:\s*"?(\d+(?:\.\d+)?)s?"?', re.IGNORECASE)
_RETRY_PROSE_RE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

#: A *per-day* quota cannot recover inside this run. Retrying it four times on
#: each of ~12 steps turns a spent quota into minutes of pointless backoff.
_DAILY_QUOTA_MARKERS = ("perday", "requestsperday", "dailylimit", "quotaperday")


class _QuotaExhausted(Exception):
    """A quota that cannot recover within this run."""


def _retry_delay_from_body(body: str) -> Optional[float]:
    """Seconds the provider asked us to wait, read out of the response body."""
    for pattern in (_RETRY_INFO_RE, _RETRY_PROSE_RE):
        m = pattern.search(body or "")
        if m:
            try:
                return max(0.0, min(float(m.group(1)), MAX_RETRY_AFTER_S))
            except ValueError:  # pragma: no cover - regex guarantees a number
                continue
    return None


def _is_daily_quota(body: str) -> bool:
    """True when a 429 names a per-day quota rather than a per-minute one."""
    squashed = re.sub(r"[^a-z]", "", (body or "").lower())
    return any(marker in squashed for marker in _DAILY_QUOTA_MARKERS)


def _retry_after_seconds(response: httpx.Response, body: str = "") -> Optional[float]:
    raw = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-after")
    if raw:
        try:
            return max(0.0, min(float(raw), MAX_RETRY_AFTER_S))
        except (TypeError, ValueError):
            logger.debug("Unparseable Retry-After header: %r", raw)
    return _retry_delay_from_body(body)


# ---------------------------------------------------------------------------


def _schema_for_decide(tool_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "One or two sentences: what is still unknown and why this step closes it.",
            },
            "tool": {"type": "string", "enum": tool_names, "description": "The tool to run next."},
            "args": {
                "type": "object",
                "description": "Arguments for that tool, matching its documented args schema.",
                "additionalProperties": True,
            },
        },
        "required": ["thought", "tool", "args"],
    }


def _describe_schema(schema: dict[str, Any]) -> str:
    """Render a JSON schema into the prompt, for rungs that cannot enforce it."""
    try:
        return json.dumps(schema, indent=2, default=str)[:6000]
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        logger.debug("Could not serialise schema for the prompt: %s", exc)
        return "{}"


def _model_json_schema(schema_model: type[BaseModel]) -> dict[str, Any]:
    try:
        schema = dict(schema_model.model_json_schema())
    except Exception as exc:  # noqa: BLE001 - pydantic raises several types here
        logger.error("Could not build a JSON schema for %s: %s", schema_model.__name__, exc)
        return {"type": "object", "properties": {}, "additionalProperties": True}
    schema.setdefault("type", "object")
    schema.pop("title", None)
    return schema


class OpenAICompatClient:
    """LLMClient for any OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        settings: Settings,
        provider: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.settings = settings
        self.name = (provider or settings.resolved_llm() or "openai_compat").strip().lower()
        base_url, model = settings.endpoint_for(self.name)
        if not base_url:
            raise ValueError(f"No base_url configured for LLM provider {self.name!r}")
        self.base_url = base_url.rstrip("/")
        self._configured_model = model
        # A model this process already found dead stays repaired for the rest
        # of the run, so one retirement costs one lookup, not one per call.
        self.model = _MODEL_RESOLUTION.get((self.base_url, model), model)
        self.api_key = settings.key_for(self.name)
        self._max_context = MAX_CONTEXT_CHARS
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(max(30.0, settings.http_timeout_s * 2)),
            follow_redirects=True,
        )

    # -- plumbing --------------------------------------------------------

    @property
    def _cache_key(self) -> tuple[str, str]:
        return (self.base_url, self.model)

    @property
    def capability(self) -> str:
        """The best rung known to work for this endpoint (assume the best until told otherwise)."""
        return _CAPABILITY_CACHE.get(self._cache_key, Capability.TOOLS)

    def _demote(self, failed: str, reason: str) -> Optional[str]:
        """Record that *failed* does not work here; return the next rung down."""
        try:
            nxt = Capability.ORDER[Capability.ORDER.index(failed) + 1]
        except (ValueError, IndexError):
            return None
        _CAPABILITY_CACHE[self._cache_key] = nxt
        # Happens at most twice per endpoint per process, and materially changes
        # output quality, so it earns a WARNING — but a one-line one.
        logger.warning(
            "%s/%s does not support %s (%s); falling back to %s for the rest of this run.",
            self.name, self.model, failed, _error_message(reason, 120) or reason, nxt,
        )
        return nxt

    @property
    def auth_style(self) -> str:
        """The transport in use: pinned, previously negotiated, or the default."""
        return pinned_auth_style() or _AUTH_STYLE_CACHE.get(self.base_url, AuthStyle.BEARER)

    def _headers(self) -> dict[str, str]:
        return auth_transport(self.base_url, self.api_key, self.auth_style)[0]

    def _demote_auth_style(self, reason: str) -> Optional[str]:
        """Move to the next transport. ``None`` when they are all exhausted."""
        if pinned_auth_style():
            return None  # the operator chose; do not second-guess them
        current = self.auth_style
        try:
            nxt = AuthStyle.ORDER[AuthStyle.ORDER.index(current) + 1]
        except (ValueError, IndexError):
            return None
        _AUTH_STYLE_CACHE[self.base_url] = nxt
        logger.debug("%s: auth transport %s rejected (%s); trying %s", self.name, current, reason, nxt)
        return nxt

    def _redact(self, text: str) -> str:
        """Never let the credential reach a log line or the agent trace."""
        if self.api_key and self.api_key in text:
            text = text.replace(self.api_key, "<redacted>")
        return _redact_query_key(text)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /chat/completions with bounded jittered retry.

        Raises `_Unsupported` when the endpoint rejects a capability,
        `_AuthFailure` for 401/403, and the last transport/HTTP error when
        retries are exhausted.

        Per-attempt noise is logged at DEBUG on purpose: the CLI installs no
        handler, so anything at WARNING lands on stderr in the middle of the
        rendered trace. Only the final give-up is worth a WARNING.
        """
        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[Exception] = None
        # Probing an auth transport is negotiation, not a retry, so it does not
        # spend the retry budget — but it is bounded by the number of styles.
        transport_probes = 0
        attempt = 0

        while attempt < MAX_ATTEMPTS:
            attempt += 1
            headers, params = auth_transport(self.base_url, self.api_key, self.auth_style)
            try:
                response = self._client.post(url, headers=headers, params=params, json=payload)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.debug("%s: transport error on attempt %d/%d: %s", self.name, attempt, MAX_ATTEMPTS, self._redact(str(exc)))
                if attempt == MAX_ATTEMPTS:
                    break
                self._sleep(attempt, None)
                continue

            status = response.status_code
            if status < 300:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    last_exc = exc
                    logger.debug("%s: non-JSON body on a %d response: %s", self.name, status, self._redact(str(exc)))
                    if attempt == MAX_ATTEMPTS:
                        break
                    self._sleep(attempt, None)
                    continue
                _AUTH_FAILURES.pop(self.base_url, None)  # the credential works
                if _AUTH_STYLE_CACHE.get(self.base_url) != self.auth_style:
                    _AUTH_STYLE_CACHE[self.base_url] = self.auth_style
                    if self.auth_style != AuthStyle.BEARER:
                        logger.info("%s: authenticating with the %s transport", self.name, self.auth_style)
                return data

            body = _safe_body(response)
            message = self._redact(_error_message(body))
            body = self._redact(body)

            # Before anything else: did we present the credential the wrong way?
            # A 400 "Missing or invalid Authorization header" says nothing about
            # whether the key is good, so switching transports is free of charge.
            if _looks_wrong_auth_transport(status, body) and transport_probes < len(AuthStyle.ORDER):
                if self._demote_auth_style(message) is not None:
                    transport_probes += 1
                    attempt -= 1  # negotiation, not a retry
                    continue

            # Checked BEFORE the capability probe: Google's dead-model 404 says
            # "... or is not supported for generateContent", which would
            # otherwise be misread as "this model cannot do tools" and demote
            # the ladder forever instead of fixing the model id.
            if _looks_model_not_found(status, body, self.model):
                replacement = self._resolve_dead_model(body)
                if replacement and replacement != payload.get("model"):
                    payload = {**payload, "model": replacement}
                    continue
                raise _ModelUnavailable(
                    f"model {self.model!r} is not available at {self.name} and no "
                    f"replacement could be found — run `lea doctor --models` to see "
                    f"what your key serves, then set LEA_MODEL: {message}"
                )

            if _looks_unsupported(status, body):
                raise _Unsupported(f"HTTP {status}: {message}")

            if status in (401, 403):
                raise_now = self._auth_outcome(status, body, message, attempt)
                if raise_now is not None:
                    raise raise_now
                logger.debug(
                    "%s: HTTP %d looks transient on attempt %d/%d: %s",
                    self.name, status, attempt, MAX_ATTEMPTS, message,
                )
                self._sleep(attempt, _retry_after_seconds(response, body))
                continue

            if status == 429 and _is_daily_quota(body):
                # A per-day quota will not come back before this run ends.
                raise _QuotaExhausted(
                    f"{self.name} daily free-tier quota is exhausted for {self.model} — "
                    f"retrying will not help today. Set LEA_MODEL to another model "
                    f"(`lea doctor --models`) or wait for the quota to reset: {message}"
                )

            if status == 429 or status >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {status}: {message}", request=response.request, response=response
                )
                if attempt == MAX_ATTEMPTS:
                    break
                wait = _retry_after_seconds(response, body)
                logger.debug(
                    "%s: HTTP %d on attempt %d/%d%s",
                    self.name, status, attempt, MAX_ATTEMPTS,
                    f"; Retry-After {wait}s" if wait is not None else "",
                )
                self._sleep(attempt, wait)
                continue

            # A 4xx we cannot interpret as a capability gap: not retryable.
            raise httpx.HTTPStatusError(f"HTTP {status}: {message}", request=response.request, response=response)

        # The caller logs one WARNING for the failed step; logging here too would
        # double every failure in the rendered trace.
        logger.debug("%s: request failed after %d attempts: %s", self.name, MAX_ATTEMPTS, self._redact(str(last_exc)))
        raise last_exc or RuntimeError(f"{self.name}: request failed after {MAX_ATTEMPTS} attempts")

    def _resolve_dead_model(self, body: str) -> Optional[str]:
        """Find a live model to replace the configured one. Cached per run.

        Order: the successor the provider named in its own error, else the
        best-ranked id from `GET /models`. Logs exactly one WARNING, because
        the user has to know they are not running what they configured.
        """
        key = (self.base_url, self._configured_model)
        cached = _MODEL_RESOLUTION.get(key)
        if cached:
            # Already resolved this run. If we are *already* using it, the
            # replacement is dead too and there is nothing further to try.
            return None if cached == self.model else cached

        suggested = _replacement_from_error(body, exclude=self.model)
        source = "named by the provider"
        if not suggested:
            available = list_models(self.settings, provider=self.name)
            ranked = [m for m in rank_models(available, provider=self.name, configured=self._configured_model)
                      if m != self.model]
            suggested = ranked[0] if ranked else None
            source = f"best of {len(available)} models offered by the endpoint"

        if not suggested:
            logger.debug("%s: no usable replacement for %r", self.name, self.model)
            return None

        _MODEL_RESOLUTION[key] = suggested
        logger.warning(
            "%s: model %r is unavailable; using %r instead (%s). "
            "Set LEA_MODEL to pin a different one.",
            self.name, self.model, suggested, source,
        )
        self.model = suggested
        return suggested

    def _auth_outcome(
        self, status: int, body: str, message: str, attempt: int
    ) -> Optional[_AuthFailure]:
        """Decide what to do with a 401/403. ``None`` means "retry"."""
        kind = classify_auth_failure(status, body)

        if kind == "terminal":
            _note_auth_failure(self.base_url)
            return _AuthFailure(
                f"authentication rejected by {self.name} — the API key looks wrong or "
                f"lacks access to {self.model}: {message}",
                kind,
            )

        # A dead key that merely *sounds* transient must not cost the whole run.
        if _auth_breaker_open(self.base_url):
            return _AuthFailure(
                f"authentication keeps failing at {self.name}; not retrying further "
                f"this run: {message}",
                "terminal",
            )

        if attempt == MAX_ATTEMPTS:
            # The caller logs one WARNING for the failed step; a second one here
            # would just double every auth failure in the rendered trace.
            count = _note_auth_failure(self.base_url)
            logger.debug("%s: auth failed on all %d attempts (%d step(s) lost)", self.name, MAX_ATTEMPTS, count)
            return _AuthFailure(
                f"authentication failed at {self.name} after {MAX_ATTEMPTS} attempts — "
                f"looks transient, the key may still be provisioning: {message}",
                "transient",
            )
        return None

    @staticmethod
    def _sleep(attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(BASE_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
        delay += random.uniform(0, 0.25 * max(delay, 0.1))  # jitter: avoid lockstep retries
        time.sleep(delay)

    @staticmethod
    def _usage(data: dict[str, Any], model: str) -> LLMUsage:
        usage = data.get("usage") or {}
        return LLMUsage(
            tokens_in=_as_int(usage.get("prompt_tokens")),
            tokens_out=_as_int(usage.get("completion_tokens")),
            model=str(data.get("model") or model),
        )

    @staticmethod
    def _message(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return {}
        message = choices[0].get("message")
        return message if isinstance(message, dict) else {}

    @classmethod
    def _tool_call_args(cls, data: dict[str, Any], name: str) -> Optional[dict[str, Any]]:
        """`function.arguments` is a JSON *string*: parse it, never string-match."""
        message = cls._message(data)
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            return None
        chosen = None
        for call in calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            if fn.get("name") == name:
                chosen = fn
                break
            if chosen is None:
                chosen = fn
        if not isinstance(chosen, dict):
            return None
        raw = chosen.get("arguments")
        if isinstance(raw, dict):
            return raw
        parsed = extract_json(raw) if isinstance(raw, str) else None
        if parsed is None and raw:
            logger.debug("%s: could not parse tool_call arguments: %r", name, str(raw)[:200])
        return parsed

    @classmethod
    def _text(cls, data: dict[str, Any]) -> str:
        message = cls._message(data)
        content = message.get("content")
        if isinstance(content, str):
            return content
        # Some gateways return the multimodal content-part array.
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    def _payload(
        self,
        *,
        rung: str,
        system: str,
        user: str,
        function_name: str,
        function_schema: dict[str, Any],
        function_description: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the request for one rung of the ladder."""
        prompt = user
        if rung != Capability.TOOLS:
            # No schema enforcement available — put the schema in the prompt.
            prompt = (
                f"{user}\n\nReturn ONLY a single JSON object matching this schema "
                f"(no prose, no markdown fence):\n{_describe_schema(function_schema)}"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if rung == Capability.TOOLS:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": function_description,
                        "parameters": function_schema,
                    },
                }
            ]
            payload["tool_choice"] = {"type": "function", "function": {"name": function_name}}
        elif rung == Capability.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _call_ladder(
        self,
        *,
        system: str,
        user: str,
        function_name: str,
        function_schema: dict[str, Any],
        function_description: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[Optional[dict[str, Any]], LLMUsage, str]:
        """Walk down the ladder until a rung works.

        Returns ``(payload_dict_or_None, usage, raw_text)``. Raises only for
        genuinely fatal transport/HTTP problems; callers degrade from there.
        """
        rung = self.capability
        total = LLMUsage(model=self.model)
        raw_text = ""

        while rung is not None:
            payload = self._payload(
                rung=rung,
                system=system,
                user=user,
                function_name=function_name,
                function_schema=function_schema,
                function_description=function_description,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                data = self._post(payload)
            except _Unsupported as exc:
                nxt = self._demote(rung, str(exc))
                if nxt is None:
                    raise
                rung = nxt
                continue

            usage = self._usage(data, self.model)
            total = LLMUsage(
                tokens_in=total.tokens_in + usage.tokens_in,
                tokens_out=total.tokens_out + usage.tokens_out,
                model=usage.model or self.model,
            )

            if rung == Capability.TOOLS:
                parsed = self._tool_call_args(data, function_name)
                if parsed is not None:
                    _CAPABILITY_CACHE.setdefault(self._cache_key, Capability.TOOLS)
                    return parsed, total, self._text(data)
                # A 200 that ignored tool_choice: the model "supports" tools but
                # will not use them. Same practical outcome — drop a rung.
                raw_text = self._text(data)
                salvaged = extract_json(raw_text)
                if salvaged is not None:
                    return salvaged, total, raw_text
                nxt = self._demote(rung, "responded without a tool_call")
                if nxt is None:
                    return None, total, raw_text
                rung = nxt
                continue

            raw_text = self._text(data)
            parsed = extract_json(raw_text)
            if parsed is not None:
                _CAPABILITY_CACHE.setdefault(self._cache_key, rung)
                return parsed, total, raw_text
            nxt = self._demote(rung, "returned no parseable JSON object")
            if nxt is None:
                return None, total, raw_text
            rung = nxt

        return None, total, raw_text

    def _repair(
        self,
        *,
        system: str,
        user: str,
        bad_payload: Any,
        errors: Any,
        function_name: str,
        function_schema: dict[str, Any],
        function_description: str,
        max_tokens: int,
    ) -> tuple[Optional[dict[str, Any]], LLMUsage]:
        """One corrective round-trip after a validation failure."""
        repair_user = f"{user}\n\n{repair_prompt(bad_payload, errors)}"
        try:
            parsed, usage, _ = self._call_ladder(
                system=system,
                user=repair_user,
                function_name=function_name,
                function_schema=function_schema,
                function_description=function_description,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return parsed, usage
        except Exception as exc:  # noqa: BLE001 - repair is best-effort by definition
            logger.warning("%s: repair retry failed: %s", self.name, self._redact(_describe_exception(exc)))
            return None, LLMUsage(model=self.model)

    # -- LLMClient -------------------------------------------------------

    def decide(
        self,
        system: str,
        context: str,
        allowed_tools: list[dict[str, Any]],
    ) -> tuple[AgentAction, LLMUsage]:
        names = [n for n in (_tool_spec_name(s) for s in allowed_tools or []) if n]
        if not names:
            names = [t.value for t in ToolName]

        user = (
            f"{truncate_context(context, self._max_context)}\n\n"
            f"TOOLS AVAILABLE THIS TURN:\n{_render_tool_menu(allowed_tools)}\n\n"
            f"Choose exactly one next step."
        )

        try:
            payload, usage, raw = self._call_ladder(
                system=system,
                user=user,
                function_name=DECIDE_TOOL_NAME,
                function_schema=_schema_for_decide(names),
                function_description=(
                    "Record the single next research action. `thought` names the "
                    "specific information gap this step closes."
                ),
                temperature=0.2,  # a little latitude: the plan benefits from variety
                max_tokens=1024,
            )
        except Exception as exc:  # noqa: BLE001 - must never raise into the loop
            detail = self._redact(_describe_exception(exc))
            logger.warning("decide(): %s call failed: %s", self.name, detail)
            return (
                AgentAction(tool=ToolName.FINISH, thought=f"LLM error: {detail}"),
                LLMUsage(model=self.model),
            )

        if not isinstance(payload, dict) or not payload:
            logger.warning("decide(): %s returned no usable action", self.name)
            logger.debug("decide(): raw response was %r", (raw or "")[:400])
            return (
                AgentAction(tool=ToolName.FINISH, thought="LLM error: model returned no action."),
                usage,
            )

        return _action_from_payload(payload, names, usage, self.model)

    def extract(
        self,
        system: str,
        context: str,
        schema_model: type[T],
        hint: Optional[str] = None,
    ) -> tuple[T, LLMUsage]:
        schema = _model_json_schema(schema_model)
        function_name = f"record_{schema_model.__name__.lower()}"[:64]
        description = (
            f"Record the extracted {schema_model.__name__}. Leave fields null/empty "
            "when the evidence does not establish them."
        )

        user = truncate_context(context, self._max_context)
        if hint:
            user = f"{user}\n\nFOCUS: {hint}"

        try:
            payload, usage, raw = self._call_ladder(
                system=system,
                user=user,
                function_name=function_name,
                function_schema=schema,
                function_description=description,
                temperature=0.0,  # extraction should be as close to deterministic as we can get
                max_tokens=4096,
            )
        except Exception as exc:  # noqa: BLE001 - must never raise into the loop
            logger.warning("extract(): %s call failed: %s", self.name, self._redact(_describe_exception(exc)))
            return best_effort_instance(schema_model), LLMUsage(model=self.model)

        total = usage
        if isinstance(payload, dict) and payload:
            try:
                return schema_model.model_validate(payload), total
            except ValidationError as exc:
                logger.warning(
                    "extract(): %s failed validation from %s: %s",
                    schema_model.__name__, self.name, exc.errors()[:3],
                )
                errors: Any = exc
        else:
            # Nothing parseable at all — try the tolerant path on the raw text once.
            salvaged = coerce_to_model(raw, schema_model)
            if salvaged is not None:
                return salvaged, total
            payload, errors = raw, "the response contained no JSON object"

        repaired, repair_usage = self._repair(
            system=system,
            user=user,
            bad_payload=payload,
            errors=errors,
            function_name=function_name,
            function_schema=schema,
            function_description=description,
            max_tokens=4096,
        )
        total = LLMUsage(
            tokens_in=total.tokens_in + repair_usage.tokens_in,
            tokens_out=total.tokens_out + repair_usage.tokens_out,
            model=total.model or self.model,
        )
        if isinstance(repaired, dict) and repaired:
            try:
                return schema_model.model_validate(repaired), total
            except ValidationError as exc:
                logger.warning(
                    "extract(): repair retry still invalid for %s: %s",
                    schema_model.__name__, exc.errors()[:3],
                )
                return best_effort_instance(schema_model, repaired), total

        seed = payload if isinstance(payload, dict) else None
        return best_effort_instance(schema_model, seed), total

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception as exc:  # noqa: BLE001 - closing must never raise
                logger.debug("Error closing the HTTP client for %s: %s", self.name, exc)


# ---------------------------------------------------------------------------


def _action_from_payload(
    payload: dict[str, Any],
    names: list[str],
    usage: LLMUsage,
    model: str,
) -> tuple[AgentAction, LLMUsage]:
    thought = str(payload.get("thought") or "").strip()
    requested = str(payload.get("tool") or "").strip()
    args = payload.get("args")
    if isinstance(args, str):
        args = extract_json(args) or {}
    if not isinstance(args, dict):
        args = {}

    chosen, coerced = _closest_tool(requested, names)
    if coerced:
        logger.warning("decide(): model chose unavailable tool %r; using %r", requested, chosen)
        thought = f"[coerced tool {requested!r} -> {chosen!r}] {thought}"

    try:
        tool = ToolName(chosen)
    except ValueError:
        logger.error("decide(): %r is not a ToolName; finishing", chosen)
        return AgentAction(tool=ToolName.FINISH, thought=f"LLM error: unknown tool {chosen!r}."), usage

    try:
        return AgentAction(thought=thought or "(no thought supplied)", tool=tool, args=args), usage
    except ValidationError as exc:
        logger.error("decide(): AgentAction failed validation: %s", exc.errors()[:2])
        return AgentAction(tool=ToolName.FINISH, thought="LLM error: invalid action."), usage


#: `?key=<secret>` can surface in an httpx error's URL. Strip it regardless of
#: which key it is — this must not depend on knowing the credential.
_QUERY_KEY_RE = re.compile(r"([?&](?:key|api_key|access_token)=)[^&\s\"']+", re.IGNORECASE)


def _redact_query_key(text: str) -> str:
    return _QUERY_KEY_RE.sub(r"\1<redacted>", text or "")


def _describe_exception(exc: BaseException) -> str:
    """A single clean line for the agent trace — no JSON braces, no newlines."""
    if isinstance(exc, (_AuthFailure, _Unsupported)):
        text = str(exc)
    elif isinstance(exc, httpx.HTTPStatusError):
        text = str(exc)
    elif isinstance(exc, httpx.HTTPError):
        text = f"network error: {exc}"
    else:
        text = f"{type(exc).__name__}: {exc}"
    flat = " ".join(_redact_query_key(str(text)).split())
    return flat[:299] + "…" if len(flat) > 300 else flat


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_body(response: httpx.Response) -> str:
    try:
        return response.text or ""
    except Exception as exc:  # noqa: BLE001 - reading an error body must not raise
        logger.debug("Could not read the response body: %s", exc)
        return ""


def resolve_model(settings: Settings, provider: Optional[str] = None) -> Optional[str]:
    """The model id that should actually be used, or ``None`` if unknowable.

    Returns the configured id when the endpoint really serves it, otherwise the
    best-ranked live alternative. Cached per (base_url, configured model), so
    `lea doctor` and the agent agree without paying two lookups.
    """
    name = (provider or settings.resolved_llm() or "").strip().lower()
    try:
        base_url, configured = settings.endpoint_for(name)
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        logger.debug("resolve_model: no endpoint for %r: %s", name, exc)
        return None
    if not base_url:
        return None

    key = (base_url.rstrip("/"), configured)
    if key in _MODEL_RESOLUTION:
        return _MODEL_RESOLUTION[key]

    available = list_models(settings, provider=name)
    ranked = rank_models(available, provider=name, configured=configured)
    if not ranked:
        return None
    _MODEL_RESOLUTION[key] = ranked[0]
    return ranked[0]


def list_models(settings: Settings, provider: Optional[str] = None) -> list[str]:
    """Model ids the configured endpoint will actually serve.

    Free model lineups rotate, so any hardcoded default goes stale eventually;
    this lets a user see what their own key serves today. Returns [] on any
    failure — it is a diagnostic, never a hard dependency.
    """
    name = (provider or settings.resolved_llm() or "").strip().lower()
    try:
        base_url, _ = settings.endpoint_for(name)
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        logger.debug("list_models: no endpoint for %r: %s", name, exc)
        return []
    if not base_url:
        return []

    # Use whatever transport this endpoint already accepted, so discovery does
    # not fail on an endpoint the chat path has successfully negotiated.
    base = base_url.rstrip("/")
    headers, params = auth_transport(base, settings.key_for(name))
    headers["Accept"] = "application/json"

    try:
        response = httpx.get(
            f"{base}/models",
            headers=headers,
            params=params,
            timeout=max(10.0, settings.http_timeout_s),
            follow_redirects=True,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - any failure is just "no list"
        logger.debug("list_models(%s): %s: %s", name, type(exc).__name__, _redact_query_key(str(exc)))
        return []

    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            ident = row.get("id") or row.get("name")
        else:
            ident = row
        if ident:
            # Google prefixes ids with "models/"; strip it so the value is
            # usable as-is in LEA_MODEL.
            text = str(ident)
            out.append(text[len("models/"):] if text.startswith("models/") else text)
    return sorted(set(out))
