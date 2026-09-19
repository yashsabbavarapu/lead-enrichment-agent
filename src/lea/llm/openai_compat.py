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
import random
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

__all__ = ["OpenAICompatClient", "list_models", "Capability", "reset_capability_cache"]

DECIDE_TOOL_NAME = "next_action"

#: Bounded, jittered retry. Free tiers rate-limit hard; a batch run must bend
#: rather than break.
MAX_ATTEMPTS = 4
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 20.0
#: Never honour an absurd Retry-After — better to degrade than hang a run.
MAX_RETRY_AFTER_S = 30.0


class Capability:
    """Rungs of the structured-output ladder, best first."""

    TOOLS = "tools"
    JSON_OBJECT = "json_object"
    PROSE = "prose"

    ORDER = (TOOLS, JSON_OBJECT, PROSE)


#: (base_url, model) -> the best rung known to work. Process-local by design:
#: it is a performance cache, not state worth persisting.
_CAPABILITY_CACHE: dict[tuple[str, str], str] = {}


def reset_capability_cache() -> None:
    """Forget probed capabilities. Exposed for tests and for `lea doctor`."""
    _CAPABILITY_CACHE.clear()


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


def _looks_unsupported(status: int, body: str) -> bool:
    """A 4xx that means "drop down a rung", not "give up"."""
    if status not in (400, 404, 405, 415, 422, 501):
        return False
    lowered = (body or "").lower()
    return any(marker in lowered for marker in _UNSUPPORTED_MARKERS)


class _Unsupported(Exception):
    """The endpoint rejected this rung of the ladder."""


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    raw = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-after")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), MAX_RETRY_AFTER_S))
    except (TypeError, ValueError):
        logger.debug("Unparseable Retry-After header: %r", raw)
        return None


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
        self.model = model
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
        logger.warning(
            "%s/%s does not support %s (%s); falling back to %s for the rest of this run.",
            self.name, self.model, failed, reason, nxt,
        )
        return nxt

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # Ollama is local and keyless: sending an empty bearer token makes some
        # servers 401 on a request that would otherwise have worked.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /chat/completions with bounded jittered retry.

        Raises `_Unsupported` when the endpoint rejects a capability, and the
        last transport/HTTP error when retries are exhausted.
        """
        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.post(url, headers=self._headers(), json=payload)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("%s: transport error on attempt %d/%d: %s", self.name, attempt, MAX_ATTEMPTS, exc)
                self._sleep(attempt, None)
                continue

            status = response.status_code
            if status < 300:
                try:
                    return response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    last_exc = exc
                    logger.warning("%s: non-JSON body on a %d response: %s", self.name, status, exc)
                    self._sleep(attempt, None)
                    continue

            body = _safe_body(response)
            if _looks_unsupported(status, body):
                raise _Unsupported(f"HTTP {status}: {body[:300]}")

            if status == 429 or status >= 500:
                last_exc = httpx.HTTPStatusError(f"HTTP {status}: {body[:300]}", request=response.request, response=response)
                if attempt == MAX_ATTEMPTS:
                    break
                wait = _retry_after_seconds(response)
                logger.warning(
                    "%s: HTTP %d on attempt %d/%d%s",
                    self.name, status, attempt, MAX_ATTEMPTS,
                    f"; Retry-After {wait}s" if wait is not None else "",
                )
                self._sleep(attempt, wait)
                continue

            # A 4xx we cannot interpret as a capability gap: not retryable.
            raise httpx.HTTPStatusError(f"HTTP {status}: {body[:300]}", request=response.request, response=response)

        raise last_exc or RuntimeError(f"{self.name}: request failed after {MAX_ATTEMPTS} attempts")

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
            logger.warning("%s: could not parse tool_call arguments: %r", name, str(raw)[:200])
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
            logger.warning("%s: repair retry failed: %s: %s", self.name, type(exc).__name__, exc)
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
            logger.error("decide(): %s call failed: %s: %s", self.name, type(exc).__name__, exc)
            return (
                AgentAction(tool=ToolName.FINISH, thought=f"LLM error: {type(exc).__name__}: {exc}"),
                LLMUsage(model=self.model),
            )

        if not isinstance(payload, dict) or not payload:
            logger.error("decide(): %s returned no usable action (raw: %r)", self.name, (raw or "")[:200])
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
            logger.error("extract(): %s call failed: %s: %s", self.name, type(exc).__name__, exc)
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

    headers = {"Accept": "application/json"}
    key = settings.key_for(name)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers=headers,
            timeout=max(10.0, settings.http_timeout_s),
            follow_redirects=True,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - any failure is just "no list"
        logger.debug("list_models(%s): %s: %s", name, type(exc).__name__, exc)
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
