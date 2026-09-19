"""Anthropic-backed LLMClient.

Both calls are schema-constrained by a FORCED tool call rather than by parsing
JSON out of prose:

  * ``decide``  -> one tool, ``next_action``, whose input_schema is
    ``{thought, tool(enum), args}``. The tool enum is built from the tools the
    agent loop is actually willing to run this turn, so an out-of-range choice is
    structurally hard rather than merely discouraged.
  * ``extract`` -> one tool whose input_schema IS ``schema_model.model_json_schema()``,
    so a well-formed call is a valid record by construction.

Neither call raises into the agent loop. ``decide`` degrades to
``AgentAction(tool=FINISH, ...)``; ``extract`` retries once with a repair prompt
and then returns a best-effort instance.
"""

from __future__ import annotations

import difflib
import json
import logging
from typing import Any, Optional, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

from lea.config import Settings
from lea.llm.base import LLMUsage
from lea.llm.json_utils import extract_json, repair_prompt
from lea.models import AgentAction, ToolName

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Hard ceiling on the rendered context we send. The agent trace grows every step.
MAX_CONTEXT_CHARS = 60_000
_ELISION = "\n\n[... {n} characters elided from the middle of this context ...]\n\n"

DECIDE_TOOL_NAME = "next_action"

#: Models that reject `temperature` / `top_p` (sampling params were removed on the
#: 4.7+ generation). For those we express "how much latitude" via output_config.effort.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

#: Models that reject forced tool_choice ("any"/"tool"). We fall back to auto + an
#: explicit instruction naming the tool.
_NO_FORCED_TOOL_PREFIXES = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
)


def _supports_sampling(model: str) -> bool:
    return not any(model.startswith(p) for p in _NO_SAMPLING_PREFIXES)


def _supports_forced_tool(model: str) -> bool:
    return not any(model.startswith(p) for p in _NO_FORCED_TOOL_PREFIXES)


def truncate_context(text: str, budget: int = MAX_CONTEXT_CHARS) -> str:
    """Keep the head and tail of a very long context, elide the middle.

    The head holds the target and what is known; the tail holds the most recent
    steps and the remaining budget. The middle is the part we can afford to lose.
    """
    if text is None:
        return ""
    if len(text) <= budget:
        return text
    head = int(budget * 0.55)
    tail = budget - head
    elided = len(text) - head - tail
    logger.info("Context of %d chars exceeds budget %d; eliding %d chars", len(text), budget, elided)
    return text[:head] + _ELISION.format(n=elided) + text[-tail:]


# ---------------------------------------------------------------------------
# helpers for tool specs and fallbacks
# ---------------------------------------------------------------------------


def _tool_spec_name(spec: Any) -> Optional[str]:
    """Pull a tool name out of whatever shape the loop handed us."""
    if isinstance(spec, dict):
        raw = spec.get("name") or spec.get("tool")
    else:
        raw = getattr(spec, "name", None)
    if raw is None:
        return None
    return str(getattr(raw, "value", raw))


def _render_tool_menu(allowed_tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for spec in allowed_tools or []:
        name = _tool_spec_name(spec)
        if not name:
            continue
        desc = ""
        schema: Any = None
        if isinstance(spec, dict):
            desc = spec.get("description") or ""
            schema = spec.get("args_schema") or spec.get("input_schema") or spec.get("schema")
        else:
            desc = getattr(spec, "description", "") or ""
            schema = getattr(spec, "args_schema", None)
        lines.append(f"- {name}: {desc}".rstrip())
        if schema:
            try:
                lines.append(f"    args: {json.dumps(schema, default=str)}")
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("Could not render args_schema for %s: %s", name, exc)
    return "\n".join(lines) or "- finish: stop and return what you have"


def _closest_tool(requested: str, allowed: list[str]) -> tuple[str, bool]:
    """Map a model-chosen tool name onto the allowed set. Returns (name, coerced)."""
    if requested in allowed:
        return requested, False
    lowered = {a.lower(): a for a in allowed}
    if requested.lower() in lowered:
        return lowered[requested.lower()], True
    close = difflib.get_close_matches(requested.lower(), list(lowered), n=1, cutoff=0.5)
    if close:
        return lowered[close[0]], True
    if ToolName.FINISH.value in allowed:
        return ToolName.FINISH.value, True
    return (allowed[0] if allowed else ToolName.FINISH.value), True


def _empty_for_annotation(annotation: Any) -> Any:
    """A type-appropriate empty value, used only to build a fallback instance."""
    origin = get_origin(annotation)
    if origin is not None:
        if origin in (list, set, tuple, frozenset):
            return []
        if origin is dict:
            return {}
        # Optional[...] / Union[...]
        args = [a for a in get_args(annotation) if a is not type(None)]  # noqa: E721
        if len(args) != len(get_args(annotation)):
            return None
        if args:
            return _empty_for_annotation(args[0])
        return None
    if annotation is str:
        return ""
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is list:
        return []
    if annotation is dict:
        return {}
    return None


def best_effort_instance(model: type[T], seed: Optional[dict[str, Any]] = None) -> T:
    """Construct *some* valid-ish instance of *model*. Never raises.

    Tries, in order: the seed as-is, the seed minus whatever failed validation,
    required fields filled with type-appropriate empties, and finally
    ``model_construct`` (which skips validation entirely).
    """
    data = dict(seed or {})

    # 1. seed, dropping offending keys until it validates (bounded).
    for _ in range(6):
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            bad = {str(e["loc"][0]) for e in exc.errors() if e.get("loc")}
            removable = bad & set(data)
            if not removable:
                break
            logger.debug("best_effort_instance(%s): dropping %s", model.__name__, sorted(removable))
            for key in removable:
                data.pop(key, None)

    # 2. fill every required field with an empty of the right shape.
    filled = dict(data)
    for name, field in model.model_fields.items():
        if name in filled:
            continue
        if field.is_required():
            filled[name] = _empty_for_annotation(field.annotation)
    try:
        return model.model_validate(filled)
    except ValidationError as exc:
        logger.warning(
            "best_effort_instance(%s): falling back to model_construct (%s)",
            model.__name__,
            exc.errors()[:2],
        )

    # 3. last resort: no validation at all.
    return model.model_construct(**filled)


# ---------------------------------------------------------------------------


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
        self.model = settings.model
        self._max_context = MAX_CONTEXT_CHARS
        if client is not None:
            self._client = client
        else:
            import anthropic  # imported lazily so the stub path needs no SDK

            kwargs: dict[str, Any] = {"max_retries": 2, "timeout": settings.http_timeout_s * 4}
            if settings.anthropic_api_key:
                kwargs["api_key"] = settings.anthropic_api_key
            self._client = anthropic.Anthropic(**kwargs)

    # -- plumbing --------------------------------------------------------

    def _usage(self, response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return LLMUsage(model=self.model)
        tokens_in = 0
        for attr in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = getattr(usage, attr, 0) or 0
            try:
                tokens_in += int(value)
            except (TypeError, ValueError):
                logger.debug("Non-numeric usage.%s: %r", attr, value)
        try:
            tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        except (TypeError, ValueError):
            tokens_out = 0
        return LLMUsage(tokens_in=tokens_in, tokens_out=tokens_out, model=getattr(response, "model", self.model))

    def _call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        forced_tool: str,
        effort: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> Any:
        """One Messages call with a forced tool. Retries once with a minimal
        parameter set if the API rejects an optional parameter."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        if _supports_forced_tool(self.model):
            kwargs["tool_choice"] = {"type": "tool", "name": forced_tool}
        else:
            kwargs["tool_choice"] = {"type": "auto"}
        if _supports_sampling(self.model):
            kwargs["temperature"] = temperature
        else:
            # Sampling params are rejected on this generation; effort is the lever.
            kwargs["output_config"] = {"effort": effort}

        try:
            return self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - narrowed below, then re-raised
            if not _is_bad_request(exc):
                raise
            logger.warning(
                "Anthropic rejected optional params for model %s (%s); retrying minimally",
                self.model,
                exc,
            )
            minimal = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }
            return self._client.messages.create(**minimal)

    @staticmethod
    def _tool_use_block(response: Any, name: str) -> Optional[Any]:
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == name:
                return block
        # Forced tool_choice was unavailable and the model free-chose: take any tool_use.
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "tool_use":
                return block
        return None

    @staticmethod
    def _text_of(response: Any) -> str:
        parts = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        return "\n".join(parts)

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

        tool_def = {
            "name": DECIDE_TOOL_NAME,
            "description": (
                "Record the single next research action. Call this exactly once. "
                "`thought` names the specific information gap this step closes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "One or two sentences: what is still unknown and why this step closes it.",
                    },
                    "tool": {
                        "type": "string",
                        "enum": names,
                        "description": "The tool to run next.",
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments for that tool, matching its documented args schema.",
                        "additionalProperties": True,
                    },
                },
                "required": ["thought", "tool", "args"],
            },
        }

        user = (
            f"{truncate_context(context, self._max_context)}\n\n"
            f"TOOLS AVAILABLE THIS TURN:\n{_render_tool_menu(allowed_tools)}\n\n"
            f"Call the `{DECIDE_TOOL_NAME}` tool exactly once with your chosen next step."
        )

        try:
            response = self._call(
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[tool_def],
                forced_tool=DECIDE_TOOL_NAME,
                effort="medium",
                temperature=0.2,  # a little latitude: the plan benefits from variety
                max_tokens=2048,
            )
        except Exception as exc:  # noqa: BLE001 - must never raise into the loop
            logger.error("decide(): Anthropic call failed: %s: %s", type(exc).__name__, exc)
            return (
                AgentAction(tool=ToolName.FINISH, thought=f"LLM error: {type(exc).__name__}: {exc}"),
                LLMUsage(model=self.model),
            )

        usage = self._usage(response)
        block = self._tool_use_block(response, DECIDE_TOOL_NAME)
        payload: Any = getattr(block, "input", None) if block is not None else None
        if not isinstance(payload, dict):
            # Model answered in prose despite the forced tool — salvage JSON from it.
            payload = extract_json(self._text_of(response)) or {}
        if not payload:
            logger.error("decide(): no usable tool_use block in response")
            return (
                AgentAction(tool=ToolName.FINISH, thought="LLM error: model returned no action."),
                usage,
            )

        thought = str(payload.get("thought") or "").strip()
        requested = str(payload.get("tool") or "").strip()
        args = payload.get("args")
        if isinstance(args, str):
            args = extract_json(args) or {}
        if not isinstance(args, dict):
            args = {}

        chosen, coerced = _closest_tool(requested, names)
        if coerced:
            note = f"[coerced tool '{requested}' -> '{chosen}'] "
            logger.warning("decide(): model chose unavailable tool %r; using %r", requested, chosen)
            thought = note + thought

        try:
            tool = ToolName(chosen)
        except ValueError:
            logger.error("decide(): %r is not a ToolName; finishing", chosen)
            return (
                AgentAction(tool=ToolName.FINISH, thought=f"LLM error: unknown tool {chosen!r}."),
                usage,
            )

        try:
            return AgentAction(thought=thought or "(no thought supplied)", tool=tool, args=args), usage
        except ValidationError as exc:
            logger.error("decide(): AgentAction failed validation: %s", exc.errors()[:2])
            return (
                AgentAction(tool=ToolName.FINISH, thought=f"LLM error: invalid action ({exc.error_count()} issues)."),
                usage,
            )

    def extract(
        self,
        system: str,
        context: str,
        schema_model: type[T],
        hint: Optional[str] = None,
    ) -> tuple[T, LLMUsage]:
        schema = _tool_input_schema(schema_model)
        tool_name = _extract_tool_name(schema_model)
        tool_def = {
            "name": tool_name,
            "description": (
                f"Record the extracted {schema_model.__name__}. Call this exactly once. "
                "Leave fields null/empty when the evidence does not establish them."
            ),
            "input_schema": schema,
        }

        user = truncate_context(context, self._max_context)
        if hint:
            user = f"{user}\n\nFOCUS: {hint}"
        user = f"{user}\n\nCall the `{tool_name}` tool exactly once with the extracted record."

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        total = LLMUsage(model=self.model)

        for attempt in (1, 2):
            try:
                response = self._call(
                    system=system,
                    messages=messages,
                    tools=[tool_def],
                    forced_tool=tool_name,
                    effort="low",
                    temperature=0.0,  # extraction should be as close to deterministic as we can get
                    max_tokens=8192,
                )
            except Exception as exc:  # noqa: BLE001 - must never raise into the loop
                logger.error("extract(): Anthropic call failed on attempt %d: %s: %s", attempt, type(exc).__name__, exc)
                return best_effort_instance(schema_model), total

            usage = self._usage(response)
            total = LLMUsage(
                tokens_in=total.tokens_in + usage.tokens_in,
                tokens_out=total.tokens_out + usage.tokens_out,
                model=usage.model or self.model,
            )

            block = self._tool_use_block(response, tool_name)
            payload: Any = getattr(block, "input", None) if block is not None else None
            if not isinstance(payload, dict):
                payload = extract_json(self._text_of(response))
            if not isinstance(payload, dict):
                logger.warning("extract(): attempt %d produced no object", attempt)
                payload = {}

            try:
                return schema_model.model_validate(payload), total
            except ValidationError as exc:
                logger.warning(
                    "extract(): %s failed validation on attempt %d: %s",
                    schema_model.__name__,
                    attempt,
                    exc.errors()[:3],
                )
                if attempt == 1:
                    messages = [
                        {"role": "user", "content": user},
                        {"role": "user", "content": repair_prompt(payload, exc)},
                    ]
                    continue
                return best_effort_instance(schema_model, payload), total

        # Unreachable, but keeps the type checker and the contract honest.
        return best_effort_instance(schema_model), total


def _extract_tool_name(schema_model: type[BaseModel]) -> str:
    raw = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in schema_model.__name__)
    return f"record_{raw.lower()}"[:64]


def _tool_input_schema(schema_model: type[BaseModel]) -> dict[str, Any]:
    """`model_json_schema()`, shaped into something the API accepts as input_schema."""
    try:
        schema = schema_model.model_json_schema()
    except Exception as exc:  # noqa: BLE001 - pydantic can raise several types here
        logger.error("Could not build JSON schema for %s: %s", schema_model.__name__, exc)
        return {"type": "object", "properties": {}, "additionalProperties": True}
    schema = dict(schema)
    schema.setdefault("type", "object")
    schema.pop("title", None)
    return schema


def _is_bad_request(exc: BaseException) -> bool:
    """True for a 400 from the API — i.e. we sent something the model rejects."""
    status = getattr(exc, "status_code", None)
    if status == 400:
        return True
    return type(exc).__name__ in ("BadRequestError", "UnprocessableEntityError")
