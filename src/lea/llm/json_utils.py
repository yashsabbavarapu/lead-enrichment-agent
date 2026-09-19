"""Tolerant JSON extraction for model output.

Models wrap JSON in prose, in ```json fences, and occasionally leave a trailing
comma. Nothing in here raises: the caller gets ``None`` and decides what to do.

Leaf module — imports nothing from lea.* except typing-safe pydantic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json|JSON|javascript|js)?\s*\n?(.*?)```", re.DOTALL)

__all__ = [
    "strip_code_fences",
    "find_json_object",
    "strip_trailing_commas",
    "extract_json",
    "coerce_to_model",
    "repair_prompt",
]


def strip_code_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    # Unterminated fence: ```json { ... (no closing fence)
    if "```" in text:
        after = text.split("```", 1)[1]
        after = re.sub(r"^(json|JSON|javascript|js)\s*\n?", "", after)
        return after.replace("```", "").strip()
    return text.strip()


def find_json_object(text: str) -> Optional[str]:
    """Return the outermost balanced ``{...}`` span in *text*.

    String-aware (braces inside string literals do not count). If the object is
    never closed, the missing closers are appended as a best-effort repair.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Truncated output — close what is open.
    if depth > 0:
        logger.debug("find_json_object: unbalanced JSON, appending %d closing brace(s)", depth)
        tail = '"' if in_str else ""
        return text[start:] + tail + ("}" * depth)
    return None


def strip_trailing_commas(blob: str) -> str:
    """Remove ``,`` that directly precedes a ``}`` or ``]``, outside of strings."""
    out: list[str] = []
    in_str = False
    escape = False
    for i, ch in enumerate(blob):
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            continue
        if ch == ",":
            rest = blob[i + 1 :]
            stripped = rest.lstrip(" \t\r\n")
            if stripped[:1] in ("}", "]"):
                continue  # drop it
        out.append(ch)
    return "".join(out)


def extract_json(raw: Any) -> Optional[dict[str, Any]]:
    """Best-effort: pull a JSON object out of anything a model handed back."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        logger.debug("extract_json: unsupported input type %s", type(raw).__name__)
        return None

    candidates: list[str] = []
    unfenced = strip_code_fences(raw)
    for text in (unfenced, raw):
        obj = find_json_object(text)
        if obj and obj not in candidates:
            candidates.append(obj)
        text = text.strip()
        if text.startswith("{") and text not in candidates:
            candidates.append(text)

    for cand in candidates:
        for attempt in (cand, strip_trailing_commas(cand)):
            try:
                parsed = json.loads(attempt)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug("extract_json: json.loads failed (%s)", exc)
                continue
            if isinstance(parsed, dict):
                return parsed
            logger.debug("extract_json: parsed a %s, not an object", type(parsed).__name__)
    return None


def coerce_to_model(raw: Any, model: type[T]) -> Optional[T]:
    """Parse *raw* into *model*. Returns ``None`` instead of raising."""
    data = extract_json(raw)
    if data is None:
        logger.debug("coerce_to_model: no JSON object found for %s", model.__name__)
        return None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        logger.debug("coerce_to_model: %s failed validation: %s", model.__name__, exc.errors()[:3])
        return None


def repair_prompt(raw: Any, errors: Any) -> str:
    """A short corrective instruction for one retry after a validation failure."""
    if isinstance(errors, ValidationError):
        items = errors.errors()
    elif isinstance(errors, (list, tuple)):
        items = list(errors)
    else:
        items = [errors]

    lines: list[str] = []
    for err in items[:8]:
        if isinstance(err, dict):
            loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            lines.append(f"- {loc}: {err.get('msg', 'invalid')}")
        else:
            lines.append(f"- {err}")

    snippet = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    snippet = (snippet or "")[:1500]

    return (
        "Your previous output did not validate against the required schema.\n\n"
        f"What you returned:\n{snippet}\n\n"
        "Validation errors:\n" + ("\n".join(lines) or "- unparseable JSON") + "\n\n"
        "Fix ONLY these problems and return the corrected object. Keep every value you "
        "already had right. Use null for anything you genuinely cannot determine from the "
        "evidence — do not invent a value to satisfy the schema."
    )
