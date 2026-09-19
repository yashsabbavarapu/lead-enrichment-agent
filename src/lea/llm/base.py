"""LLM protocol.

Only two calls the rest of the codebase ever makes:
  - decide():  pick the next agent action (constrained to the tool set)
  - extract(): fill a pydantic model from evidence, returning valid JSON

Both must work for the keyless `stub` provider too, so the CLI runs with no keys.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from lea.models import AgentAction

T = TypeVar("T", bound=BaseModel)


class LLMUsage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    name: str  # "anthropic" | "stub"
    model: str

    def decide(
        self,
        system: str,
        context: str,
        allowed_tools: list[dict[str, Any]],
    ) -> tuple[AgentAction, LLMUsage]:
        """Choose the next tool call. Must always return a valid AgentAction."""
        ...

    def extract(
        self,
        system: str,
        context: str,
        schema_model: type[T],
        hint: Optional[str] = None,
    ) -> tuple[T, LLMUsage]:
        """Return an instance of schema_model parsed from context. Never raise on
        malformed model output — retry, then fall back to a best-effort instance."""
        ...
