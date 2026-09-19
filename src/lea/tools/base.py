"""Tool protocol. Tools are pure-ish: they take args, return an Observation, never raise."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from lea.config import Settings
from lea.models import Observation, ToolName


@runtime_checkable
class Tool(Protocol):
    name: ToolName
    description: str
    # JSON-schema-ish dict describing args, surfaced to the LLM in the prompt.
    args_schema: dict[str, Any]

    def run(self, args: dict[str, Any], settings: Settings) -> Observation:
        """Execute. MUST catch its own exceptions and return Observation(ok=False, error=...)."""
        ...
