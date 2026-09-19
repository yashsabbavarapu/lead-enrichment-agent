"""Tool registry: what the agent loop can call, and how it is described to the LLM."""

from __future__ import annotations

from typing import Any

from lea.config import Settings
from lea.models import ToolName
from lea.tools.base import Tool

from .fetch import FetchPageTool
from .search import WebSearchTool
from .tech_detect import DetectTechTool

FINISH_DESCRIPTION = (
    "Stop researching and write the final record. Call this once you have enough "
    "evidence, or when further steps are unlikely to add anything."
)

FINISH_ARGS_SCHEMA: dict[str, Any] = {
    "reason": {
        "type": "string",
        "required": True,
        "description": "One sentence: why the research is complete (or why it cannot continue).",
    }
}


def build_registry(settings: Settings) -> dict[ToolName, Tool]:
    """The executable tools. FINISH is deliberately absent — the loop handles it."""
    tools: list[Tool] = [WebSearchTool(), FetchPageTool(), DetectTechTool()]
    return {t.name: t for t in tools}


def tool_specs(registry: dict[ToolName, Tool]) -> list[dict[str, Any]]:
    """Prompt-facing description of every callable action, FINISH included."""
    specs: list[dict[str, Any]] = []
    for name in (ToolName.WEB_SEARCH, ToolName.FETCH_PAGE, ToolName.DETECT_TECH):
        tool = registry.get(name)
        if tool is None:
            continue
        specs.append(
            {
                "name": tool.name.value,
                "description": tool.description,
                "args_schema": tool.args_schema,
            }
        )
    # Anything in the registry the loop does not know about statically.
    for name, tool in registry.items():
        if name in (ToolName.WEB_SEARCH, ToolName.FETCH_PAGE, ToolName.DETECT_TECH):
            continue
        specs.append(
            {"name": tool.name.value, "description": tool.description, "args_schema": tool.args_schema}
        )
    specs.append(
        {
            "name": ToolName.FINISH.value,
            "description": FINISH_DESCRIPTION,
            "args_schema": FINISH_ARGS_SCHEMA,
        }
    )
    return specs
