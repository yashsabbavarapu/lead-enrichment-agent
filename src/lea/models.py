"""Shared data contracts. Every module in this package codes against these types.

Nothing here imports from lea.* — this is the leaf module of the dependency graph.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Tool layer
# --------------------------------------------------------------------------


class ToolName(str, Enum):
    WEB_SEARCH = "web_search"
    FETCH_PAGE = "fetch_page"
    DETECT_TECH = "detect_tech"
    FINISH = "finish"


class SearchResult(BaseModel):
    title: str = ""
    url: str
    snippet: str = ""


class Observation(BaseModel):
    """What a tool hands back to the agent loop. Never raises into the loop."""

    tool: ToolName
    ok: bool = True
    # Compact text the LLM actually reads. Tools are responsible for truncating.
    summary: str = ""
    # Structured payload for downstream extraction (search hits, page text, tech list).
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    # URLs this observation makes citable.
    sources: list[str] = Field(default_factory=list)
    latency_ms: int = 0


# --------------------------------------------------------------------------
# Agent loop / trace
# --------------------------------------------------------------------------


class AgentAction(BaseModel):
    """The LLM's decision for one turn: why, and what to do next."""

    thought: str = Field(description="One or two sentences: what is still unknown and why this step closes the gap.")
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """One plan->act->observe cycle, rendered in the CLI and exported to JSON."""

    n: int
    at: str = Field(default_factory=_now)
    thought: str = ""
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    observation: str = ""
    sources: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


# --------------------------------------------------------------------------
# Enrichment output
# --------------------------------------------------------------------------

SignalKind = Literal["funding", "news", "hiring", "product", "leadership", "partnership", "other"]


class Signal(BaseModel):
    kind: SignalKind = "other"
    headline: str
    detail: str = ""
    date: Optional[str] = None  # ISO-ish or free text as published
    source_url: Optional[str] = None
    recency_note: str = ""  # e.g. "within 30 days" / "undated"


class TechStack(BaseModel):
    """Inferred, never asserted. Each entry carries how we know."""

    category: str  # "analytics", "crm", "hosting", ...
    name: str
    evidence: str = ""
    confidence: float = 0.5


class ICPCriterion(BaseModel):
    key: str
    description: str
    weight: float = 1.0
    # How to read evidence for this criterion, in plain language.
    rubric: str = ""


class ICPProfile(BaseModel):
    """Derived from example customers, or loaded from icp.yaml."""

    name: str = "default"
    summary: str = ""
    criteria: list[ICPCriterion] = Field(default_factory=list)
    disqualifiers: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)  # example customer names
    created_at: str = Field(default_factory=_now)


class CriterionScore(BaseModel):
    key: str
    score: float = Field(ge=0, le=10)
    weight: float = 1.0
    reasoning: str = ""
    evidence_urls: list[str] = Field(default_factory=list)


class ICPFit(BaseModel):
    score: float = Field(ge=0, le=100, description="Weighted 0-100.")
    tier: Literal["A", "B", "C", "D"] = "C"
    summary: str = ""
    criteria: list[CriterionScore] = Field(default_factory=list)
    disqualifiers_hit: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class OutreachHook(BaseModel):
    angle: str  # short label, e.g. "Series B -> new GTM headcount"
    hook: str  # the actual 1-2 sentence opener
    grounded_in: str = ""  # which signal/fact it leans on
    source_url: Optional[str] = None


class CompanyRecord(BaseModel):
    """The deliverable for one company."""

    input: str
    company_name: Optional[str] = None
    domain: Optional[str] = None
    description: str = ""
    industry: Optional[str] = None
    sub_industry: Optional[str] = None
    employee_range: Optional[str] = None
    employee_estimate: Optional[int] = None
    hq_city: Optional[str] = None
    hq_country: Optional[str] = None
    founded_year: Optional[int] = None
    signals: list[Signal] = Field(default_factory=list)
    tech_stack: list[TechStack] = Field(default_factory=list)
    icp_fit: Optional[ICPFit] = None
    hooks: list[OutreachHook] = Field(default_factory=list)
    # Fields the agent could not establish. Honesty beats hallucination.
    unknowns: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class RunStats(BaseModel):
    steps: int = 0
    searches: int = 0
    fetches: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms: int = 0
    stop_reason: str = ""
    llm_provider: str = ""
    search_provider: str = ""


class EnrichmentResult(BaseModel):
    """What one `enrich` invocation returns: the record plus the visible agent work."""

    record: CompanyRecord
    trace: list[TraceStep] = Field(default_factory=list)
    stats: RunStats = Field(default_factory=RunStats)
    started_at: str = Field(default_factory=_now)
    finished_at: Optional[str] = None
