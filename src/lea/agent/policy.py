"""Guardrails for the autonomous loop.

The loop itself contains no research plan — the LLM decides every step. This module
is the *only* thing that constrains it, and it constrains in three ways:

1. Budget   — hard ceilings on steps / searches / fetches / wall-clock.
2. Filtering — `allowed_tools` removes moves the budget can no longer afford, so the
   model is never offered a choice it cannot make.
3. Feedback — `is_redundant` detects near-duplicate work. Crucially it *reports*
   rather than *blocks*: the loop turns the reason into an observation the model
   reads on the next turn. A silently-dropped action teaches the model nothing and
   usually produces the same action again; a visible "you already fetched X" pushes
   it somewhere new. That self-correction is part of what makes this an agent.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from lea.config import Settings
from lea.models import AgentAction, CompanyRecord, ToolName, TraceStep

# Tools that consume the "fetch a page over HTTP" budget.
_FETCHING_TOOLS = (ToolName.FETCH_PAGE, ToolName.DETECT_TECH)

# How many consecutive useless steps (failed tool call or redundancy guard) before
# we conclude the agent is stuck and stop it ourselves.
MAX_CONSECUTIVE_DEAD_STEPS = 3

# Similarity above which two search queries count as "the same question again".
QUERY_SIMILARITY_THRESHOLD = 0.8

# How many times the loop may push back on a FINISH that would produce an empty
# record. Bounded, because an obstinate model must still be able to stop.
MAX_FINISH_PUSHBACKS = 2

# --------------------------------------------------------------------------
# Stop-reason vocabulary
#
# Every stop_reason is "<class>: <detail>". The class is what callers branch on:
# a report that says tier B because the work finished is a very different thing
# from one that says tier B because every search 403'd, and the CLI exit code
# has to be able to tell them apart.
# --------------------------------------------------------------------------

STOP_COMPLETE = "complete"     # the agent had what it needed
STOP_EXHAUSTED = "exhausted"   # ran out of budget, but learned something
STOP_BLOCKED = "blocked"       # could not learn anything: failures, not limits
STOP_ERROR = "error"           # the harness itself broke

_STOP_CLASSES = (STOP_COMPLETE, STOP_EXHAUSTED, STOP_BLOCKED, STOP_ERROR)


def stop_class(reason: str) -> str:
    """The class of a stop_reason string. Unknown/legacy values read as complete."""
    head = (reason or "").split(":", 1)[0].strip().lower()
    return head if head in _STOP_CLASSES else STOP_COMPLETE


def is_trustworthy(reason: str) -> bool:
    """False when the run's output must not be presented as a real assessment."""
    return stop_class(reason) not in (STOP_BLOCKED, STOP_ERROR)

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "is", "at",
    "with", "about", "by", "from", "what", "who", "how", "company", "inc", "llc",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


@dataclass
class Budget:
    """Tracks what the agent has spent, and what it may still spend."""

    max_steps: int = 12
    max_searches: int = 5
    max_fetches: int = 6
    total_timeout_s: float = 180.0

    steps: int = 0
    searches: int = 0
    fetches: int = 0

    # monotonic clock so a system time change cannot extend or kill a run
    started_monotonic: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Budget":
        return cls(
            max_steps=settings.max_steps,
            max_searches=settings.max_searches,
            max_fetches=settings.max_fetches,
            total_timeout_s=settings.total_timeout_s,
        )

    # -- clock ------------------------------------------------------------
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_monotonic

    def deadline_passed(self) -> bool:
        return self.total_timeout_s > 0 and self.elapsed_s() >= self.total_timeout_s

    # -- permissions ------------------------------------------------------
    def allows(self, tool: ToolName | str | None) -> bool:
        """Can the agent still afford this tool? FINISH is always affordable —
        the agent must always be able to say 'I'm done', even out of budget."""
        if tool == ToolName.FINISH or tool == "finish":
            return True
        if self.deadline_passed():
            return False
        if self.steps >= self.max_steps:
            return False
        if tool in (ToolName.WEB_SEARCH, "web_search"):
            return self.searches < self.max_searches
        if tool in _FETCHING_TOOLS or tool in ("fetch_page", "detect_tech"):
            return self.fetches < self.max_fetches
        # Unknown tool: allow it if there are steps left; the registry will reject it.
        return True

    def consume(self, tool: ToolName | str | None) -> None:
        """Record one step, plus the per-tool resource it used."""
        self.steps += 1
        if tool in (ToolName.WEB_SEARCH, "web_search"):
            self.searches += 1
        elif tool in _FETCHING_TOOLS or tool in ("fetch_page", "detect_tech"):
            self.fetches += 1

    def remaining(self) -> dict[str, Any]:
        return {
            "steps": max(0, self.max_steps - self.steps),
            "searches": max(0, self.max_searches - self.searches),
            "fetches": max(0, self.max_fetches - self.fetches),
            "seconds": max(0.0, round(self.total_timeout_s - self.elapsed_s(), 1)),
        }

    def exhausted_reason(self) -> Optional[str]:
        """A human-readable reason the run must stop, or None."""
        if self.deadline_passed():
            return f"{STOP_EXHAUSTED}: time budget ({self.total_timeout_s:.0f}s)"
        if self.steps >= self.max_steps:
            return f"{STOP_EXHAUSTED}: step budget ({self.steps}/{self.max_steps} steps)"
        if self.searches >= self.max_searches and self.fetches >= self.max_fetches:
            return (
                f"{STOP_EXHAUSTED}: research budget ({self.searches}/{self.max_searches} searches, "
                f"{self.fetches}/{self.max_fetches} pages)"
            )
        return None

    def can_research(self) -> bool:
        """Is there any budget left for a move that could actually learn something?"""
        return self.allows(ToolName.WEB_SEARCH) or self.allows(ToolName.FETCH_PAGE)


# --------------------------------------------------------------------------
# Tool filtering
# --------------------------------------------------------------------------


def _fallback_specs(registry: dict[Any, Any]) -> list[dict[str, Any]]:
    """Derive specs straight off the tool objects, for when lea.tools.registry
    isn't importable yet (parallel development) or a test passes fake tools."""
    specs: list[dict[str, Any]] = []
    for tool in registry.values():
        name = getattr(tool, "name", None)
        specs.append(
            {
                "name": getattr(name, "value", name),
                "description": getattr(tool, "description", ""),
                "args_schema": getattr(tool, "args_schema", {}) or {},
            }
        )
    if not any(s.get("name") == ToolName.FINISH.value for s in specs):
        specs.append(
            {
                "name": ToolName.FINISH.value,
                "description": "Stop researching and synthesize the final record from the evidence gathered.",
                "args_schema": {"reason": "why you have enough to finish"},
            }
        )
    return specs


def allowed_tools(budget: Budget, registry: dict[Any, Any]) -> list[dict[str, Any]]:
    """The tool specs the model may choose from *right now*.

    Filtering here (rather than rejecting afterwards) means the model never wastes
    a turn proposing a search when the search budget is gone.
    """
    try:  # lazy: the registry module lands in parallel with this one
        from lea.tools.registry import tool_specs  # type: ignore

        specs = list(tool_specs(registry))
    except Exception:
        specs = _fallback_specs(registry)

    out: list[dict[str, Any]] = []
    for spec in specs:
        name = spec.get("name")
        if name == ToolName.FINISH.value:
            out.append(spec)
            continue
        if budget.allows(name):
            out.append(spec)

    if not any(s.get("name") == ToolName.FINISH.value for s in out):
        out.append(
            {
                "name": ToolName.FINISH.value,
                "description": "Stop researching and synthesize the final record from the evidence gathered.",
                "args_schema": {"reason": "why you have enough to finish"},
            }
        )
    return out


# --------------------------------------------------------------------------
# Redundancy guard
# --------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Collapse the cosmetic differences that make the same page look like two."""
    if not url:
        return ""
    raw = url.strip()
    if "//" not in raw:
        raw = "//" + raw
    parts = urlsplit(raw if "://" in raw else "http:" + raw)
    host = (parts.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith(":80"):
        host = host[:-3]
    if host.endswith(":443"):
        host = host[:-4]
    path = (parts.path or "").rstrip("/").lower()
    return f"{host}{path}"


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def query_similarity(a: str, b: str) -> float:
    """Jaccard overlap of significant tokens. 1.0 == same question."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    return len(ta & tb) / len(ta | tb)


def _arg(args: dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def is_redundant(action: AgentAction, history: Iterable[TraceStep]) -> Optional[str]:
    """Return a reason string when `action` repeats work already done, else None.

    The caller does NOT drop the action on the floor — it records the reason as a
    failed step so the model sees it next turn and picks something else.
    """
    steps = list(history)
    if action.tool == ToolName.FINISH:
        return None

    if action.tool in _FETCHING_TOOLS:
        url = _arg(action.args, "url", "page", "link", "domain")
        target = normalize_url(url)
        if target:
            for s in steps:
                # Scoped per (tool, url) on purpose. `detect_tech` on a URL that
                # `fetch_page` already read is the CORRECT sequence — read the page,
                # then inspect its stack — so only the SAME tool on the same URL is
                # repeated work. Grouping them cost every run its tech_stack.
                if s.tool != action.tool:
                    continue
                prior = normalize_url(_arg(s.args, "url", "page", "link", "domain"))
                if prior and prior == target:
                    return (
                        f"redundant: you already ran {s.tool.value} on {url} at step {s.n}. "
                        "Pick a different URL, or a different tool on this one."
                    )
        return None

    if action.tool == ToolName.WEB_SEARCH:
        query = _arg(action.args, "query", "q", "search", "text")
        if not query:
            return None
        for s in steps:
            if s.tool != ToolName.WEB_SEARCH:
                continue
            prior = _arg(s.args, "query", "q", "search", "text")
            if not prior:
                continue
            if query_similarity(query, prior) >= QUERY_SIMILARITY_THRESHOLD:
                return (
                    f'redundant: step {s.n} already searched "{prior}", which asks the same '
                    "question. Search for something you do not know yet, or fetch a page."
                )
        return None

    # Any other tool: exact repeat of tool+args.
    for s in steps:
        if s.tool == action.tool and s.args == action.args:
            return f"redundant: step {s.n} already called {action.tool.value} with those exact arguments."
    return None


# --------------------------------------------------------------------------
# Gaps and stop conditions
# --------------------------------------------------------------------------

# Priority order matters: it is both what the prompt tells the model to chase next
# and what `should_stop` uses to decide the record is complete.
GAP_LABELS: dict[str, str] = {
    "domain": "primary web domain",
    "industry": "industry / what they sell",
    "employees": "employee count or range",
    "hq": "headquarters location",
    "signals": "recent buying signals (funding, hiring, launches)",
    "tech": "technology stack",
}


def derive_gaps(record: CompanyRecord) -> list[str]:
    """Which fields are still unknown, most important first."""
    gaps: list[str] = []
    if not record.domain:
        gaps.append("domain")
    if not record.industry:
        gaps.append("industry")
    if not (record.employee_range or record.employee_estimate):
        gaps.append("employees")
    if not (record.hq_city or record.hq_country):
        gaps.append("hq")
    if not record.signals:
        gaps.append("signals")
    if not record.tech_stack:
        gaps.append("tech")
    return gaps


def gap_labels(gaps: Iterable[str]) -> list[str]:
    return [GAP_LABELS.get(g, g) for g in gaps]


# Steps the loop generated to talk to the model (redundancy guard, finish pushback)
# rather than results from the outside world.
GUARD_ERRORS = {"redundant", "premature finish"}


def _consecutive_dead_steps(history: list[TraceStep]) -> int:
    """Consecutive *tool* failures at the tail of the trace.

    Guard feedback is skipped, not counted: a redundancy bounce means the loop
    corrected the model and the model then picks something else, which is the system
    working. Counting it used to end runs one step before the agent tried its
    fallback route — it had said what it would do next and never got the turn.
    """
    n = 0
    for step in reversed(history):
        if step.ok:
            break
        if step.error in GUARD_ERRORS:
            continue  # neither progress nor a dead end; doesn't count, doesn't reset
        n += 1
    return n


def learned_anything(history: Iterable[TraceStep]) -> bool:
    """Did any research step actually succeed? Finish/synthesis steps don't count —
    the agent congratulating itself is not evidence."""
    return any(s.ok and s.tool != ToolName.FINISH for s in history)


def last_failure(history: Iterable[TraceStep]) -> Optional[TraceStep]:
    for s in reversed(list(history)):
        if not s.ok and s.tool != ToolName.FINISH:
            return s
    return None


# The OpenAI-compatible client never raises into the loop: it reports a failed call
# by returning FINISH with a thought starting "LLM error:" (see openai_compat.decide).
# Without recognising that, a dead endpoint looks exactly like an agent that calmly
# decided it was finished — which is how a stale model id becomes invisible.
_LLM_ERROR_PREFIX = "llm error:"

# Substrings meaning "the endpoint rejected this outright", which on a free provider
# is nearly always a model id that has rotated out from under the preset.
_STALE_MODEL_MARKERS = (
    "404", "400", "not found", "does not exist", "invalid model", "unknown model",
    "model_not_found", "decommissioned", "unsupported model", "no such model",
)

MODELS_HINT = "run `lea doctor --models` to see what your provider actually serves, then set LEA_MODEL"


def llm_error_detail(action: AgentAction) -> Optional[str]:
    """The error text when this 'action' is really a failed LLM call, else None."""
    if action.tool != ToolName.FINISH:
        return None
    thought = (action.thought or "").strip()
    return thought if thought.lower().startswith(_LLM_ERROR_PREFIX) else None


def looks_like_stale_model(text: str) -> bool:
    return any(m in (text or "").lower() for m in _STALE_MODEL_MARKERS)


def llm_failure_stop_reason(details: list[str], provider: str = "", model: str = "") -> str:
    """A stop_reason naming the real cause when every LLM call failed.

    Otherwise the run reads as "the agent gave up" and the user never learns that
    their endpoint is 404ing.
    """
    first = (details[0] if details else "").strip()
    where = "/".join(p for p in (provider, model) if p) or "the configured LLM"
    if looks_like_stale_model(first):
        return (
            f"{STOP_ERROR}: {where} rejected every request ({first[:140]}). "
            f"The model id is probably stale — {MODELS_HINT}."
        )
    return f"{STOP_ERROR}: every LLM call to {where} failed ({first[:140]})."


def transient_failure_hint(step: TraceStep) -> str:
    """Appended to a failed step's observation so the model reacts to the *specific*
    error instead of treating any failure as a dead end.

    Deliberately route-agnostic: it names no query, site or order of operations.
    """
    if step.tool == ToolName.WEB_SEARCH:
        return (
            " — this is usually transient (rate limit or blocked provider), not a fact about "
            "the company. Retry with different wording, or reach a source directly with "
            "fetch_page instead of relying on search. Do not finish on a failed search."
        )
    return (
        " — one URL failing says nothing about the company. Try another source or another "
        "tool before concluding anything."
    )


def premature_finish_reason(
    budget: Budget, history: Iterable[TraceStep], pushbacks: int = 0
) -> Optional[str]:
    """Reason to refuse a FINISH, or None to allow it.

    The only case we refuse: the agent wants to stop having learned *nothing*, while
    it still has budget to try. Finishing there yields a record of all-None fields
    and a fabricated-looking score, which is worse than spending another step.
    """
    steps = list(history)
    if learned_anything(steps):
        return None
    if pushbacks >= MAX_FINISH_PUSHBACKS:
        return None
    if not budget.can_research():
        return None

    failure = last_failure(steps)
    detail = ""
    if failure is not None:
        err = (failure.error or failure.observation or "").strip()
        detail = f' The last failure was step {failure.n}: "{err[:200]}".'
    remaining = budget.remaining()
    return (
        "not so fast: no research step has succeeded yet, so finishing now would produce an "
        f"empty record with a meaningless score.{detail} You still have "
        f"{remaining['searches']} search(es) and {remaining['fetches']} fetch(es) left — "
        "use one on a different route before finishing."
    )


def should_stop(
    record: CompanyRecord,
    budget: Budget,
    history: list[TraceStep],
) -> Optional[str]:
    """Reason to stop researching, or None to keep going.

    Never inspects *what* the company is — only budget, completeness and
    productivity. Nothing here encodes a research plan.
    """
    if budget.deadline_passed():
        return f"{STOP_EXHAUSTED}: time budget ({budget.total_timeout_s:.0f}s)"

    reason = budget.exhausted_reason()
    if reason:
        # Out of budget having learned nothing is being blocked, not being done.
        if not learned_anything(history):
            return f"{STOP_BLOCKED}: {reason.split(':', 1)[1].strip()} spent without a single successful step"
        return reason

    if history and not derive_gaps(record):
        return f"{STOP_COMPLETE}: all priority fields resolved"

    dead = _consecutive_dead_steps(history)
    if dead >= MAX_CONSECUTIVE_DEAD_STEPS:
        # A stall is never "done" — say so, so nobody reads the output as an answer.
        if not learned_anything(history):
            return f"{STOP_BLOCKED}: {dead} consecutive failed steps, nothing learned"
        return f"{STOP_EXHAUSTED}: stalled after {dead} consecutive unproductive steps"

    return None
