"""Agent loop + policy tests. No network, no real LLM, no real tools.

Everything the loop reaches for goes through a `_get_*` / `_call_*` seam in
lea.agent.loop, so these tests replace the whole outside world with fakes.
"""

from __future__ import annotations

from typing import Any

import pytest

from lea.agent import loop as loop_mod
from lea.agent.policy import (
    Budget,
    allowed_tools,
    derive_gaps,
    gap_labels,
    is_redundant,
    learned_anything,
    normalize_url,
    premature_finish_reason,
    query_similarity,
    should_stop,
    stop_class,
    transient_failure_hint,
)
from lea.config import Settings
from lea.llm.base import LLMUsage
from lea.models import (
    AgentAction,
    CompanyRecord,
    ICPCriterion,
    ICPFit,
    ICPProfile,
    Observation,
    OutreachHook,
    ToolName,
    TraceStep,
)

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeTool:
    """A tool that returns canned observations, or raises if told to."""

    def __init__(self, name: ToolName, observations=None, raises: Exception | None = None):
        self.name = name
        self.description = f"fake {name.value}"
        self.args_schema = {"query": "str"} if name == ToolName.WEB_SEARCH else {"url": "str"}
        self._observations = list(observations or [])
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def run(self, args: dict[str, Any], settings: Settings) -> Observation:
        self.calls.append(dict(args))
        if self._raises is not None:
            raise self._raises
        if self._observations:
            return self._observations.pop(0)
        return Observation(
            tool=self.name,
            ok=True,
            summary=f"{self.name.value} default observation",
            data={"text": "some page text about the company"},
            sources=[f"https://example.com/{self.name.value}/{len(self.calls)}"],
            latency_ms=7,
        )


class FakeLLM:
    """Replays a scripted sequence of AgentActions; repeats the last one forever."""

    name = "fake"
    model = "fake-model"

    def __init__(self, script: list[AgentAction]):
        self.script = list(script)
        self.calls: list[tuple[str, str, list[dict[str, Any]]]] = []

    def decide(self, system, context, allowed):
        self.calls.append((system, context, allowed))
        action = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return action, LLMUsage(tokens_in=100, tokens_out=20, model=self.model)

    def extract(self, system, context, schema_model, hint=None):  # pragma: no cover
        return schema_model(), LLMUsage()


def act(tool: ToolName, thought: str = "because", **args) -> AgentAction:
    return AgentAction(thought=thought, tool=tool, args=args)


def make_settings(max_jobs_lookups: int | None = None, **kw) -> Settings:
    base = dict(
        llm_provider="stub", search_provider="fixture",
        max_steps=6, max_searches=4, max_fetches=4, total_timeout_s=60.0,
        offline=True,
    )
    base.update(kw)
    settings = Settings(**base)
    if max_jobs_lookups is not None:
        # Not a Settings field; Budget.from_settings reads it through getattr.
        settings.max_jobs_lookups = max_jobs_lookups
    return settings


DEFAULT_ICP = ICPProfile(
    name="test-icp",
    summary="fast-growing fintech",
    criteria=[ICPCriterion(key="size", description="50-500 employees", weight=2.0)],
)


@pytest.fixture
def wired(monkeypatch):
    """Patch every seam. Returns a helper that installs a given llm + registry."""

    state: dict[str, Any] = {}

    def install(llm, registry, *, extract=None, fit=None, hooks=None, extract_raises=None):
        state["llm"] = llm
        state["registry"] = registry

        monkeypatch.setattr(loop_mod, "_get_llm", lambda s: llm)
        monkeypatch.setattr(loop_mod, "_get_registry", lambda s: registry)

        # Record the record the loop shows the model each turn (the "steering view"),
        # so tests can tell it apart from the authoritative record.
        steering: list[CompanyRecord] = []
        budgets: list[dict[str, Any]] = []
        state["steering"] = steering
        state["budgets"] = budgets

        def _build_context(target, record_so_far, trace, remaining, gaps):
            steering.append(record_so_far.model_copy(deep=True))
            budgets.append(dict(remaining or {}))
            return "CONTEXT"

        monkeypatch.setattr(loop_mod, "_get_prompts", lambda: ("SYSTEM", _build_context))
        monkeypatch.setattr(loop_mod, "_default_icp", lambda s: DEFAULT_ICP)

        def _extract(target, evidence, llm_, settings, base):
            if extract_raises:
                raise extract_raises
            state["evidence"] = list(evidence)
            out = extract if extract is not None else base.model_copy(deep=True)
            out.company_name = out.company_name or "Acme Inc"
            out.industry = out.industry or "fintech"
            return out

        monkeypatch.setattr(loop_mod, "_call_extract", _extract)
        monkeypatch.setattr(
            loop_mod, "_call_icp",
            lambda rec, icp, ev, llm_, s: fit if fit is not None else ICPFit(score=72.0, tier="B", summary="good fit"),
        )
        monkeypatch.setattr(
            loop_mod, "_call_hooks",
            lambda rec, f, ev, llm_, s: hooks if hooks is not None else [
                OutreachHook(angle="growth", hook="Saw you raised a Series B.")
            ],
        )
        return state

    return install


def registry_of(*tools: FakeTool) -> dict[ToolName, FakeTool]:
    return {t.name: t for t in tools}


# --------------------------------------------------------------------------
# Policy: budget
# --------------------------------------------------------------------------


def test_budget_allows_and_consumes():
    b = Budget(max_steps=9, max_searches=1, max_fetches=2, max_jobs=1, total_timeout_s=60)
    assert b.allows(ToolName.WEB_SEARCH)
    b.consume(ToolName.WEB_SEARCH)
    assert b.searches == 1 and b.steps == 1
    assert not b.allows(ToolName.WEB_SEARCH)
    assert b.allows(ToolName.FETCH_PAGE)
    b.consume(ToolName.FETCH_PAGE)
    b.consume(ToolName.DETECT_TECH)  # detect_tech spends the fetch budget too
    assert b.fetches == 2
    assert not b.allows(ToolName.FETCH_PAGE)

    # fetch_jobs has its OWN allowance: it is one JSON call, not a page crawl, so
    # it is still affordable with the page budget gone.
    assert b.allows(ToolName.FETCH_JOBS)
    assert b.exhausted_reason() is None, "a usable job-board lookup is research left"
    b.consume(ToolName.FETCH_JOBS)
    assert b.jobs == 1 and b.fetches == 2, "fetch_jobs must not spend the page budget"
    assert not b.allows(ToolName.FETCH_JOBS)

    # finish is always affordable, even with nothing left
    assert b.allows(ToolName.FINISH)
    assert "exhausted: research budget" in (b.exhausted_reason() or "")
    assert "job boards" in (b.exhausted_reason() or "")
    assert not b.can_research()


def test_budget_remaining_and_step_exhaustion():
    b = Budget(max_steps=2, max_searches=9, max_fetches=9, total_timeout_s=60)
    assert b.remaining()["steps"] == 2
    b.consume(ToolName.WEB_SEARCH)
    b.consume(ToolName.WEB_SEARCH)
    assert b.remaining()["steps"] == 0
    assert "exhausted: step budget" in (b.exhausted_reason() or "")


def test_budget_deadline():
    b = Budget(max_steps=9, total_timeout_s=0.0001)
    b.started_monotonic -= 5.0
    assert b.deadline_passed()
    assert "exhausted: time budget" in (b.exhausted_reason() or "")
    assert b.allows(ToolName.FINISH)
    assert not b.allows(ToolName.WEB_SEARCH)


def test_allowed_tools_filters_to_what_budget_permits():
    registry = registry_of(
        FakeTool(ToolName.WEB_SEARCH),
        FakeTool(ToolName.FETCH_PAGE),
        FakeTool(ToolName.DETECT_TECH),
    )
    b = Budget(max_steps=9, max_searches=0, max_fetches=3, total_timeout_s=60)
    names = {s["name"] for s in allowed_tools(b, registry)}
    assert "web_search" not in names       # no search budget left
    assert "fetch_page" in names
    assert "finish" in names               # finish is always offered


# --------------------------------------------------------------------------
# Policy: redundancy, gaps, stopping
# --------------------------------------------------------------------------


def test_normalize_url_collapses_cosmetic_differences():
    assert normalize_url("https://www.Ramp.com/about/") == normalize_url("http://ramp.com/about")


def test_query_similarity():
    assert query_similarity("ramp.com funding 2025", "funding ramp.com 2025") == 1.0
    assert query_similarity("ramp funding", "ramp headquarters location") < 0.8


def test_is_redundant_detects_repeat_fetch_and_near_identical_search():
    history = [
        TraceStep(n=1, tool=ToolName.FETCH_PAGE, args={"url": "https://www.ramp.com/about/"}),
        TraceStep(n=2, tool=ToolName.WEB_SEARCH, args={"query": "ramp.com series d funding"}),
    ]
    assert is_redundant(act(ToolName.FETCH_PAGE, url="http://ramp.com/about"), history)
    assert is_redundant(act(ToolName.WEB_SEARCH, query="series d funding ramp.com"), history)
    assert is_redundant(act(ToolName.FETCH_PAGE, url="https://ramp.com/careers"), history) is None
    assert is_redundant(act(ToolName.WEB_SEARCH, query="ramp employee headcount"), history) is None
    assert is_redundant(act(ToolName.FINISH), history) is None


def test_redundancy_is_scoped_per_tool_not_per_url():
    """Regression: grouping fetch_page and detect_tech by URL flagged the CORRECT
    sequence (read a page, then inspect its stack) as redundant, so every run
    finished with an empty tech_stack."""
    history = [TraceStep(n=2, tool=ToolName.FETCH_PAGE, args={"url": "https://ramp.com/about-us/"})]

    # detect_tech on a page fetch_page already read is exactly what should happen
    assert is_redundant(act(ToolName.DETECT_TECH, url="https://ramp.com/about-us/"), history) is None
    # ...but fetching the same URL twice is still wasteful
    dup = is_redundant(act(ToolName.FETCH_PAGE, url="https://www.ramp.com/about-us"), history)
    assert dup and "fetch_page" in dup

    # and detect_tech twice on the same URL is caught too
    history.append(TraceStep(n=3, tool=ToolName.DETECT_TECH, args={"url": "https://ramp.com/about-us/"}))
    again = is_redundant(act(ToolName.DETECT_TECH, url="http://ramp.com/about-us"), history)
    assert again and "detect_tech" in again


def test_derive_gaps_narrows_as_record_fills():
    rec = CompanyRecord(input="ramp.com")
    assert derive_gaps(rec)[0] == "domain"
    rec.domain = "ramp.com"
    assert "domain" not in derive_gaps(rec)
    assert "industry" in derive_gaps(rec)


def test_should_stop_on_stall():
    rec = CompanyRecord(input="x")
    b = Budget(max_steps=20, max_searches=9, max_fetches=9, total_timeout_s=60)
    history = [TraceStep(n=i, tool=ToolName.WEB_SEARCH, ok=False) for i in range(1, 4)]
    # three failures having learned nothing is *blocked*, never "done"
    assert should_stop(rec, b, history) == "blocked: 3 consecutive failed steps, nothing learned"
    assert should_stop(rec, b, history[:1]) is None

    # same stall, but something did land earlier -> out of road, not blocked
    productive = [TraceStep(n=0, tool=ToolName.WEB_SEARCH, ok=True)] + history
    assert stop_class(should_stop(rec, b, productive) or "") == "exhausted"


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_loop_terminates_on_finish(wired):
    search = FakeTool(ToolName.WEB_SEARCH)
    fetch = FakeTool(ToolName.FETCH_PAGE)
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "find the site", query="acme corp official website"),
        act(ToolName.FETCH_PAGE, "read the about page", url="https://acme.com/about"),
        act(ToolName.FINISH, "enough evidence", reason="have the basics"),
    ])
    wired(llm, registry_of(search, fetch))

    result = loop_mod.run_agent("acme corp", make_settings())

    assert result.stats.stop_reason == "complete: agent called finish"
    tools_used = [s.tool for s in result.trace]
    assert tools_used[:3] == [ToolName.WEB_SEARCH, ToolName.FETCH_PAGE, ToolName.FINISH]
    # 2 research steps + the finish decision + 3 synthesis steps, all visible
    assert len(result.trace) == 6
    # target was a plain name, so it seeds company_name and wins the merge
    assert result.record.company_name == "acme corp"
    assert result.record.icp_fit is not None and result.record.icp_fit.tier == "B"
    assert result.record.hooks and result.record.hooks[0].angle == "growth"
    assert result.finished_at


def test_loop_terminates_on_budget_exhaustion(wired):
    """A model that never finishes must still be stopped, with an honest reason."""
    # Sourceless results on purpose: this test is about the step budget, and a
    # search that surfaced unread URLs would (correctly) trip the balance guard
    # instead, which is covered separately.
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(tool=ToolName.WEB_SEARCH, ok=True, summary="no usable hits", data={})
        for _ in range(4)
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "q1", query="acme revenue"),
        act(ToolName.WEB_SEARCH, "q2", query="acme headquarters address"),
        act(ToolName.WEB_SEARCH, "q3", query="acme hiring engineers 2025"),
        act(ToolName.WEB_SEARCH, "q4", query="acme customers case studies"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme", make_settings(max_steps=3, max_searches=9))

    assert "exhausted: step budget" in result.stats.stop_reason
    assert result.stats.searches == 3
    assert len(search.calls) == 3


def test_loop_stops_when_search_budget_runs_out(wired):
    search = FakeTool(ToolName.WEB_SEARCH)
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "q1", query="acme revenue"),
        act(ToolName.WEB_SEARCH, "q2", query="acme headquarters address"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent(
        "acme", make_settings(max_steps=9, max_searches=2, max_fetches=0, max_jobs_lookups=0)
    )

    assert "exhausted: research budget" in result.stats.stop_reason
    assert result.stats.searches == 2


def test_redundancy_guard_blocks_the_call_and_feeds_the_reason_back(wired):
    """The duplicate must NOT reach the tool, but it MUST reach the trace: the trace
    is what the next prompt shows the model, which is how it self-corrects."""
    fetch = FakeTool(ToolName.FETCH_PAGE)
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "read about page", url="https://acme.com/about"),
        act(ToolName.FETCH_PAGE, "read it again", url="http://www.acme.com/about/"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(fetch))

    result = loop_mod.run_agent("acme", make_settings())

    assert len(fetch.calls) == 1, "the duplicate fetch must not be executed"
    dup = result.trace[1]
    assert dup.ok is False
    assert dup.error == "redundant"
    assert "already" in dup.observation.lower()
    assert "step 1" in dup.observation
    # a blocked duplicate spends a step but not a fetch
    assert result.stats.fetches == 1


def test_tool_exception_becomes_a_failed_step_and_the_loop_continues(wired):
    boom = FakeTool(ToolName.FETCH_PAGE, raises=RuntimeError("connection reset"))
    search = FakeTool(ToolName.WEB_SEARCH)
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "try the site", url="https://acme.com"),
        act(ToolName.WEB_SEARCH, "fall back to search", query="acme corp profile"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(boom, search))

    result = loop_mod.run_agent("acme", make_settings())

    assert result.trace[0].ok is False
    assert "connection reset" in (result.trace[0].error or "")
    assert result.trace[1].ok is True and result.trace[1].tool == ToolName.WEB_SEARCH
    assert result.stats.stop_reason == "complete: agent called finish"


def test_unknown_tool_is_reported_not_crashed(wired):
    llm = FakeLLM([
        act(ToolName.DETECT_TECH, "check the stack", url="https://acme.com"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(FakeTool(ToolName.WEB_SEARCH)))

    result = loop_mod.run_agent("acme", make_settings())

    assert result.trace[0].ok is False
    assert result.trace[0].error == "unknown tool"
    # nothing was ever learned, so the run is blocked rather than complete
    assert stop_class(result.stats.stop_reason) == "blocked"


def test_detect_tech_is_allowed_right_after_fetching_the_same_url(wired):
    """Bug 1 end to end: the correct fetch-then-detect sequence used to be blocked,
    so every run finished with an empty tech_stack."""
    fetch = FakeTool(ToolName.FETCH_PAGE)
    tech = FakeTool(ToolName.DETECT_TECH, observations=[
        Observation(
            tool=ToolName.DETECT_TECH, ok=True, summary="1 technology",
            data={"url": "https://acme.com/about", "tech": [{"category": "hosting", "name": "Vercel"}]},
            sources=["https://acme.com/about"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "read the page", url="https://acme.com/about"),
        act(ToolName.DETECT_TECH, "now inspect its stack", url="https://acme.com/about"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(fetch, tech))

    result = loop_mod.run_agent("acme", make_settings())

    assert len(tech.calls) == 1, "detect_tech must run after fetch_page on the same URL"
    assert result.trace[1].ok is True
    assert [t.name for t in result.record.tech_stack] == ["Vercel"]


def test_loop_refuses_to_finish_having_learned_nothing(wired):
    """Bug 3: two rate-limited searches is a transient block, not a conclusion.

    The model asks to finish; the loop pushes back with the actual error until it
    tries something else (or runs out of pushbacks).
    """
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(tool=ToolName.WEB_SEARCH, ok=False,
                    summary="search failed: rate limited", error="429 rate limited"),
    ])
    fetch = FakeTool(ToolName.FETCH_PAGE)
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="acme corp overview"),
        act(ToolName.FINISH, "giving up", reason="search is broken"),
        act(ToolName.FETCH_PAGE, "try the domain directly", url="https://acme.com"),
        act(ToolName.FINISH, "now I have something", reason="ok"),
    ])
    wired(llm, registry_of(search, fetch))

    result = loop_mod.run_agent("acme.com", make_settings())

    pushback = result.trace[1]
    assert pushback.ok is False and pushback.error == "premature finish"
    assert "no research step has succeeded yet" in pushback.observation
    assert "429 rate limited" in pushback.observation, "the specific error must reach the model"
    # the failed search carries a retry nudge, not just the error
    assert "transient" in result.trace[0].observation
    # and the agent went on to learn something, so the run reads as complete
    assert len(fetch.calls) == 1
    assert result.stats.stop_reason == "complete: agent called finish"


def test_pushback_is_bounded_so_a_stuck_agent_can_still_stop(wired):
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(tool=ToolName.WEB_SEARCH, ok=False, summary="failed", error="429"),
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="acme corp overview"),
        act(ToolName.FINISH, "giving up", reason="blocked"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme.com", make_settings(max_steps=12))

    pushbacks = [s for s in result.trace if s.error == "premature finish"]
    assert len(pushbacks) == 2, "bounded by MAX_FINISH_PUSHBACKS"
    assert stop_class(result.stats.stop_reason) == "blocked"


def test_no_evidence_means_no_score_is_invented(wired):
    """Bug 2 at the source: synthesis is skipped, not faked, when nothing was learned."""
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(tool=ToolName.WEB_SEARCH, ok=False, summary="failed", error="429 rate limited"),
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="acme corp overview"),
        act(ToolName.FINISH, "giving up", reason="blocked"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme.com", make_settings(max_steps=8))

    assert result.record.icp_fit is None, "never manufacture a score from zero evidence"
    assert result.record.hooks == []
    assert result.record.unknowns
    assert stop_class(result.stats.stop_reason) == "blocked"
    skipped = [s for s in result.trace if s.error == "no evidence"]
    assert len(skipped) == 1
    assert "would invent a result" in skipped[0].observation


def test_partial_evidence_still_gets_scored(wired):
    """The mirror image: one successful step is enough to justify synthesis."""
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(tool=ToolName.WEB_SEARCH, ok=False, summary="failed", error="429"),
        Observation(
            tool=ToolName.WEB_SEARCH, ok=True, summary="1 hit",
            data={"results": [{"title": "Acme", "url": "https://acme.com", "snippet": "payments"}]},
            sources=["https://acme.com"],
        ),
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "first try", query="acme corp overview"),
        act(ToolName.WEB_SEARCH, "different angle", query="acme payments platform berlin"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme.com", make_settings())

    assert result.record.icp_fit is not None
    assert stop_class(result.stats.stop_reason) == "complete"


def test_learned_anything_ignores_synthesis_steps():
    assert not learned_anything([TraceStep(n=1, tool=ToolName.FINISH, ok=True)])
    assert learned_anything([TraceStep(n=1, tool=ToolName.WEB_SEARCH, ok=True)])
    assert not learned_anything([TraceStep(n=1, tool=ToolName.WEB_SEARCH, ok=False)])


def test_premature_finish_allows_finishing_when_out_of_budget():
    """Pushing back is pointless with no budget to act on the advice."""
    spent = Budget(max_steps=1, max_searches=0, max_fetches=0, max_jobs=0, total_timeout_s=60)
    history = [TraceStep(n=1, tool=ToolName.WEB_SEARCH, ok=False)]
    assert premature_finish_reason(spent, history) is None

    fresh = Budget(max_steps=9, max_searches=3, max_fetches=3, total_timeout_s=60)
    assert premature_finish_reason(fresh, history) is not None
    # once something has been learned, finishing is always the agent's call
    assert premature_finish_reason(fresh, [TraceStep(n=1, tool=ToolName.WEB_SEARCH, ok=True)]) is None


def test_stats_add_up(wired):
    search = FakeTool(ToolName.WEB_SEARCH)
    fetch = FakeTool(ToolName.FETCH_PAGE)
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "s", query="acme corp website"),
        act(ToolName.FETCH_PAGE, "f", url="https://acme.com"),
        act(ToolName.FETCH_PAGE, "f2", url="https://acme.com/about"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search, fetch))

    result = loop_mod.run_agent("acme", make_settings())
    s = result.stats

    assert s.steps == len(result.trace)
    assert s.searches == 1 == len(search.calls)
    assert s.fetches == 2 == len(fetch.calls)
    assert s.tokens_in == sum(step.tokens_in for step in result.trace) == 4 * 100
    assert s.tokens_out == sum(step.tokens_out for step in result.trace) == 4 * 20
    assert s.wall_ms >= 0
    assert s.llm_provider == "fake"
    assert s.search_provider == "fixture"


def test_evidence_accumulates_and_reaches_extraction(wired):
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(
            tool=ToolName.WEB_SEARCH, ok=True, summary="3 hits",
            data={"results": [{"title": "Acme", "url": "https://acme.com", "snippet": "payments"}]},
            sources=["https://acme.com"],
        )
    ])
    fetch = FakeTool(ToolName.FETCH_PAGE, observations=[
        Observation(
            tool=ToolName.FETCH_PAGE, ok=True, summary="about page",
            data={"text": "Acme builds payment rails.", "title": "Acme | Payments"},
            sources=["https://acme.com/about"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "s", query="acme corp website"),
        act(ToolName.FETCH_PAGE, "f", url="https://acme.com/about"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    state = wired(llm, registry_of(search, fetch))

    result = loop_mod.run_agent("acme", make_settings())

    evidence = state["evidence"]
    assert len(evidence) == 2
    assert any("payment rails" in text for _, text in evidence)
    assert "https://acme.com/about" in result.record.sources


def test_detect_tech_payload_lands_on_the_record(wired):
    """Locks the payload shape lea.tools.tech_detect actually emits:
    data={"url": ..., "tech": [TechStack.model_dump(), ...]}."""
    from lea.models import TechStack

    detected = [
        TechStack(category="analytics", name="Segment", evidence="script tag", confidence=0.9),
        TechStack(category="hosting", name="Cloudflare"),
    ]
    tech = FakeTool(ToolName.DETECT_TECH, observations=[
        Observation(
            tool=ToolName.DETECT_TECH, ok=True, summary="2 technologies on acme.com",
            data={"url": "https://acme.com", "tech": [t.model_dump() for t in detected]},
            sources=["https://acme.com"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.DETECT_TECH, "check the stack", url="https://acme.com"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(tech))

    result = loop_mod.run_agent("acme.com", make_settings())

    names = {t.name for t in result.record.tech_stack}
    assert names == {"Segment", "Cloudflare"}
    # target was a domain, so the domain IS authoritative and stays seeded
    assert result.record.domain == "acme.com"
    # detect_tech spends the fetch budget, not the search budget
    assert result.stats.fetches == 1 and result.stats.searches == 0
    assert "tech" not in derive_gaps(result.record)


def test_record_updates_narrow_the_gap_list_during_the_run(wired):
    fetch = FakeTool(ToolName.FETCH_PAGE, observations=[
        Observation(
            tool=ToolName.FETCH_PAGE, ok=True, summary="home",
            data={"text": "welcome", "title": "Acme | Payments for teams"},
            sources=["https://acme.com"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "f", url="https://acme.com"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    # base record passed straight through so we can see exactly what the loop set
    state = wired(llm, registry_of(fetch), extract=None)

    result = loop_mod.run_agent("Acme Corp", make_settings())

    # The inferred domain steers the NEXT step: the model is told to stop hunting
    # for it, so the gap list shrinks during the run...
    seen = state["steering"]
    assert seen and seen[-1].domain == "acme.com"
    assert "domain" not in derive_gaps(seen[-1])

    # ...but it is a guess, so it never lands on the record, where it would win the
    # merge and pre-empt choose_domain.
    assert result.record.domain is None
    # and the page title never becomes the name — the user's word does
    assert result.record.company_name == "Acme Corp"


def test_keyword_stuffed_page_title_never_becomes_the_company_name(monkeypatch, tmp_path):
    """Regression for the Vanta bug.

    The loop used to set company_name from a page <title>. Because merge_records
    lets the base win any scalar it already has, that title then beat Agent D's
    sanitize_company_name and poisoned the header, every hook, AND the filename:

        company_name "SOC 2, HIPAA, ISO 27001, PCI, and GDPR Compliance"
        out file     out/soc-2-hipaa-iso-27001-pci-and-gdpr-compliance.json

    This runs the REAL extraction + merge, only faking the tools and the brain,
    because the bug lived in the composition of the two, not in either one.
    """
    stuffed = "SOC 2, HIPAA, ISO 27001, PCI, and GDPR Compliance | Vanta"
    fetch = FakeTool(ToolName.FETCH_PAGE, observations=[
        Observation(
            tool=ToolName.FETCH_PAGE, ok=True, summary=stuffed,
            data={
                "url": "https://www.vanta.com/",
                "title": stuffed,
                "text": "Vanta automates compliance. " + stuffed,
            },
            sources=["https://www.vanta.com/"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "read the homepage", url="https://www.vanta.com/"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    monkeypatch.setattr(loop_mod, "_get_llm", lambda s: llm)
    monkeypatch.setattr(loop_mod, "_get_registry", lambda s: registry_of(fetch))
    monkeypatch.setattr(loop_mod, "_get_prompts", lambda: ("SYSTEM", lambda *a, **k: "CONTEXT"))
    # _call_extract / _call_icp / _call_hooks are NOT patched: real extraction runs.

    result = loop_mod.run_agent("Vanta", make_settings(llm_provider="stub"))

    name = result.record.company_name
    assert name == "Vanta", f"page title leaked into company_name: {name!r}"
    assert "SOC 2" not in (name or "")
    assert "|" not in (name or "")

    # ...and nothing downstream is named after the title either
    from lea.cli import output_slug

    slug = output_slug(result.record, "Vanta")
    assert slug == "vanta", f"filename derived from the title: {slug!r}"

    from lea import trace as trace_mod

    path = trace_mod.save_json(result, tmp_path / f"{slug}.json")
    assert path.name == "vanta.json"

    for hook in result.record.hooks:
        assert "SOC 2, HIPAA" not in hook.hook


def test_domain_target_lets_extraction_name_the_company(monkeypatch):
    """The mirror case: 'ramp.com' must NOT be asserted as the company name."""
    fetch = FakeTool(ToolName.FETCH_PAGE, observations=[
        Observation(
            tool=ToolName.FETCH_PAGE, ok=True, summary="home",
            data={
                "url": "https://ramp.com/",
                "title": "About Us - Different By Design",
                "text": "Ramp is the corporate card and spend management platform.",
            },
            sources=["https://ramp.com/"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "read it", url="https://ramp.com/"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    monkeypatch.setattr(loop_mod, "_get_llm", lambda s: llm)
    monkeypatch.setattr(loop_mod, "_get_registry", lambda s: registry_of(fetch))
    monkeypatch.setattr(loop_mod, "_get_prompts", lambda: ("SYSTEM", lambda *a, **k: "CONTEXT"))

    result = loop_mod.run_agent("ramp.com", make_settings(llm_provider="stub"))

    assert result.record.domain == "ramp.com"          # the user pointed at this
    assert result.record.company_name != "ramp.com"    # but that is not a NAME
    assert "About Us" not in (result.record.company_name or "")
    assert "Different By Design" not in (result.record.company_name or "")


def test_output_slug_rules():
    from lea.cli import output_slug

    rec = CompanyRecord(input="x", company_name="SOC 2, HIPAA, ISO 27001, PCI, and GDPR Compliance")
    # a plain-name target always names its own file, whatever extraction produced
    assert output_slug(rec, "Vanta") == "vanta"
    # a domain target falls back to the registrable label when the name is junk
    assert output_slug(rec, "ramp.com") == "ramp"
    assert output_slug(rec, "https://www.ramp.com/about") == "ramp"
    # ...but uses a clean resolved name when there is one
    assert output_slug(CompanyRecord(input="x", company_name="Ramp"), "ramp.com") == "ramp"
    assert output_slug(CompanyRecord(input="x"), "acme.co.uk") == "acme"
    assert output_slug(CompanyRecord(input="x"), "https://www.acme.co.uk/team") == "acme"


def test_extraction_failure_does_not_lose_the_run(wired):
    fetch = FakeTool(ToolName.FETCH_PAGE)
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "f", url="https://acme.com"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(fetch), extract_raises=ValueError("bad json"))

    result = loop_mod.run_agent("acme.com", make_settings())

    failed = [s for s in result.trace if not s.ok]
    assert any("extraction failed" in s.observation for s in failed)
    assert result.record.domain == "acme.com"   # seeded from the input, still there
    assert result.record.icp_fit is not None    # later stages still ran
    assert result.record.unknowns


def test_missing_registry_module_does_not_crash_the_run(monkeypatch):
    def boom(settings):
        raise ImportError("no module named lea.tools.registry")

    monkeypatch.setattr(loop_mod, "_get_registry", boom)
    monkeypatch.setattr(loop_mod, "_default_icp", lambda s: DEFAULT_ICP)
    monkeypatch.setattr(loop_mod, "_call_icp", lambda *a: ICPFit(score=1.0, tier="D"))
    monkeypatch.setattr(loop_mod, "_call_hooks", lambda *a: [])

    result = loop_mod.run_agent("acme", make_settings())

    assert result.stats.stop_reason.startswith("error: tool registry unavailable")
    assert result.record.input == "acme"


def test_run_batch_isolates_failures(monkeypatch):
    calls: list[str] = []

    def fake_run(target, settings, icp=None, on_step=None):
        calls.append(target)
        if target == "bad":
            raise RuntimeError("kaboom")
        from lea.models import EnrichmentResult

        return EnrichmentResult(record=CompanyRecord(input=target))

    monkeypatch.setattr(loop_mod, "run_agent", fake_run)

    seen: list[str] = []
    results = loop_mod.run_batch(
        ["a", "bad", "c"], make_settings(), on_result=lambda r: seen.append(r.record.input)
    )

    assert calls == ["a", "bad", "c"]
    assert [r.record.input for r in results] == ["a", "bad", "c"]
    assert "kaboom" in results[1].stats.stop_reason
    assert seen == ["a", "bad", "c"]


# --------------------------------------------------------------------------
# Provider capability + stale model diagnosis
# --------------------------------------------------------------------------


class FakeCompatLLM(FakeLLM):
    """Stands in for OpenAICompatClient: has .capability, and reports a failed call
    by RETURNING a FINISH whose thought starts 'LLM error:' rather than raising."""

    def __init__(self, script, *, name="google", model="gemini-2.0-flash", capability="json_object"):
        super().__init__(script)
        self.name = name
        self.model = model
        self.capability = capability


def llm_error(detail: str) -> AgentAction:
    return AgentAction(tool=ToolName.FINISH, thought=f"LLM error: {detail}")


def test_provider_label_carries_the_capability_rung():
    s = make_settings()
    assert loop_mod.provider_label(FakeCompatLLM([], capability="tools"), s) == \
        "google:gemini-2.0-flash (tools)"
    assert loop_mod.provider_label(FakeCompatLLM([], capability="prose"), s) == \
        "google:gemini-2.0-flash (prose)"
    # anthropic/stub/fakes have no .capability and must be untouched
    assert loop_mod.provider_label(FakeLLM([]), s) == "fake"

    class Bare:
        pass

    assert loop_mod.provider_label(Bare(), s) == s.resolved_llm()


def test_capability_rung_reaches_the_stats(wired):
    search = FakeTool(ToolName.WEB_SEARCH)
    llm = FakeCompatLLM([
        act(ToolName.WEB_SEARCH, "s", query="acme corp overview"),
        act(ToolName.FINISH, "done", reason="ok"),
    ], capability="prose")
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme", make_settings())
    assert result.stats.llm_provider == "google:gemini-2.0-flash (prose)"


def test_stale_model_id_is_named_as_the_cause(wired):
    """A 404ing model id used to look like an agent that calmly decided it was done."""
    detail = "HTTPStatusError: HTTP 404: model 'gemini-1.0-pro' not found"
    llm = FakeCompatLLM([llm_error(detail)])
    wired(llm, registry_of(FakeTool(ToolName.WEB_SEARCH)))

    result = loop_mod.run_agent("acme", make_settings(max_steps=8))

    assert stop_class(result.stats.stop_reason) == "error"
    reason = result.stats.stop_reason
    assert "404" in reason
    assert "stale" in reason.lower()
    assert "lea doctor --models" in reason
    assert "LEA_MODEL" in reason

    # the failed calls are visible as failed steps, not as a tidy finish
    failed = [s for s in result.trace if s.error and "404" in s.error]
    assert failed, "the LLM error must appear in the trace"
    assert all(not s.ok for s in failed)
    assert "lea doctor --models" in failed[0].observation
    # and no score is invented on top of it
    assert result.record.icp_fit is None


def test_non_stale_llm_failure_still_names_the_llm(wired):
    llm = FakeCompatLLM([llm_error("ConnectError: connection refused")])
    wired(llm, registry_of(FakeTool(ToolName.WEB_SEARCH)))

    result = loop_mod.run_agent("acme", make_settings(max_steps=8))

    assert stop_class(result.stats.stop_reason) == "error"
    assert "every LLM call" in result.stats.stop_reason
    assert "connection refused" in result.stats.stop_reason
    # not misdiagnosed as a stale model
    assert "stale" not in result.stats.stop_reason.lower()


def test_an_llm_error_does_not_mask_a_successful_run(wired):
    """One transient LLM error then real work: the run is complete, not an error."""
    search = FakeTool(ToolName.WEB_SEARCH)
    llm = FakeCompatLLM([
        llm_error("HTTP 429: rate limited"),
        act(ToolName.WEB_SEARCH, "retry", query="acme corp overview"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme", make_settings())

    assert stop_class(result.stats.stop_reason) == "complete"
    assert result.record.icp_fit is not None


def test_looks_like_stale_model_discriminates():
    from lea.agent.policy import looks_like_stale_model

    assert looks_like_stale_model("HTTP 404: model not found")
    assert looks_like_stale_model("invalid model id")
    assert looks_like_stale_model("model_not_found")
    assert not looks_like_stale_model("connection refused")
    assert not looks_like_stale_model("HTTP 500: server error")


# --------------------------------------------------------------------------
# Truncation: a cut-off error is one the user cannot search for
# --------------------------------------------------------------------------


def test_truncate_words_cuts_on_a_word_boundary():
    from lea.agent.policy import truncate_words

    text = "authentication rejected by google: Request had invalid authentication credentials"

    out = truncate_words(text, 60)
    assert len(out) <= 60
    assert out.endswith("…")
    # the defect: never end mid-token
    assert not out.rstrip("…").endswith("authenticati")
    assert out.rstrip("…").split()[-1] in text.split()

    # short enough is returned untouched, with no ellipsis
    assert truncate_words(text, 500) == text
    assert truncate_words("short", 50) == "short"
    assert truncate_words("", 50) == ""


def test_truncate_words_falls_back_to_a_hard_cut_for_one_long_token():
    from lea.agent.policy import truncate_words

    url = "https://" + "a" * 300
    out = truncate_words(url, 40)
    assert len(out) <= 40
    assert out.endswith("…")


def test_llm_error_stop_reason_is_not_cut_mid_word(wired):
    """Regression: the stop_reason trimmed B's already-200-char message again at
    140 and landed mid-token ('...invalid authenticati') — the one string a user
    would have pasted into a search box verbatim."""
    detail = (
        "LLM error: authentication rejected by google — the API key looks wrong or "
        "lacks access to gemini-3.6-flash: Request had invalid authentication credentials "
        "expected OAuth 2 access token"
    )
    llm = FakeCompatLLM([llm_error(detail)])
    wired(llm, registry_of(FakeTool(ToolName.WEB_SEARCH)))

    reason = loop_mod.run_agent("Vanta", make_settings(max_steps=8)).stats.stop_reason

    assert "authenticati)" not in reason
    assert "authenticati…" not in reason
    # it WAS long enough to trim, and the trim landed between words
    assert "…" in reason
    last_word = reason.split("…")[0].split()[-1]
    assert last_word in detail.split(), f"{last_word!r} is not a whole word from the error"


def test_observation_truncation_keeps_whole_words(wired):
    long_error = "connection reset by peer while contacting " + "example-host-name " * 40
    boom = FakeTool(ToolName.FETCH_PAGE, raises=RuntimeError(long_error))
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "try it", url="https://acme.com"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(boom))

    step = loop_mod.run_agent("acme.com", make_settings()).trace[0]
    shown = step.observation.rstrip("…").strip()
    assert shown
    assert not shown.endswith("example-host-nam")


def test_auth_failure_is_not_misdiagnosed_as_a_stale_model(wired):
    """'HTTP 400: Please pass a valid API key' contains '400', but the user's
    problem is the key — pointing them at --models wastes their time."""
    from lea.agent.policy import looks_like_auth_failure, looks_like_stale_model

    msg = "LLM error: HTTP 400: Please pass a valid API key"
    assert looks_like_auth_failure(msg)
    assert not looks_like_stale_model(msg)

    llm = FakeCompatLLM([llm_error(msg)])
    wired(llm, registry_of(FakeTool(ToolName.WEB_SEARCH)))
    reason = loop_mod.run_agent("Vanta", make_settings(max_steps=8)).stats.stop_reason

    assert "authentication problem" in reason
    assert "check the key" in reason
    assert "stale" not in reason.lower()
    assert "--models" not in reason

    # a genuine model-not-found still routes to --models
    assert looks_like_stale_model("HTTP 404: model 'x' does not exist")


# --------------------------------------------------------------------------
# Search/fetch balance: the "searched 4 times, read 2 pages" failure
# --------------------------------------------------------------------------


def _search_step(n: int, query: str, *urls: str) -> TraceStep:
    return TraceStep(n=n, tool=ToolName.WEB_SEARCH, args={"query": query}, ok=True,
                     sources=list(urls))


def _fetch_step(n: int, url: str, ok: bool = True) -> TraceStep:
    return TraceStep(n=n, tool=ToolName.FETCH_PAGE, args={"url": url}, ok=ok)


def test_unread_urls_tracks_what_search_surfaced_but_nobody_opened():
    from lea.agent.policy import fetched_urls, seen_urls, unread_urls

    history = [
        _search_step(1, "vanta overview", "https://vanta.com", "https://techcrunch.com/vanta-c"),
        _fetch_step(2, "https://www.vanta.com/"),          # www + slash: same page
        _search_step(3, "vanta funding", "https://techcrunch.com/vanta-c", "https://crunchbase.com/vanta"),
    ]

    assert seen_urls(history) == [
        "https://vanta.com", "https://techcrunch.com/vanta-c", "https://crunchbase.com/vanta",
    ]
    assert "vanta.com" in fetched_urls(history)
    # the fetched one drops out despite the www/trailing-slash difference
    assert unread_urls(history) == [
        "https://techcrunch.com/vanta-c", "https://crunchbase.com/vanta",
    ]


def test_budget_used_reports_the_balance():
    b = Budget(max_steps=12, max_searches=5, max_fetches=6, total_timeout_s=60)
    for _ in range(4):
        b.consume(ToolName.WEB_SEARCH)
    for _ in range(2):
        b.consume(ToolName.FETCH_PAGE)
    used = b.used()
    assert used["searches_used"] == "4/5"
    assert used["fetches_used"] == "2/6"
    assert used["steps_used"] == "6/12"


def test_budget_view_carries_spend_without_duplicating_the_prompt():
    b = Budget(max_steps=12, max_searches=5, max_fetches=6, total_timeout_s=60)
    b.consume(ToolName.WEB_SEARCH)
    trace = [_search_step(1, "q", "https://a.example", "https://b.example")]

    view = loop_mod.budget_view(b, trace)

    # remaining keys survive for whoever already reads them
    assert view["searches"] == 4 and view["fetches"] == 6
    # spend comes from the Budget, not the trace: a guard bounce is a step that
    # spends no search, so the trace alone would miscount
    assert view["searches_used"] == "1/5"
    assert view["fetches_used"] == "0/6"
    # prompts.py derives and formats the unread set from the trace itself; passing
    # it here too put a raw list repr in every prompt beside that block
    assert "unread_urls" not in view


def test_budget_view_reaches_the_prompt(wired):
    """The spend numbers must actually arrive in build_agent_context, and must
    climb as the run proceeds — a static 0/4 would steer nothing."""
    search = FakeTool(ToolName.WEB_SEARCH)
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "s", query="acme corp overview"),
        act(ToolName.FETCH_PAGE, "f", url="https://acme.com/about"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    state = wired(llm, registry_of(search, FakeTool(ToolName.FETCH_PAGE)))

    loop_mod.run_agent("acme", make_settings())

    seen = state["budgets"]
    assert seen, "build_agent_context was never called"
    assert seen[0]["searches_used"] == "0/4"
    assert seen[1]["searches_used"] == "1/4"
    assert seen[2]["fetches_used"] == "1/4"


def test_search_guard_fires_on_back_to_back_searches_with_unread_urls():
    from lea.agent.policy import search_without_reading_reason

    history = [_search_step(1, "vanta overview", "https://techcrunch.com/vanta-series-c")]
    reason = search_without_reading_reason(act(ToolName.WEB_SEARCH, query="vanta hq"), history)

    assert reason
    assert "techcrunch.com/vanta-series-c" in reason
    assert "unread" in reason
    assert "headquarters, founding year or headcount" in reason


def test_search_guard_stays_quiet_when_it_would_be_wrong():
    from lea.agent.policy import MAX_SEARCH_PUSHBACKS, search_without_reading_reason

    search_again = act(ToolName.WEB_SEARCH, query="vanta hq")

    # a second search after one that returned NOTHING is exactly right
    assert search_without_reading_reason(search_again, [_search_step(1, "q")]) is None

    # ...as is searching after actually reading the hit
    read_it = [
        _search_step(1, "q", "https://vanta.com"),
        _fetch_step(2, "https://vanta.com"),
    ]
    assert search_without_reading_reason(search_again, read_it) is None

    unread = [_search_step(1, "q", "https://vanta.com")]
    # ...and fetching is not being second-guessed
    assert search_without_reading_reason(act(ToolName.FETCH_PAGE, url="https://x.example"), unread) is None
    # ...nor is a search when no fetch budget remains to act on the advice
    assert search_without_reading_reason(search_again, unread, can_fetch=False) is None
    # ...and it is bounded, so an insistent model still gets through
    assert search_without_reading_reason(search_again, unread, pushbacks=MAX_SEARCH_PUSHBACKS) is None


def test_loop_pushes_back_on_searching_instead_of_reading(wired):
    """The live failure: search, read snippets, search again, never open the page."""
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(
            tool=ToolName.WEB_SEARCH, ok=True, summary="3 hits",
            data={"results": [{"title": "Vanta raises $150M", "url": "https://tc.example/vanta"}]},
            sources=["https://tc.example/vanta"],
        ),
    ])
    fetch = FakeTool(ToolName.FETCH_PAGE)
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="vanta company overview"),
        act(ToolName.WEB_SEARCH, "search again", query="vanta headquarters founded"),
        act(ToolName.FETCH_PAGE, "fine, read it", url="https://tc.example/vanta"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search, fetch))

    result = loop_mod.run_agent("Vanta", make_settings())

    bounced = result.trace[1]
    assert bounced.ok is False and bounced.error == "unread sources"
    assert "https://tc.example/vanta" in bounced.observation
    # the second search never executed, and the page did get read
    assert len(search.calls) == 1
    assert len(fetch.calls) == 1
    # a guard bounce costs a step but not a search
    assert result.stats.searches == 1
    assert stop_class(result.stats.stop_reason) == "complete"


def test_search_guard_is_bounded_so_an_insistent_model_proceeds(wired):
    from lea.agent.policy import MAX_SEARCH_PUSHBACKS

    # sources must plausibly relate to the target, or the guard correctly stays
    # silent (that is the junk-results case, covered separately)
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(
            tool=ToolName.WEB_SEARCH, ok=True, summary=f"hit {i}",
            data={"results": [{"title": "Vanta", "url": f"https://vanta.com/p{i}", "snippet": "x"}]},
            sources=[f"https://vanta.com/p{i}"],
        )
        for i in range(6)
    ])
    # keeps asking to search, never takes the hint
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "a", query="vanta company overview"),
        act(ToolName.WEB_SEARCH, "b", query="vanta headquarters founded year"),
        act(ToolName.WEB_SEARCH, "c", query="vanta employee headcount linkedin"),
        act(ToolName.WEB_SEARCH, "d", query="vanta pricing customers reviews"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search, FakeTool(ToolName.FETCH_PAGE)))

    result = loop_mod.run_agent("Vanta", make_settings(max_steps=12, max_searches=6))

    bounces = [s for s in result.trace if s.error == "unread sources"]
    assert len(bounces) == MAX_SEARCH_PUSHBACKS, "the guard must stop nagging"
    # it was never hard-blocked: once the bound is hit, searches go through again
    assert len(search.calls) >= 2
    assert stop_class(result.stats.stop_reason) == "complete"


def test_stall_is_extended_while_a_rescue_is_available():
    """The reported cut-off: 3 dead steps ended the run with 4 fetches of budget
    left and an unopened URL sitting in its own search results."""
    from lea.agent.policy import MAX_CONSECUTIVE_DEAD_STEPS, STALL_GRACE_WITH_UNREAD

    rec = CompanyRecord(input="Vanta")
    budget = Budget(max_steps=12, max_searches=5, max_fetches=6, total_timeout_s=60)

    found = _search_step(1, "vanta", "https://tc.example/vanta")
    dead = [TraceStep(n=i, tool=ToolName.FETCH_PAGE, ok=False, error="403")
            for i in range(2, 2 + MAX_CONSECUTIVE_DEAD_STEPS)]

    # unread URL + fetch budget -> keep going, the redirect can still rescue it
    assert should_stop(rec, budget, [found, *dead]) is None

    # but the grace is bounded
    more = dead + [TraceStep(n=90 + i, tool=ToolName.FETCH_PAGE, ok=False, error="403")
                   for i in range(STALL_GRACE_WITH_UNREAD)]
    assert should_stop(rec, budget, [found, *more]) is not None

    # with nothing left to read, the original threshold stands
    nothing_found = _search_step(1, "vanta")
    assert should_stop(rec, budget, [nothing_found, *dead]) is not None

    # ...and so it does when there is no fetch budget to rescue with
    spent = Budget(max_steps=12, max_searches=5, max_fetches=0, total_timeout_s=60)
    assert should_stop(rec, spent, [found, *dead]) is not None


def test_rate_limit_is_not_reported_as_a_broken_setup(wired):
    """Free tiers 429 constantly; 'every LLM call failed' reads like a broken key."""
    from lea.agent.policy import looks_like_auth_failure, looks_like_rate_limit

    msg = ("LLM error: HTTP 429: You exceeded your current quota, please check your "
           "plan and billing details.")
    assert looks_like_rate_limit(msg)
    assert not looks_like_auth_failure(msg)  # "billing" must not read as auth

    llm = FakeCompatLLM([llm_error(msg)])
    wired(llm, registry_of(FakeTool(ToolName.WEB_SEARCH)))
    reason = loop_mod.run_agent("Vanta", make_settings(max_steps=8)).stats.stop_reason

    assert "quota problem" in reason
    assert "nothing is wrong with your setup" in reason
    assert "authentication" not in reason


# --------------------------------------------------------------------------
# End-to-end on fixtures (no network, no LLM: the deterministic stub brain)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target, tier, score, industry",
    [
        # The keyless baseline. These move whenever the agent researches differently,
        # which is the point of the test -- so the value is re-derived, never adjusted
        # to make a diff pass. Both are deterministic across repeated runs.
        #
        # Current values date from the stub planner learning to call fetch_jobs: reading
        # the ATS board grades company_size on real evidence instead of a default, which
        # lifts confidence to 0.95 and carries Ramp from B to A on 148 open roles, 63 of
        # them in sales. Earlier movements came from signal recency weighting and from
        # replacing an invented TechCrunch fixture with the real article.
        ("Vanta", "A", 90.6, "Security & Compliance"),
        # 63.9 -> 68.4 (Agent D): the real TechCrunch headline ("nearly doubles
        # valuation to $13B in secondary share sale") has no raise verb and was
        # being filed as a `product` signal. It is now correctly `funding`,
        # discounted 0.8x because a secondary sale brings no new primary
        # capital. Verified stable across 6 consecutive runs.
        ("Ramp", "A", 81.7, "Fintech"),
    ],
)
def test_stub_pipeline_on_fixtures_does_not_regress(target, tier, score, industry):
    """The keyless baseline every change is measured against.

    Real tools, real extraction, real ICP scoring — only the brain is the stub and
    every HTTP call is served from examples/fixtures. If a policy change makes the
    agent research differently, this is what notices.
    """
    settings = Settings(llm_provider="stub", search_provider="fixture", offline=True)
    result = loop_mod.run_agent(target, settings)

    rec = result.record
    assert rec.icp_fit is not None, "the stub path must always produce a score"
    assert rec.icp_fit.tier == tier
    assert rec.icp_fit.score == pytest.approx(score, abs=0.05)
    assert rec.industry == industry
    assert rec.signals and rec.tech_stack
    assert stop_class(result.stats.stop_reason) == "complete"
    # and it stays cheap: no runaway searching
    assert result.stats.searches <= settings.max_searches
    assert result.stats.fetches <= settings.max_fetches


# --------------------------------------------------------------------------
# fetch_jobs: a fourth tool with its own economics
# --------------------------------------------------------------------------


def test_board_identity_collapses_the_ways_to_name_one_board():
    from lea.agent.policy import board_identity

    same = [
        "vanta.com",
        "Vanta",
        "https://jobs.ashbyhq.com/vanta",
        "https://jobs.ashbyhq.com/vanta/",
        "http://www.vanta.com",
        "https://boards.greenhouse.io/vanta",
    ]
    keys = {board_identity(v) for v in same}
    assert keys == {"vanta"}, keys

    # ...without collapsing genuinely different companies
    assert board_identity("ramp.com") != board_identity("vanta.com")
    assert board_identity("https://jobs.lever.co/acme") == "acme"
    assert board_identity("Acme Corp") == "acme-corp"
    assert board_identity("") == ""


def test_redundant_job_board_lookup_is_caught_across_spellings():
    history = [TraceStep(n=2, tool=ToolName.FETCH_JOBS,
                         args={"company_or_url": "https://jobs.ashbyhq.com/vanta"}, ok=True,
                         sources=["https://jobs.ashbyhq.com/vanta"])]

    dup = is_redundant(act(ToolName.FETCH_JOBS, company_or_url="vanta.com"), history)
    assert dup and "already read vanta's job board" in dup

    # a different company is fine, and so is a different tool on the same URL
    assert is_redundant(act(ToolName.FETCH_JOBS, company_or_url="ramp.com"), history) is None
    assert is_redundant(
        act(ToolName.FETCH_PAGE, url="https://jobs.ashbyhq.com/vanta"), history
    ) is None


def test_job_board_lookup_marks_that_url_read():
    """Otherwise the unread-URL nudge keeps pointing at a board already read."""
    from lea.agent.policy import unread_urls

    history = [
        _search_step(1, "vanta careers", "https://jobs.ashbyhq.com/vanta", "https://vanta.com"),
        TraceStep(n=2, tool=ToolName.FETCH_JOBS,
                  args={"company_or_url": "https://jobs.ashbyhq.com/vanta"}, ok=True),
    ]
    assert unread_urls(history) == ["https://vanta.com"]


def test_job_board_lookup_counts_as_reading_for_the_search_guard():
    """search -> fetch_jobs -> search is a sane rhythm, not two searches in a row."""
    from lea.agent.policy import search_without_reading_reason

    history = [
        _search_step(1, "vanta", "https://vanta.com"),
        TraceStep(n=2, tool=ToolName.FETCH_JOBS, args={"company_or_url": "vanta.com"}, ok=True),
    ]
    assert search_without_reading_reason(
        act(ToolName.WEB_SEARCH, query="vanta funding round"), history
    ) is None


def test_allowed_tools_offers_fetch_jobs_only_while_affordable():
    registry = registry_of(
        FakeTool(ToolName.WEB_SEARCH), FakeTool(ToolName.FETCH_PAGE),
        FakeTool(ToolName.DETECT_TECH), FakeTool(ToolName.FETCH_JOBS),
    )
    b = Budget(max_steps=9, max_searches=2, max_fetches=2, max_jobs=1, total_timeout_s=60)
    assert "fetch_jobs" in {s["name"] for s in allowed_tools(b, registry)}

    b.consume(ToolName.FETCH_JOBS)
    names = {s["name"] for s in allowed_tools(b, registry)}
    assert "fetch_jobs" not in names, "spent allowance must not be offered"
    # the page budget is untouched by it
    assert "fetch_page" in names and "finish" in names


def test_loop_spends_the_jobs_allowance_not_the_page_budget(wired):
    jobs = FakeTool(ToolName.FETCH_JOBS, observations=[
        Observation(
            tool=ToolName.FETCH_JOBS, ok=True,
            summary="Vanta — Ashby — 93 open roles (Sales 21)",
            data={"company": "Vanta", "slug": "vanta", "open_roles": 93},
            sources=["https://jobs.ashbyhq.com/vanta"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.FETCH_JOBS, "count the open roles", company_or_url="vanta.com"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(jobs))

    result = loop_mod.run_agent("Vanta", make_settings())

    assert len(jobs.calls) == 1
    assert result.stats.fetches == 0, "a job-board call is not a page fetch"
    assert result.trace[0].ok is True
    # and it does not eat the search budget either
    assert result.stats.searches == 0


def test_loop_blocks_a_repeat_job_board_lookup(wired):
    jobs = FakeTool(ToolName.FETCH_JOBS)
    llm = FakeLLM([
        act(ToolName.FETCH_JOBS, "roles", company_or_url="https://jobs.ashbyhq.com/vanta"),
        act(ToolName.FETCH_JOBS, "again", company_or_url="Vanta"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(jobs))

    result = loop_mod.run_agent("Vanta", make_settings())

    assert len(jobs.calls) == 1, "the duplicate must not reach the tool"
    assert result.trace[1].error == "redundant"
    assert "job board" in result.trace[1].observation


def test_hiring_is_its_own_gap_so_the_agent_goes_and_gets_it():
    from lea.models import Signal

    rec = CompanyRecord(input="Vanta")
    assert "hiring" in derive_gaps(rec)
    assert "open roles" in " ".join(gap_labels(derive_gaps(rec)))

    # a funding signal is not a hiring signal
    rec.signals = [Signal(kind="funding", headline="Series C")]
    assert "signals" not in derive_gaps(rec)
    assert "hiring" in derive_gaps(rec)

    rec.signals.append(Signal(kind="hiring", headline="93 open roles"))
    assert "hiring" not in derive_gaps(rec)


def test_synthesis_overrun_is_reported_not_hidden(wired, monkeypatch):
    """A run reporting a 180s budget beside a 247s wall clock looks like it
    ignored its own limit."""
    search = FakeTool(ToolName.WEB_SEARCH)
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "s", query="vanta company overview"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search))

    # make extraction "slow" by advancing the clock the loop measures
    real = loop_mod.time.monotonic
    offset = {"v": 0.0}

    def fake_monotonic():
        return real() + offset["v"]

    monkeypatch.setattr(loop_mod.time, "monotonic", fake_monotonic)

    original = loop_mod._call_extract

    def slow_extract(*a, **k):
        offset["v"] += 200.0  # synthesis burns 200s of wall clock
        return original(*a, **k)

    monkeypatch.setattr(loop_mod, "_call_extract", slow_extract)

    result = loop_mod.run_agent("Vanta", make_settings(total_timeout_s=180.0))

    reason = result.stats.stop_reason
    assert "synthesis then ran a further" in reason
    assert "past the 180s budget" in reason
    assert result.stats.wall_ms / 1000 > 180


# --------------------------------------------------------------------------
# Malformed calls vs genuine provider failures
# --------------------------------------------------------------------------


def test_required_args_reads_both_schema_shapes():
    from lea.agent.policy import required_args

    # the flat shape this project's tools use
    flat = {"query": {"type": "string", "required": True},
            "num_results": {"type": "integer", "required": False}}
    assert required_args(flat) == ["query"]

    # JSON-schema shape
    js = {"properties": {"q": {"type": "string"}, "n": {"type": "integer"}}, "required": ["q"]}
    assert required_args(js) == ["q"]

    # unreadable/absent schemas let the call through rather than blocking it
    assert required_args(None) == []
    assert required_args({}) == []
    assert required_args({"query": "str"}) == []


def test_missing_required_args_treats_blank_as_missing():
    from lea.agent.policy import missing_required_args

    schema = {"query": {"required": True}}
    assert missing_required_args(schema, {}) == ["query"]
    assert missing_required_args(schema, None) == ["query"]
    assert missing_required_args(schema, {"query": "   "}) == ["query"]
    assert missing_required_args(schema, {"query": None}) == ["query"]
    assert missing_required_args(schema, {"query": "vanta"}) == []
    # an optional arg missing is not a violation
    assert missing_required_args({"reason": {"required": False}}, {}) == []


def test_malformed_call_is_caught_before_the_tool(wired):
    """The live Figma failure: web_search() with no query. It cannot succeed, so
    it must not cost a search credit or an HTTP round trip."""
    search = FakeTool(ToolName.WEB_SEARCH)
    search.args_schema = {"query": {"type": "string", "required": True}}
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "find them"),                      # no query!
        act(ToolName.WEB_SEARCH, "with a query", query="vanta compliance"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("Vanta", make_settings())

    bad = result.trace[0]
    assert bad.ok is False
    assert bad.error == "missing required argument(s): query"
    assert "query" in bad.observation
    assert "malformed call, not a provider problem" in bad.observation
    assert "retrying it unchanged will fail identically" in bad.observation
    # the misleading old wording must not appear on a malformed call
    assert "usually transient" not in bad.observation
    assert "rate limit" not in bad.observation

    # it never reached the tool, and cost no search budget
    assert len(search.calls) == 1, "only the well-formed call should execute"
    assert search.calls[0] == {"query": "vanta compliance"}
    assert result.stats.searches == 1
    # reported, never repaired: the loop did not invent a query from the target
    assert bad.args == {}


def test_genuine_provider_failure_keeps_the_transient_wording():
    """The distinction has to cut both ways."""
    blocked = TraceStep(
        n=1, tool=ToolName.WEB_SEARCH, args={"query": "vanta"}, ok=False,
        observation="search failed: duckduckgo returned no parseable results",
        error="duckduckgo returned no parseable results (rate limited?)",
    )
    hint = transient_failure_hint(blocked)
    assert "usually transient" in hint
    assert "malformed" not in hint

    malformed = TraceStep(
        n=1, tool=ToolName.WEB_SEARCH, args={}, ok=False,
        observation="search failed: missing required arg 'query'",
        error="missing required arg 'query'",
    )
    hint = transient_failure_hint(malformed)
    assert "malformed call" in hint
    assert "usually transient" not in hint
    assert "rate limit" not in hint


def test_tool_reported_bad_args_also_get_the_right_hint(wired):
    """A tool that validates its own args (rather than the loop catching it first)
    must still produce the malformed-call wording, not the transient one."""
    picky = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(tool=ToolName.WEB_SEARCH, ok=False,
                    summary="search failed: missing required arg 'query'",
                    error="missing required arg 'query'"),
    ])
    picky.args_schema = {}  # declares nothing, so the loop cannot pre-empt it
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "go", query="something"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(picky))

    step = loop_mod.run_agent("Vanta", make_settings()).trace[0]
    assert "malformed call" in step.observation
    assert "usually transient" not in step.observation


def test_schema_guard_does_not_block_well_formed_or_unknown_schemas(wired):
    """A guard that misfires on an unfamiliar schema is worse than one that misses."""
    odd = FakeTool(ToolName.WEB_SEARCH)
    odd.args_schema = {"query": "a bare string, not a dict"}   # unreadable
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "go", query="vanta"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(odd))

    result = loop_mod.run_agent("Vanta", make_settings())
    assert result.trace[0].ok is True
    assert len(odd.calls) == 1


# --------------------------------------------------------------------------
# Citations: a URL earns one by being opened, not by being returned
# --------------------------------------------------------------------------


_JUNK_HITS = [
    {"title": "Help - IRS", "url": "https://irs.gov/help/app", "snippet": "tax help"},
    {"title": "App Store", "url": "https://apple.com/app-store", "snippet": "apps"},
    {"title": "J.K. Rowling", "url": "https://jkrowling.com", "snippet": "author"},
    {"title": "Linear", "url": "https://linear.app", "snippet": "issue tracking for teams"},
]


def _search_only_registry():
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(
            tool=ToolName.WEB_SEARCH, ok=True, summary="4 hits",
            data={"results": list(_JUNK_HITS)},
            sources=[h["url"] for h in _JUNK_HITS],
        )
    ])
    return search


def test_a_search_only_run_cites_nothing(wired):
    """The invariant the docstring always claimed.

    A live Linear run reported 2 searches, 0 fetches and TEN cited sources —
    irs.gov, youtube, J.K. Rowling's homepage — none of which were ever opened.
    """
    search = _search_only_registry()
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="Linear issue tracking company"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("Linear", make_settings())

    assert result.stats.searches == 1 and result.stats.fetches == 0
    assert result.record.sources == [], (
        f"opened nothing, so must cite nothing; got {result.record.sources}"
    )
    for junk in ("irs.gov", "apple.com", "jkrowling.com"):
        assert junk not in " ".join(result.record.sources)


def test_only_the_opened_url_becomes_a_citation(wired):
    """Search surfaces four links, the agent opens one. One citation."""
    search = _search_only_registry()
    fetch = FakeTool(ToolName.FETCH_PAGE, observations=[
        Observation(
            tool=ToolName.FETCH_PAGE, ok=True, summary="Linear homepage",
            data={"text": "Linear is issue tracking for software teams.", "url": "https://linear.app"},
            sources=["https://linear.app"],
        )
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="Linear issue tracking company"),
        act(ToolName.FETCH_PAGE, "read the real one", url="https://linear.app"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search, fetch))

    result = loop_mod.run_agent("Linear", make_settings())

    assert result.record.sources == ["https://linear.app"]
    assert "jkrowling.com" not in " ".join(result.record.sources)


@pytest.mark.parametrize("tool, args, url", [
    (ToolName.FETCH_PAGE, {"url": "https://acme.com/about"}, "https://acme.com/about"),
    (ToolName.DETECT_TECH, {"url": "https://acme.com"}, "https://acme.com"),
    (ToolName.FETCH_JOBS, {"company_or_url": "acme.com"}, "https://jobs.ashbyhq.com/acme"),
])
def test_every_reading_tool_still_contributes_its_url(wired, tool, args, url):
    """detect_tech and fetch_jobs read documents, so they must still cite."""
    reader = FakeTool(tool, observations=[
        Observation(tool=tool, ok=True, summary="read it",
                    data={"text": "some real page text", "tech": []}, sources=[url])
    ])
    llm = FakeLLM([
        AgentAction(tool=tool, thought="read", args=args),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(reader))

    result = loop_mod.run_agent("Acme", make_settings())
    assert url in result.record.sources, f"{tool.value} reads a document and must cite it"


def test_search_snippets_are_attributed_per_result_not_to_the_first_hit(wired):
    """A fact from result #4 must not be filed under result #1's URL."""
    search = _search_only_registry()
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="Linear issue tracking company"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    state = wired(llm, registry_of(search))

    loop_mod.run_agent("Linear", make_settings())

    evidence = state["evidence"]
    assert len(evidence) == len(_JUNK_HITS), "one evidence block per result"
    by_url = dict(evidence)
    # each snippet sits under its OWN url
    assert "issue tracking" in by_url["https://linear.app"]
    assert "author" in by_url["https://jkrowling.com"]
    # nothing is lumped under whichever result sorted first
    assert "author" not in by_url["https://irs.gov/help/app"]


def test_snippets_still_reach_extraction(wired):
    """Snippets inform the record — they just never become citations."""
    search = _search_only_registry()
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "look it up", query="Linear issue tracking company"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    state = wired(llm, registry_of(search))

    result = loop_mod.run_agent("Linear", make_settings())

    assert state["evidence"], "search snippets must still reach extraction"
    # ...but contribute no citations
    assert result.record.sources == []


def test_a_url_invented_downstream_cannot_become_a_citation(wired):
    """Belt and braces: the invariant holds whatever extraction proposes."""
    fetch = FakeTool(ToolName.FETCH_PAGE, observations=[
        Observation(tool=ToolName.FETCH_PAGE, ok=True, summary="page",
                    data={"text": "real text"}, sources=["https://acme.com"])
    ])
    forged = CompanyRecord(
        input="Acme", company_name="Acme", industry="fintech",
        sources=["https://acme.com", "https://jkrowling.com", "https://never-opened.example"],
    )
    llm = FakeLLM([
        act(ToolName.FETCH_PAGE, "read", url="https://acme.com"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(fetch), extract=forged)

    result = loop_mod.run_agent("Acme", make_settings())

    assert result.record.sources == ["https://acme.com"]


# --------------------------------------------------------------------------
# Search provider fallback, surfaced like the capability rung
# --------------------------------------------------------------------------


def test_search_provider_label_names_who_actually_served():
    from lea.agent.loop import search_provider_label

    assert search_provider_label(["scrappa"], "scrappa") == "scrappa"
    assert search_provider_label(["scrappa", "duckduckgo"], "scrappa") == \
        "duckduckgo (fell back from scrappa)"
    assert search_provider_label(["scrappa", "duckduckgo", "fixture"], "scrappa") == \
        "fixture (fell back from scrappa)"
    # nothing observed -> report what was configured
    assert search_provider_label([], "tavily") == "tavily"


def _search_obs(provider: str, chain: list[str]) -> Observation:
    return Observation(
        tool=ToolName.WEB_SEARCH, ok=True, summary="1 hit",
        data={"provider": provider, "provider_chain": chain,
              "results": [{"title": "Acme", "url": "https://acme.com", "snippet": "payments"}]},
        sources=["https://acme.com"],
    )


def test_a_fallback_is_recorded_in_the_stats(wired):
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        _search_obs("duckduckgo", ["scrappa", "duckduckgo"]),
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "s", query="acme corp overview"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme", make_settings())
    assert result.stats.search_provider == "duckduckgo (fell back from scrappa)"


def test_no_fallback_reports_the_provider_plainly(wired):
    search = FakeTool(ToolName.WEB_SEARCH, observations=[_search_obs("scrappa", ["scrappa"])])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "s", query="acme corp overview"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search))

    result = loop_mod.run_agent("acme", make_settings())
    assert result.stats.search_provider == "scrappa"
    assert "fell back" not in result.stats.search_provider


# --------------------------------------------------------------------------
# The guards must not defend junk
# --------------------------------------------------------------------------


_JUNK_URL = "https://theworkspaceforchildren.com/about"   # the real live offender
_GOOD_URL = "https://notion.so/about"


def test_relevant_unread_urls_separates_junk_from_the_target():
    from lea.agent.policy import relevant_unread_urls

    history = [_search_step(1, "Notion workspace software company", _JUNK_URL, _GOOD_URL,
                            "https://www.crunchbase.com/organization/notion")]

    rel = relevant_unread_urls(history, "Notion")
    assert _GOOD_URL in rel
    assert "https://www.crunchbase.com/organization/notion" in rel
    assert _JUNK_URL not in rel

    # with no target nothing can be judged, so nothing is filtered
    from lea.agent.policy import unread_urls

    assert relevant_unread_urls(history, "") == unread_urls(history)


def test_search_guard_stays_silent_when_every_unread_link_is_junk():
    """The live Notion bug: three correct re-anchors blocked to defend a
    children's play consultancy."""
    from lea.agent.policy import search_without_reading_reason

    history = [_search_step(1, "Notion workspace software company", _JUNK_URL)]

    # re-anchoring on the domain, and on a site: query, are exactly right here
    for q in ("notion.so", "site:notion.com about"):
        assert search_without_reading_reason(
            act(ToolName.WEB_SEARCH, query=q), history, target="Notion"
        ) is None, f"guard must not block the re-anchor {q!r}"

    # ...but once something relevant IS sitting unread, it fires again
    history.append(_search_step(2, "site:notion.so about", _GOOD_URL))
    reason = search_without_reading_reason(
        act(ToolName.WEB_SEARCH, query="notion pricing"), history, target="Notion"
    )
    assert reason and _GOOD_URL in reason


def test_stall_grace_is_not_extended_for_junk_links():
    """Extending the leash to 'go read junk' is not a rescue."""
    rec = CompanyRecord(input="Notion")
    budget = Budget(max_steps=12, max_searches=5, max_fetches=6, total_timeout_s=60)
    dead = [TraceStep(n=i, tool=ToolName.WEB_SEARCH, ok=False, error="boom")
            for i in range(2, 5)]

    junk = [_search_step(1, "q", _JUNK_URL), *dead]
    assert should_stop(rec, budget, junk, "Notion") is not None, "junk is not a rescue"

    good = [_search_step(1, "q", _GOOD_URL), *dead]
    assert should_stop(rec, budget, good, "Notion") is None, "a real lead earns the grace"


def test_redundancy_relaxes_while_the_agent_is_stuck_in_junk():
    """Escaping a hole is not circling in one."""
    from lea.agent.policy import (
        QUERY_SIMILARITY_THRESHOLD, STUCK_SIMILARITY_THRESHOLD, is_redundant, query_similarity,
    )

    assert STUCK_SIMILARITY_THRESHOLD > QUERY_SIMILARITY_THRESHOLD

    first = "notion workspace software docs"
    junk_only = [_search_step(1, first, _JUNK_URL)]
    reword = act(ToolName.WEB_SEARCH, query="notion workspace software docs wiki")
    # similar enough to be blocked normally...
    sim = query_similarity("notion workspace software docs wiki", first)
    assert QUERY_SIMILARITY_THRESHOLD <= sim < STUCK_SIMILARITY_THRESHOLD
    # ...but allowed while nothing relevant has been found
    assert is_redundant(reword, junk_only, "Notion") is None

    # an exact repeat is still blocked: same query, same junk
    exact = act(ToolName.WEB_SEARCH, query=first)
    blocked = is_redundant(exact, junk_only, "Notion")
    assert blocked and "re-anchor" in blocked.lower()

    # and once something relevant is found, the normal threshold returns
    found = [*junk_only, _search_step(2, "site:notion.so", _GOOD_URL)]
    assert is_redundant(reword, found, "Notion") is not None


def test_loop_lets_a_re_anchor_through_after_junk(wired):
    """End to end: search returns junk, the agent re-anchors, nothing is bounced."""
    search = FakeTool(ToolName.WEB_SEARCH, observations=[
        Observation(tool=ToolName.WEB_SEARCH, ok=True, summary="junk",
                    data={"results": [{"title": "Kids workspace", "url": _JUNK_URL, "snippet": "play"}]},
                    sources=[_JUNK_URL]),
        Observation(tool=ToolName.WEB_SEARCH, ok=True, summary="good",
                    data={"results": [{"title": "Notion", "url": _GOOD_URL, "snippet": "workspace"}]},
                    sources=[_GOOD_URL]),
    ])
    llm = FakeLLM([
        act(ToolName.WEB_SEARCH, "broad", query="Notion workspace software company"),
        act(ToolName.WEB_SEARCH, "re-anchor on the domain", query="site:notion.so about"),
        act(ToolName.FINISH, "done", reason="ok"),
    ])
    wired(llm, registry_of(search, FakeTool(ToolName.FETCH_PAGE)))

    result = loop_mod.run_agent("Notion", make_settings())

    bounced = [s for s in result.trace if s.error in ("unread sources", "redundant")]
    assert not bounced, f"the re-anchor was blocked: {[s.observation for s in bounced]}"
    assert len(search.calls) == 2
