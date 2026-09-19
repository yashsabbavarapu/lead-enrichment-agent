"""Rendering + persistence tests. Nothing here touches the network or an LLM.

Rendering is a demo surface, so the bar is: it must never raise on a partial or
ugly record, and a saved run must replay byte-identically.
"""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from lea import trace as trace_mod
from lea.models import (
    CompanyRecord,
    CriterionScore,
    EnrichmentResult,
    ICPFit,
    OutreachHook,
    RunStats,
    Signal,
    TechStack,
    ToolName,
    TraceStep,
)


def cap() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    # no_color + fixed width so assertions are about text, not ANSI codes
    return Console(file=buf, width=100, no_color=True, force_terminal=False, legacy_windows=False), buf


def full_result() -> EnrichmentResult:
    record = CompanyRecord(
        input="ramp.com",
        company_name="Ramp",
        domain="ramp.com",
        description="Corporate cards and spend management.",
        industry="Fintech",
        sub_industry="Spend management",
        employee_range="501-1000",
        employee_estimate=900,
        hq_city="New York",
        hq_country="USA",
        founded_year=2019,
        signals=[
            Signal(
                kind="funding", headline="Raised $200M Series D",
                detail="Led by Founders Fund.", date="2025-03-11",
                source_url="https://example.com/news", recency_note="within 90 days",
            ),
            Signal(kind="hiring", headline="18 open GTM roles"),
        ],
        tech_stack=[
            TechStack(category="analytics", name="Segment", evidence="script tag", confidence=0.8),
            TechStack(category="analytics", name="Amplitude"),
            TechStack(category="hosting", name="Cloudflare"),
        ],
        icp_fit=ICPFit(
            score=84.0, tier="A", summary="Strong fit: fintech, right size, active hiring.",
            criteria=[
                CriterionScore(key="size", score=9.0, weight=2.0, reasoning="900 employees"),
                CriterionScore(key="industry", score=8.0, weight=1.5, reasoning="fintech"),
                CriterionScore(key="momentum", score=3.0, weight=1.0, reasoning="thin evidence"),
            ],
            disqualifiers_hit=[],
            confidence=0.72,
        ),
        hooks=[
            OutreachHook(
                angle="Series D -> GTM headcount",
                hook="Congrats on the Series D — with 18 GTM roles open, onboarding is about to get loud.",
                grounded_in="funding signal", source_url="https://example.com/news",
            )
        ],
        unknowns=["exact ARR"],
        sources=["https://ramp.com", "https://example.com/news"],
    )
    trace = [
        TraceStep(
            n=1, tool=ToolName.WEB_SEARCH, thought="I don't know the official domain yet.",
            args={"query": "ramp corporate cards official site"}, ok=True,
            observation="1. Ramp — https://ramp.com — Corporate cards. " + "lorem ipsum " * 60,
            sources=["https://ramp.com"], latency_ms=412, tokens_in=900, tokens_out=60,
        ),
        TraceStep(
            n=2, tool=ToolName.FETCH_PAGE, thought="Read the about page for firmographics.",
            args={"url": "https://ramp.com/about"}, ok=True,
            observation="Ramp is a finance automation platform based in New York.",
            sources=["https://ramp.com/about"], latency_ms=650,
        ),
        TraceStep(
            n=3, tool=ToolName.FETCH_PAGE, thought="Read the about page again.",
            args={"url": "https://www.ramp.com/about/"}, ok=False,
            observation="redundant: you already ran fetch_page on https://ramp.com/about at step 2.",
            error="redundant",
        ),
        TraceStep(
            n=4, tool=ToolName.DETECT_TECH, thought="Check the stack.",
            args={"url": "https://ramp.com"}, ok=True,
            observation="Segment, Amplitude, Cloudflare", latency_ms=310,
        ),
        TraceStep(
            n=5, tool=ToolName.FINISH, thought="synthesize: extract structured record from 3 sources.",
            ok=True, observation="resolved 5 core field(s).",
        ),
    ]
    return EnrichmentResult(
        record=record, trace=trace,
        stats=RunStats(
            steps=5, searches=1, fetches=3, tokens_in=900, tokens_out=60,
            wall_ms=4200, stop_reason="complete: agent called finish",
            llm_provider="anthropic", search_provider="tavily",
        ),
        finished_at="2025-05-01T12:00:05+00:00",
    )


# --------------------------------------------------------------------------
# Step rendering
# --------------------------------------------------------------------------


def test_render_step_shows_thought_call_and_truncated_observation():
    console, buf = cap()
    step = full_result().trace[0]
    trace_mod.render_step(step, console)
    out = buf.getvalue()

    assert "search" in out
    assert "I don't know the official domain yet." in out
    assert 'web_search("ramp corporate cards official site")' in out
    assert "412ms" in out
    assert "ok" in out
    # long observations are cut to a few lines, never dumped whole
    body = [l for l in out.splitlines() if "│" in l]
    assert 1 <= len(body) <= trace_mod.OBS_PREVIEW_LINES + 1
    assert "…" in out


def test_render_step_marks_failures():
    console, buf = cap()
    trace_mod.render_step(full_result().trace[2], console)
    out = buf.getvalue()
    assert "fail" in out
    assert "redundant" in out


def test_render_step_survives_an_empty_step():
    console, _ = cap()
    trace_mod.render_step(TraceStep(n=1, tool=ToolName.FINISH), console)  # must not raise


def test_call_signature_forms():
    sig = trace_mod._call_signature
    assert sig(TraceStep(n=1, tool=ToolName.WEB_SEARCH, args={"query": "acme funding"})) == \
        'web_search("acme funding")'
    assert sig(TraceStep(n=1, tool=ToolName.FETCH_PAGE, args={})) == "fetch_page()"
    long = sig(TraceStep(n=1, tool=ToolName.WEB_SEARCH, args={"query": "x" * 200}))
    assert len(long) < 120 and long.endswith('")')


# --------------------------------------------------------------------------
# Result rendering
# --------------------------------------------------------------------------


def test_render_result_shows_every_section():
    console, buf = cap()
    trace_mod.render_result(full_result(), console)
    out = buf.getvalue()

    for expected in (
        "Ramp", "ramp.com", "tier A", "84/100",
        "FIRMOGRAPHICS", "Fintech", "501-1000", "New York",
        "SIGNALS", "Raised $200M Series D", "2025-03-11",
        "TECH STACK", "analytics", "Segment",
        "ICP SCORING", "momentum", "thin evidence",
        "OUTREACH HOOKS", "Series D", "Congrats on the Series D",
        "unknowns:", "exact ARR",
        "5 steps", "1 searches", "3 pages", "agent called finish",
    ):
        assert expected in out, f"missing from report: {expected!r}"


def test_render_result_handles_a_nearly_empty_record():
    console, buf = cap()
    bare = EnrichmentResult(record=CompanyRecord(input="mystery co"))
    trace_mod.render_result(bare, console)  # must not raise
    out = buf.getvalue()
    assert "mystery co" in out
    assert "unknown" in out
    assert "NO RESEARCH SUCCEEDED" in out


def blocked_result() -> EnrichmentResult:
    """The live failure mode: every search is rate-limited, nothing is learned, and
    the old renderer still printed a confident-looking 'ICP tier C 50/100'."""
    return EnrichmentResult(
        record=CompanyRecord(input="ramp.com", domain="ramp.com"),
        trace=[
            TraceStep(
                n=n, tool=ToolName.WEB_SEARCH, args={"query": q}, ok=False,
                observation="search failed: duckduckgo returned no parseable results",
                error="duckduckgo returned no parseable results (rate limited?)",
            )
            for n, q in ((1, "ramp.com company overview"), (2, "ramp.com funding news"))
        ],
        stats=RunStats(
            steps=2, searches=2, wall_ms=900,
            stop_reason="blocked: no evidence gathered — every research step failed",
            llm_provider="stub", search_provider="duckduckgo",
        ),
    )


def test_blocked_run_never_renders_a_score():
    """Bug 2: a run that established nothing must read as a non-result."""
    assert trace_mod.is_ungrounded(blocked_result())

    console, buf = cap()
    trace_mod.render_result(blocked_result(), console)
    out = buf.getvalue()

    assert "NO RESEARCH SUCCEEDED" in out
    assert "not an assessment of the company" in out
    # the remedies belong in front of the user, not in a docstring
    assert "TAVILY_API_KEY" in out and "--offline" in out and "retry" in out
    assert "rate limited" in out          # the actual failure is named
    assert "tier" not in out.lower()      # no tier...
    assert "/100" not in out              # ...and no score, anywhere
    assert "BLOCKED" in out and "COMPLETE" not in out


def test_low_confidence_score_is_marked_provisional():
    console, buf = cap()
    result = full_result()
    result.record.icp_fit.confidence = 0.05
    trace_mod.render_result(result, console)
    out = buf.getvalue()

    assert "PROVISIONAL" in out
    assert "do not treat this tier as a finding" in out
    assert "provisional — graded without direct evidence" in out


def test_grounded_run_still_renders_normally():
    """The guard must not cry wolf on a real result."""
    result = full_result()
    assert not trace_mod.is_ungrounded(result)

    console, buf = cap()
    trace_mod.render_result(result, console)
    out = buf.getvalue()
    assert "NO RESEARCH SUCCEEDED" not in out
    assert "PROVISIONAL" not in out
    assert "tier A" in out


def test_blocked_markdown_carries_the_same_warning(tmp_path):
    md = trace_mod.save_trace_markdown(blocked_result(), tmp_path / "blocked.md").read_text()
    assert "NO RESEARCH SUCCEEDED" in md
    assert "**BLOCKED**" in md
    assert "TAVILY_API_KEY" in md


# --------------------------------------------------------------------------
# Persistence + replay
# --------------------------------------------------------------------------


def test_save_json_and_replay_round_trip(tmp_path):
    result = full_result()
    path = trace_mod.save_json(result, tmp_path / "nested" / "ramp.json")
    assert path.is_file()

    raw = json.loads(path.read_text())
    assert raw["record"]["company_name"] == "Ramp"
    assert len(raw["trace"]) == 5

    console, buf = cap()
    replayed = trace_mod.replay(path, console, delay=0.0)

    assert replayed.model_dump() == result.model_dump()
    out = buf.getvalue()
    assert "researching ramp.com" in out
    for step in result.trace:
        assert step.thought.split(".")[0] in out
    assert "OUTREACH HOOKS" in out


def test_save_json_serializes_the_populated_record_not_a_stale_copy(tmp_path):
    """Guards the reported 'report showed fields but the JSON was all None'.

    Rebuilds the exact aliasing the loop relies on: synthesis returns a NEW record
    object, which is then mutated (icp_fit, hooks) and handed to EnrichmentResult.
    If pydantic snapshotted the record at construction, or save_json dumped
    anything other than the live object, these fields would come back None.
    """
    synthesized = CompanyRecord(
        input="acme.com", company_name="Acme", domain="acme.com",
        industry="Fintech", employee_range="51-200", hq_city="Berlin",
        signals=[Signal(kind="funding", headline="Series A")],
        tech_stack=[TechStack(category="hosting", name="Vercel")],
        sources=["https://acme.com"],
    )
    result = EnrichmentResult(record=synthesized, stats=RunStats(steps=3))
    # mutate AFTER the result is built, exactly as run_agent does
    synthesized.icp_fit = ICPFit(score=77.0, tier="B", summary="good", confidence=0.6)
    synthesized.hooks = [OutreachHook(angle="funding", hook="Congrats on the raise.")]
    synthesized.unknowns = ["ARR"]

    assert result.record is synthesized, "EnrichmentResult must alias, not copy, the record"

    path = trace_mod.save_json(result, tmp_path / "acme.json")
    back = trace_mod.load_result(path)

    for field in ("company_name", "domain", "industry", "employee_range", "hq_city"):
        assert getattr(back.record, field) == getattr(synthesized, field), field
    assert back.record.icp_fit is not None and back.record.icp_fit.score == 77.0
    assert [h.angle for h in back.record.hooks] == ["funding"]
    assert back.record.unknowns == ["ARR"]
    assert [s.headline for s in back.record.signals] == ["Series A"]
    assert [t.name for t in back.record.tech_stack] == ["Vercel"]
    # and nothing silently dropped on the way through JSON
    assert back.model_dump() == result.model_dump()


def test_replay_delay_is_honoured(monkeypatch, tmp_path):
    slept: list[float] = []
    monkeypatch.setattr(trace_mod.time, "sleep", lambda s: slept.append(s))
    path = trace_mod.save_json(full_result(), tmp_path / "r.json")
    console, _ = cap()
    trace_mod.replay(path, console, delay=0.25)
    assert slept == [0.25] * 5


def test_save_trace_markdown_is_a_readable_research_log(tmp_path):
    path = trace_mod.save_trace_markdown(full_result(), tmp_path / "ramp.trace.md")
    md = path.read_text()

    assert md.startswith("# Research log: Ramp")
    for expected in (
        "## Result", "| domain | ramp.com |",
        "## Signals", "**[funding]** Raised $200M Series D",
        "## Tech stack", "**analytics**: Amplitude, Segment",
        "## ICP scoring", "| momentum | 3.0 |",
        "## Outreach hooks", "> Congrats on the Series D",
        "## Agent trace", '`web_search("ramp corporate cards official site")`',
        "(FAILED,", "## Still unknown", "- exact ARR",
        "## Sources", "- <https://ramp.com>",
        "agent called finish",
    ):
        assert expected in md, f"missing from markdown: {expected!r}"


def test_markdown_survives_a_bare_record(tmp_path):
    bare = EnrichmentResult(record=CompanyRecord(input="mystery co"))
    md = trace_mod.save_trace_markdown(bare, tmp_path / "bare.md").read_text()
    assert "# Research log: mystery co" in md


def test_load_result_rejects_garbage(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(Exception):
        trace_mod.load_result(p)


# --------------------------------------------------------------------------
# Batch summary
# --------------------------------------------------------------------------


def test_summary_table_sorts_by_icp_score_descending():
    def mk(name: str, score: float | None) -> EnrichmentResult:
        fit = ICPFit(score=score, tier="A" if score and score > 70 else "C") if score is not None else None
        return EnrichmentResult(record=CompanyRecord(input=name, company_name=name, icp_fit=fit))

    console, buf = cap()
    console.print(trace_mod.summary_table([mk("Low", 30.0), mk("High", 90.0), mk("None", None), mk("Mid", 60.0)]))
    out = buf.getvalue()

    order = [out.index(n) for n in ("High", "Mid", "Low", "None")]
    assert order == sorted(order)
