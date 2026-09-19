"""LLM layer tests. No network, no API key, no SDK required."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Optional

import httpx
import pytest
from pydantic import BaseModel

from lea import prompts
from lea.config import Settings
from lea.llm.factory import get_llm
from lea.llm.json_utils import (
    coerce_to_model,
    extract_json,
    find_json_object,
    repair_prompt,
    strip_code_fences,
    strip_trailing_commas,
)
from lea.llm.stub import StubClient
from lea.models import AgentAction, CompanyRecord, ICPFit, ToolName, TraceStep

# ---------------------------------------------------------------------------
# json_utils
# ---------------------------------------------------------------------------


class Tiny(BaseModel):
    name: str
    count: int = 0
    note: Optional[str] = None


def test_extract_json_plain():
    assert extract_json('{"name": "acme", "count": 3}') == {"name": "acme", "count": 3}


def test_extract_json_from_fenced_block():
    raw = 'Sure! Here you go:\n\n```json\n{"name": "acme", "count": 3}\n```\n\nHope that helps.'
    assert extract_json(raw) == {"name": "acme", "count": 3}


def test_extract_json_from_unlabelled_fence():
    raw = "```\n{\"name\": \"acme\"}\n```"
    assert extract_json(raw) == {"name": "acme"}


def test_extract_json_tolerates_trailing_commas():
    raw = '```json\n{\n  "name": "acme",\n  "count": 2,\n  "tags": ["a", "b",],\n}\n```'
    assert extract_json(raw) == {"name": "acme", "count": 2, "tags": ["a", "b"]}


def test_extract_json_ignores_prose_around_object():
    raw = 'I think the answer is:\n{"name": "acme"}\nLet me know if you need more.'
    assert extract_json(raw) == {"name": "acme"}


def test_find_json_object_is_string_aware():
    raw = '{"note": "a } brace in a string", "n": {"deep": 1}}'
    assert find_json_object("prefix " + raw + " suffix") == raw


def test_find_json_object_repairs_truncated_output():
    obj = find_json_object('{"name": "acme", "inner": {"a": 1')
    assert obj is not None
    assert extract_json(obj) == {"name": "acme", "inner": {"a": 1}}


def test_find_json_object_returns_none_without_braces():
    assert find_json_object("no json at all") is None


def test_strip_code_fences_passthrough():
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_strip_trailing_commas_leaves_strings_alone():
    assert strip_trailing_commas('{"a": "x, }", "b": 1,}') == '{"a": "x, }", "b": 1}'


def test_extract_json_returns_none_on_garbage():
    assert extract_json("totally not json") is None
    assert extract_json(None) is None
    assert extract_json(123) is None


def test_coerce_to_model_happy_and_sad_paths():
    assert coerce_to_model('```json\n{"name": "acme", "count": 4,}\n```', Tiny) == Tiny(name="acme", count=4)
    assert coerce_to_model('{"count": "not-an-int"}', Tiny) is None  # missing required + bad type
    assert coerce_to_model("no json", Tiny) is None
    assert coerce_to_model({"name": "acme"}, Tiny) == Tiny(name="acme")


def test_repair_prompt_names_the_failing_fields():
    try:
        Tiny.model_validate({"count": "abc"})
    except Exception as exc:  # pydantic ValidationError
        text = repair_prompt('{"count": "abc"}', exc)
    assert "name" in text
    assert "count" in text
    assert "schema" in text.lower()


# ---------------------------------------------------------------------------
# StubClient.decide — trajectory
# ---------------------------------------------------------------------------

TOOL_SPECS = [
    {"name": "web_search", "description": "Search the web", "args_schema": {"properties": {"query": {"type": "string"}}}},
    {"name": "fetch_page", "description": "Fetch a page", "args_schema": {"properties": {"url": {"type": "string"}}}},
    {"name": "detect_tech", "description": "Detect tech", "args_schema": {"properties": {"url": {"type": "string"}}}},
    {"name": "finish", "description": "Stop", "args_schema": {"properties": {}}},
]

TARGET = "Acme Robotics"


def _ctx(trace: list[TraceStep], steps_left: int = 9) -> str:
    return prompts.build_agent_context(
        target=TARGET,
        record_so_far=CompanyRecord(input=TARGET),
        trace=trace,
        remaining_budget={"steps": steps_left, "searches": 4, "fetches": 5},
        known_gaps=["industry", "signals"],
    )


def _step(n: int, action: AgentAction, observation: str, sources: list[str] | None = None) -> TraceStep:
    return TraceStep(
        n=n,
        thought=action.thought,
        tool=action.tool,
        args=action.args,
        observation=observation,
        sources=sources or [],
    )


def _run_trajectory(max_steps: int = 8) -> list[AgentAction]:
    """Drive the stub through a simulated research run."""
    client = StubClient(Settings())
    trace: list[TraceStep] = []
    actions: list[AgentAction] = []

    search_results = (
        "results: https://acmerobotics.com/ , https://www.linkedin.com/company/acme-robotics , "
        "https://www.crunchbase.com/organization/acme-robotics"
    )
    news_results = (
        "results: https://acmerobotics.com/careers , https://acmerobotics.com/news/series-b , "
        "https://techcrunch.com/2025/01/acme-robotics-raises"
    )

    for i in range(1, max_steps + 1):
        action, usage = client.decide(prompts.AGENT_SYSTEM, _ctx(trace, steps_left=max_steps - i + 2), TOOL_SPECS)
        assert usage.tokens_in == 0 and usage.tokens_out == 0 and usage.model == "stub"
        actions.append(action)
        if action.tool is ToolName.FINISH:
            break
        if action.tool is ToolName.WEB_SEARCH:
            obs = news_results if "funding" in action.args.get("query", "") else search_results
            srcs = [u.strip() for u in obs.split("results:")[1].split(",")]
        elif action.tool is ToolName.DETECT_TECH:
            obs = "detected: Next.js, Segment, HubSpot"
            srcs = [action.args.get("url", "")]
        else:
            obs = f"fetched page text for {action.args.get('url')} — Acme Robotics, founded 2019, 120 employees."
            srcs = [action.args.get("url", "")]
        trace.append(_step(i, action, obs, srcs))
    return actions


def test_stub_decide_produces_a_plausible_trajectory():
    actions = _run_trajectory()
    tools = [a.tool for a in actions]

    assert tools[0] is ToolName.WEB_SEARCH
    assert "Acme Robotics" in actions[0].args["query"]
    assert tools[1] is ToolName.FETCH_PAGE
    assert actions[1].args["url"].startswith("https://acmerobotics.com")
    assert ToolName.DETECT_TECH in tools
    assert tools[-1] is ToolName.FINISH
    # A second, signal-oriented search happens before finishing.
    queries = [a.args.get("query", "") for a in actions if a.tool is ToolName.WEB_SEARCH]
    assert len(queries) == 2
    assert "funding" in queries[1]
    assert all(a.thought.strip() for a in actions)


def test_stub_decide_never_repeats_a_fetch():
    actions = _run_trajectory()
    fetched = [a.args.get("url") for a in actions if a.tool is ToolName.FETCH_PAGE]
    assert fetched
    assert len(fetched) == len(set(fetched))
    searched = [a.args.get("query") for a in actions if a.tool is ToolName.WEB_SEARCH]
    assert len(searched) == len(set(searched))


def test_stub_decide_prefers_official_domain_over_aggregators():
    client = StubClient(Settings())
    trace = [
        _step(
            1,
            AgentAction(thought="t", tool=ToolName.WEB_SEARCH, args={"query": "Acme Robotics company overview"}),
            "results",
            ["https://www.linkedin.com/company/acme-robotics", "https://acmerobotics.com/", "https://en.wikipedia.org/wiki/Acme"],
        )
    ]
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx(trace), TOOL_SPECS)
    assert action.tool is ToolName.FETCH_PAGE
    assert "acmerobotics.com" in action.args["url"]
    assert "linkedin" not in action.args["url"]


def test_stub_decide_finishes_when_budget_is_spent():
    client = StubClient(Settings())
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx([], steps_left=1), TOOL_SPECS)
    assert action.tool is ToolName.FINISH
    assert action.thought


def test_stub_decide_respects_a_restricted_tool_set():
    client = StubClient(Settings())
    only_finish = [{"name": "finish", "description": "Stop", "args_schema": {"properties": {}}}]
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx([]), only_finish)
    assert action.tool is ToolName.FINISH


def _failed_search_step(n: int, query: str, error: str = "search provider returned 429") -> TraceStep:
    return TraceStep(
        n=n,
        thought="looking for the official site",
        tool=ToolName.WEB_SEARCH,
        args={"query": query},
        ok=False,
        error=error,
        observation="",
    )


def test_stub_decide_retries_a_different_query_after_a_failed_search():
    client = StubClient(Settings())
    trace = [_failed_search_step(1, "Acme Robotics company overview")]
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx(trace), TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH, "a provider error is not a reason to finish"
    assert action.args["query"] != "Acme Robotics company overview"
    assert "Acme Robotics" in action.args["query"]


def test_stub_decide_tries_the_obvious_domain_after_repeated_search_failures():
    client = StubClient(Settings())
    trace = [
        _failed_search_step(1, "Acme Robotics company overview"),
        _failed_search_step(2, '"Acme Robotics" official site', "blocked"),
        _failed_search_step(3, "Acme Robotics company overview", "blocked"),
    ]
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx(trace), TOOL_SPECS)

    assert action.tool is ToolName.FETCH_PAGE
    assert action.args["url"] == "https://acmerobotics.com"
    assert action.thought


def test_stub_decide_does_not_give_up_after_two_failed_searches():
    client = StubClient(Settings())
    trace = [_failed_search_step(1, "q1"), _failed_search_step(2, "q2")]
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx(trace), TOOL_SPECS)
    assert action.tool is not ToolName.FINISH


def test_stub_decide_still_finishes_when_search_fails_and_fetching_is_impossible():
    client = StubClient(Settings())
    trace = [_failed_search_step(n, f"q{n}") for n in range(1, 5)]
    only_search = [s for s in TOOL_SPECS if s["name"] in ("web_search", "finish")]
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx(trace), only_search)
    assert action.tool is ToolName.FINISH


def test_stub_decide_uses_declared_arg_names():
    client = StubClient(Settings())
    specs = [{"name": "web_search", "description": "s", "args_schema": {"properties": {"q": {"type": "string"}}}}]
    action, _ = client.decide(prompts.AGENT_SYSTEM, _ctx([]), specs)
    assert "q" in action.args


# ---------------------------------------------------------------------------
# StubClient.extract
# ---------------------------------------------------------------------------

EVIDENCE = [
    (
        "https://acmerobotics.com/about",
        "Acme Robotics builds warehouse automation robots for mid-market logistics operators. "
        "Founded in 2019 and headquartered in Austin, Texas, the company has 120 employees. "
        "We're hiring across engineering and go-to-market.",
    ),
    (
        "https://techcrunch.com/2025/01/acme-robotics-raises",
        "Acme Robotics raised a $40m Series B led by Example Ventures to expand its logistics platform.",
    ),
]


def test_stub_extract_returns_a_valid_company_record():
    client = StubClient(Settings())
    ctx = prompts.build_extraction_context(TARGET, EVIDENCE)
    record, usage = client.extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)

    assert isinstance(record, CompanyRecord)
    assert record.input == TARGET
    assert record.founded_year == 2019
    assert record.employee_estimate == 120
    assert record.hq_city == "Austin"
    assert record.domain == "acmerobotics.com"
    assert record.sources
    assert isinstance(record.unknowns, list)
    assert usage.tokens_in == 0 and usage.tokens_out == 0 and usage.model == "stub"


def test_stub_extract_returns_a_valid_icp_fit():
    client = StubClient(Settings())
    ctx = prompts.build_extraction_context(TARGET, EVIDENCE)
    fit, usage = client.extract(prompts.ICP_SCORING_SYSTEM, ctx, ICPFit)

    assert isinstance(fit, ICPFit)
    assert 0 <= fit.score <= 100
    assert fit.tier in ("A", "B", "C", "D")
    assert usage.model == "stub"


def test_stub_extract_is_generic_over_arbitrary_models():
    class Weird(BaseModel):
        input: str
        required_thing: str
        maybe: Optional[int] = None
        items: list[str] = []

    client = StubClient(Settings())
    obj, _ = client.extract("sys", prompts.build_extraction_context(TARGET, EVIDENCE), Weird)
    assert isinstance(obj, Weird)
    assert obj.input == TARGET
    assert isinstance(obj.required_thing, str)


def test_stub_extract_survives_empty_context():
    client = StubClient(Settings())
    record, _ = client.extract(prompts.EXTRACTION_SYSTEM, "", CompanyRecord)
    assert isinstance(record, CompanyRecord)
    assert record.unknowns


def test_stub_extract_marks_unknowns():
    client = StubClient(Settings())
    ctx = prompts.build_extraction_context(TARGET, [("https://acmerobotics.com/", "A page with nothing useful.")])
    record, _ = client.extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert "founded_year" in record.unknowns
    assert "employee_estimate" in record.unknowns


@pytest.mark.parametrize(
    "text,city,country",
    [
        ("Headquartered in San Francisco, California. Founded in 2018.", "San Francisco", "United States"),
        ("The company is based in Austin, Texas and ships robots.", "Austin", "United States"),
        ("HQ: Berlin, Germany. We are hiring.", "Berlin", "Germany"),
        ("Based in London. Founded in 2015.", "London", None),
        ("No location information whatsoever in this sentence.", None, None),
    ],
)
def test_stub_hq_heuristics(text, city, country):
    client = StubClient(Settings())
    ctx = prompts.build_extraction_context(TARGET, [("https://acmerobotics.com/", text)])
    record, _ = client.extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.hq_city == city
    assert record.hq_country == country


def test_stub_does_not_bleed_a_place_name_across_a_sentence():
    client = StubClient(Settings())
    ctx = prompts.build_extraction_context(
        TARGET, [("https://acmerobotics.com/", "Headquartered in Boston, Massachusetts. Founded in 2019.")]
    )
    record, _ = client.extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.hq_country == "United States"
    assert "Founded" not in (record.hq_country or "")
    assert record.founded_year == 2019


# --- the "Ramp" regression: a real keyless run that used to produce garbage ----

RAMP_EVIDENCE = [
    (
        "https://ramp.com/about-us/",
        "<title>About Us - Different By Design | Ramp</title>\n"
        "Ramp is the finance operations platform designed to save businesses time and money. "
        "Ramp's corporate card and spend management software help companies control expenses. "
        "Founded in 2019 and headquartered in New York, NY. "
        "Our team of 1,200 people is growing fast.",
    ),
    (
        "https://ramp.com/blog/ramp-raises-500m",
        "Ramp Raises $500M to Deepen Product Innovation & Expand AI for Finance\n"
        "July 30, 2025\n"
        "Today we're announcing that Ramp has raised $500M in a Series E round led by Founders "
        "Fund, at a $22.5 billion valuation.",
    ),
    ("https://ramp.com/careers", "Join our team. We have 140 open roles. Powered by Greenhouse."),
    ("https://tracxn.com/d/companies/ramp", "Fetch failed: HTTP 403 Forbidden."),
]


def _ramp_record() -> CompanyRecord:
    ctx = prompts.build_extraction_context("Ramp", RAMP_EVIDENCE)
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    return record


def test_ramp_funding_signal_is_extracted_with_amount_date_and_source():
    record = _ramp_record()
    funding = [s for s in record.signals if s.kind == "funding"]
    assert funding, "a $500M round with a date was right there in the evidence"
    signal = funding[0]
    assert signal.date == "July 30, 2025"
    assert signal.source_url == "https://ramp.com/blog/ramp-raises-500m"
    assert "$500M" in signal.detail
    assert "Series E" in signal.detail
    assert "$22.5 billion" in signal.detail
    assert signal.recency_note != "undated"


def test_ramp_hiring_signal_is_extracted():
    record = _ramp_record()
    hiring = [s for s in record.signals if s.kind == "hiring"]
    assert hiring
    assert hiring[0].source_url == "https://ramp.com/careers"
    assert "140" in hiring[0].detail


def test_ramp_page_title_never_becomes_the_company_name():
    record = _ramp_record()
    assert record.company_name == "Ramp"
    for furniture in ("About", "Different By Design", "Home", "|"):
        assert furniture not in (record.company_name or "")


def test_ramp_aggregator_never_wins_the_domain():
    record = _ramp_record()
    assert record.domain == "ramp.com"
    assert all("tracxn" not in (s or "") for s in record.sources)


def test_ramp_firmographics_are_parsed():
    record = _ramp_record()
    assert record.founded_year == 2019
    assert record.employee_estimate == 1200
    assert record.employee_range == "1001-5000"
    assert record.hq_city == "New York"
    assert record.hq_country == "United States"
    assert record.industry == "Fintech"
    assert record.sub_industry == "Spend Management"


# --- industry inference: scored, region-weighted, willing to say "unknown" -----


def _industry_of(*blocks: tuple[str, str]) -> tuple[Optional[str], Optional[str]]:
    from lea.llm.stub import _evidence_blocks, infer_industry

    return infer_industry(_evidence_blocks(prompts.build_extraction_context("Acme", list(blocks))))


def test_compliance_title_beats_an_analytics_vendor_in_the_body():
    """The Vanta regression: a title saying compliance must outrank a script vendor."""
    industry, sub = _industry_of(
        (
            "https://acme.com/",
            "<title>Compliance Automation for SOC 2 and ISO 27001</title>\n"
            "Acme automates security compliance so teams get audit-ready fast.\n"
            "Detected technologies: Google Analytics 4, Segment, Mixpanel, Amplitude\n"
            "Powered by Google Analytics and Segment analytics tooling.",
        )
    )
    assert industry == "Security & Compliance"
    assert sub == "Compliance Automation"


def test_a_detected_technologies_line_alone_never_decides_the_industry():
    """`Detected technologies: Google Analytics 4, ...` is inventory, not positioning."""
    assert _industry_of(
        ("https://acme.com/", "Detected technologies: Google Analytics 4, Google Tag Manager, Webflow, HubSpot")
    ) == (None, None)


def test_a_single_incidental_mention_resolves_to_unknown():
    assert _industry_of(("https://acme.com/", "Acme makes software for teams. We use analytics internally.")) == (
        None,
        None,
    )


def test_ramp_like_text_resolves_to_fintech():
    industry, sub = _industry_of(
        ("https://ramp.com/", "Ramp's corporate card and spend management software help companies control expenses.")
    )
    assert industry == "Fintech"
    assert sub == "Spend Management"


def test_an_ambiguous_page_resolves_to_unknown_rather_than_guessing():
    assert _industry_of(
        ("https://acme.com/", "We build payroll software and payment processing tools for small businesses.")
    ) == (None, None)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Acme is the data warehouse for product analytics and business intelligence.", "Data & Analytics"),
        ("Acme is an observability and CI/CD platform for developer teams.", "Developer Tools & Infrastructure"),
        ("Acme is HR software: payroll, HRIS and benefits administration in one.", "HR & Payroll"),
        ("Acme does marketing automation, email marketing and lifecycle marketing.", "Marketing Technology"),
        ("Acme is a telehealth platform with an electronic health record for clinics.", "Healthtech"),
        ("Acme builds warehouse automation robots for supply chain and freight operators.", "Logistics"),
    ],
)
def test_industry_vocabulary_distinguishes_adjacent_categories(text, expected):
    assert _industry_of(("https://acme.com/", text))[0] == expected


def test_industry_needles_match_on_word_boundaries():
    """`hris` must not match inside "Chris", nor `etl` inside "Bartlett"."""
    assert _industry_of(("https://acme.com/careers", "Meet Chris Bartlett, who joined us from a startup.")) == (
        None,
        None,
    )


def test_urls_and_vendor_names_do_not_vote_on_industry():
    assert _industry_of(
        ("https://acme.com/", "Acme makes things.\nhttps://analytics.example.com/tag.js\nSegment and Mixpanel are loaded.")
    ) == (None, None)


def test_ramp_signals_are_deduplicated_not_doubled():
    kinds = [s.kind for s in _ramp_record().signals]
    assert kinds.count("funding") == 1, "the headline and the body describe one round"
    assert kinds.count("hiring") == 1


def test_ramp_tech_stack_is_detected():
    assert any(t.name == "Greenhouse" for t in _ramp_record().tech_stack)


def test_a_hiring_sentence_does_not_inherit_a_founding_year_as_its_date():
    ctx = prompts.build_extraction_context(
        "Acme", [("https://acme.com/about", "Founded in 2011. We're hiring across engineering.")]
    )
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.founded_year == 2011
    for signal in record.signals:
        assert signal.date != "2011", "a founding year is not a publication date"


def _signals_for(text: str, url: str = "https://acme.com", today=None):
    from lea.llm.stub import _evidence_blocks, extract_signals

    block = f"--- BLOCK 1 | source_url: {url} ---\n{text}\n--- END BLOCK 1 ---"
    return extract_signals(_evidence_blocks(block), today=today)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("We raised $12M in a Series A round last spring.", "funding"),
        ("Announcing our Series B, led by Example Capital.", "funding"),
        ("We're hiring! Join our team in three cities.", "hiring"),
        ("Today we are launching our new billing product.", "product"),
        ("Acme acquires Beta Corp to expand into Europe.", "news"),
        ("Acme partners with Globex on payments infrastructure.", "partnership"),
        ("Acme appoints Jane Roe as its new CRO this month.", "leadership"),
        ("Our office has a very nice kitchen and good coffee.", None),
    ],
)
def test_signal_classification(text, expected):
    kinds = [s["kind"] for s in _signals_for(text)]
    if expected is None:
        assert not kinds
    else:
        assert expected in kinds


@pytest.mark.parametrize(
    "published,today,note",
    [
        ("March 3, 2026", date(2026, 3, 20), "within 30 days"),
        ("January 5, 2026", date(2026, 3, 20), "within 90 days"),
        ("July 30, 2025", date(2026, 9, 19), "over a year ago"),
        ("May 1, 2022", date(2026, 9, 19), "over two years ago"),
        ("2025-11-04", date(2026, 1, 2), "within 90 days"),
    ],
)
def test_recency_notes_are_relative_to_today(published, today, note):
    signals = _signals_for(f"Acme raised $5M in new funding. {published}", today=today)
    assert signals
    assert signals[0]["recency_note"] == note


def test_absent_facts_stay_none_and_are_reported_as_unknown():
    ctx = prompts.build_extraction_context(
        "Quietco", [("https://quietco.com/", "Quietco makes things. The site says very little else about it.")]
    )
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.founded_year is None
    assert record.employee_estimate is None
    assert record.employee_range is None
    assert record.hq_city is None
    assert record.hq_country is None
    assert record.signals == []
    for field in ("founded_year", "employee_estimate", "hq_city", "signals"):
        assert field in record.unknowns


def test_domain_is_none_rather_than_wrong_when_nothing_resembles_the_target():
    ctx = prompts.build_extraction_context(
        "Zephyr Dynamics", [("https://www.crunchbase.com/organization/zephyr", "A third-party profile page.")]
    )
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.domain is None


@pytest.mark.parametrize(
    "text,count,band",
    [
        ("The company has 120 employees.", 120, "51-200"),
        ("A team of over 2,500 people build the product.", 2500, "1001-5000"),
        ("We are 8 people working remotely.", 8, "1-10"),
    ],
)
def test_employee_parsing(text, count, band):
    ctx = prompts.build_extraction_context("Acme", [("https://acme.com/", text)])
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.employee_estimate == count
    assert record.employee_range == band


@pytest.mark.parametrize(
    "text,year",
    [
        ("Founded in 1998 by two engineers in a garage.", 1998),
        ("Serving customers since 2014 across Europe.", 2014),
        ("Est. 2021, remote-first and profitable.", 2021),
        ("Order number 1234 shipped to the warehouse.", None),
    ],
)
def test_founded_year_parsing(text, year):
    ctx = prompts.build_extraction_context("Acme", [("https://acme.com/", text)])
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.founded_year == year


def test_hq_from_a_footer_address_line():
    ctx = prompts.build_extraction_context(
        "Acme", [("https://acme.com/", "Acme Inc, 228 Park Ave S, New York, NY 10003. All rights reserved.")]
    )
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.hq_city == "New York"
    assert record.hq_country == "United States"


def test_hq_from_a_rendered_locations_line():
    ctx = prompts.build_extraction_context(
        "Acme", [("https://acme.com/", "locations: Dublin, Ireland; Berlin, Germany")]
    )
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.hq_city == "Dublin"
    assert record.hq_country == "Ireland"


def test_stub_is_deterministic():
    ctx = prompts.build_extraction_context(TARGET, EVIDENCE)
    a, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    b, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert a.model_dump() == b.model_dump()


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def test_factory_falls_back_to_stub_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = get_llm(Settings(llm_provider="auto", anthropic_api_key=None))
    assert client.name == "stub"
    assert client.model == "stub"


def test_factory_falls_back_when_anthropic_forced_without_key(monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level("WARNING"):
        client = get_llm(Settings(llm_provider="anthropic", anthropic_api_key=None))
    assert client.name == "stub"
    assert any("ANTHROPIC_API_KEY" in r.message for r in caplog.records)


def test_factory_honours_explicit_stub():
    assert get_llm(Settings(llm_provider="stub", anthropic_api_key="sk-ant-whatever")).name == "stub"


def test_factory_falls_back_on_unknown_provider(caplog):
    with caplog.at_level("WARNING"):
        client = get_llm(Settings(llm_provider="llama-at-home"))
    assert client.name == "stub"


def test_factory_returns_an_llmclient_shaped_object():
    client = get_llm(Settings(llm_provider="stub"))
    assert callable(client.decide)
    assert callable(client.extract)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_all_named_system_prompts_exist_and_are_substantial():
    for name in ("AGENT_SYSTEM", "EXTRACTION_SYSTEM", "ICP_INFERENCE_SYSTEM", "ICP_SCORING_SYSTEM", "HOOKS_SYSTEM"):
        value = getattr(prompts, name)
        assert isinstance(value, str) and len(value) > 200, name
        assert "{" not in value, f"{name} must not contain unrendered format placeholders"


def test_build_agent_context_renders_and_names_the_target():
    trace = [
        _step(
            1,
            AgentAction(thought="find the site", tool=ToolName.WEB_SEARCH, args={"query": "Acme Robotics company overview"}),
            "3 results",
            ["https://acmerobotics.com/"],
        )
    ]
    text = prompts.build_agent_context(
        target=TARGET,
        record_so_far=CompanyRecord(input=TARGET, company_name="Acme Robotics", industry="Robotics"),
        trace=trace,
        remaining_budget={"steps": 7, "searches": 3, "fetches": 5},
        known_gaps=["employee_range"],
    )
    assert TARGET in text
    assert "web_search" in text
    assert "employee_range" in text
    assert "steps=7" in text
    assert "https://acmerobotics.com/" in text


def test_build_agent_context_handles_empty_state():
    text = prompts.build_agent_context(TARGET)
    assert TARGET in text
    assert "first move" in text


def test_build_agent_context_accepts_dicts_instead_of_models():
    text = prompts.build_agent_context(
        TARGET,
        record_so_far={"company_name": "Acme"},
        trace=[{"n": 1, "tool": "fetch_page", "args": {"url": "https://x.com"}, "observation": "ok"}],
        remaining_budget={"steps": 2},
    )
    assert "Acme" in text
    assert "fetch_page" in text


def test_build_agent_context_renders_failed_steps():
    trace = [TraceStep(n=1, tool=ToolName.FETCH_PAGE, args={"url": "https://x.test"}, ok=False, error="timeout")]
    text = prompts.build_agent_context(TARGET, None, trace, {"steps": 5})
    assert "FAILED" in text
    assert "timeout" in text


def test_build_extraction_context_delimits_and_cites_blocks():
    text = prompts.build_extraction_context(TARGET, EVIDENCE)
    assert TARGET in text
    assert text.count("--- BLOCK") == 2
    assert "source_url: https://acmerobotics.com/about" in text
    assert "warehouse automation" in text


def test_build_extraction_context_renders_structured_tool_metadata():
    text = prompts.build_extraction_context(
        TARGET,
        [
            {
                "url": "https://acme.com/",
                "text": "Acme builds robots.",
                "locations": ["Dublin, Ireland", "Berlin, Germany"],
                "detected": ["Greenhouse", "Segment"],
            }
        ],
    )
    assert "locations: Dublin, Ireland; Berlin, Germany" in text
    assert "detected: Greenhouse; Segment" in text
    assert "Acme builds robots." in text


def test_structured_locations_metadata_feeds_the_stub_extractor():
    ctx = prompts.build_extraction_context(
        "Acme",
        [{"url": "https://acme.com/", "text": "Acme builds robots.", "locations": ["Dublin, Ireland"]}],
    )
    record, _ = StubClient(Settings()).extract(prompts.EXTRACTION_SYSTEM, ctx, CompanyRecord)
    assert record.hq_city == "Dublin"
    assert record.hq_country == "Ireland"


def test_build_extraction_context_without_evidence():
    text = prompts.build_extraction_context(TARGET, [])
    assert TARGET in text
    assert "none" in text.lower()


def test_prompt_renderers_never_raise_on_odd_input():
    assert prompts.build_agent_context("X", object(), None, None, None)
    assert prompts.build_extraction_context("X", [{"url": "https://a.test", "text": "hi"}])


# ---------------------------------------------------------------------------
# AnthropicClient — pure, offline pieces only (no SDK, no network)
# ---------------------------------------------------------------------------


def test_truncate_context_elides_the_middle():
    from lea.llm.anthropic_client import truncate_context

    text = "HEAD" + ("x" * 5000) + "TAIL"
    out = truncate_context(text, budget=500)
    assert len(out) < len(text)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "elided" in out
    assert truncate_context("short", budget=500) == "short"


def test_best_effort_instance_always_returns_an_instance():
    from lea.llm.anthropic_client import best_effort_instance

    rec = best_effort_instance(CompanyRecord)
    assert isinstance(rec, CompanyRecord)

    fit = best_effort_instance(ICPFit, {"score": 9000, "tier": "Z"})  # both invalid
    assert isinstance(fit, ICPFit)

    tiny = best_effort_instance(Tiny, {"name": "ok", "count": "bad"})
    assert isinstance(tiny, Tiny) and tiny.name == "ok"


class _Block:
    def __init__(self, type: str, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    def __init__(self, i=100, o=20, cache_read=5, cache_create=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_create


class _Response:
    def __init__(self, content, model="claude-sonnet-5", usage=None):
        self.content = content
        self.model = model
        self.usage = usage or _Usage()


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0) if self._responses else _Response([])
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, *responses):
        self.messages = _FakeMessages(responses)


def _anthropic_client(*responses, model="claude-sonnet-5"):
    from lea.llm.anthropic_client import AnthropicClient

    fake = _FakeClient(*responses)
    settings = Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test", model=model)
    return AnthropicClient(settings, client=fake), fake


def _tool_use(name, payload):
    return _Block("tool_use", name=name, id="tu_1", input=payload)


def test_anthropic_decide_parses_a_forced_tool_call():
    client, fake = _anthropic_client(
        _Response([_tool_use("next_action", {"thought": "need the site", "tool": "web_search", "args": {"query": "acme"}})])
    )
    action, usage = client.decide(prompts.AGENT_SYSTEM, "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert action.args == {"query": "acme"}
    assert action.thought == "need the site"
    assert usage.tokens_in == 105  # input + cache_read + cache_creation
    assert usage.tokens_out == 20

    sent = fake.messages.calls[0]
    assert sent["tool_choice"] == {"type": "tool", "name": "next_action"}
    assert sent["tools"][0]["name"] == "next_action"
    schema = sent["tools"][0]["input_schema"]
    assert schema["properties"]["tool"]["enum"] == ["web_search", "fetch_page", "detect_tech", "finish"]
    assert set(schema["required"]) == {"thought", "tool", "args"}


def test_anthropic_decide_omits_sampling_params_on_models_that_reject_them():
    client, fake = _anthropic_client(
        _Response([_tool_use("next_action", {"thought": "t", "tool": "finish", "args": {}})]),
        model="claude-sonnet-5",
    )
    client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    sent = fake.messages.calls[0]
    assert "temperature" not in sent
    assert sent["output_config"]["effort"] == "medium"


def test_anthropic_uses_low_temperature_on_models_that_accept_it():
    client, fake = _anthropic_client(
        _Response([_tool_use("record_companyrecord", {"input": "Acme"})]),
        model="claude-sonnet-4-6",
    )
    client.extract("sys", "ctx", CompanyRecord)
    sent = fake.messages.calls[0]
    assert sent["temperature"] == 0.0
    assert "output_config" not in sent


def test_anthropic_decide_coerces_an_unavailable_tool():
    client, _ = _anthropic_client(
        _Response([_tool_use("next_action", {"thought": "search it", "tool": "websearch", "args": {"query": "x"}})])
    )
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH
    assert "coerced" in action.thought


def test_anthropic_decide_returns_finish_on_api_error():
    client, _ = _anthropic_client(RuntimeError("connection reset"))
    action, usage = client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.FINISH
    assert action.thought.startswith("LLM error:")
    assert usage.tokens_in == 0


def test_anthropic_decide_salvages_json_from_prose():
    client, _ = _anthropic_client(
        _Response([_Block("text", text='{"thought": "t", "tool": "finish", "args": {}}')])
    )
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.FINISH


def test_anthropic_extract_uses_the_model_json_schema():
    payload = {"input": "Acme", "company_name": "Acme Robotics", "founded_year": 2019}
    client, fake = _anthropic_client(_Response([_tool_use("record_companyrecord", payload)]))
    record, usage = client.extract(prompts.EXTRACTION_SYSTEM, "ctx", CompanyRecord, hint="focus on firmographics")

    assert isinstance(record, CompanyRecord)
    assert record.company_name == "Acme Robotics"
    assert record.founded_year == 2019
    assert usage.tokens_out == 20

    sent = fake.messages.calls[0]
    assert sent["tool_choice"]["name"] == "record_companyrecord"
    assert "company_name" in sent["tools"][0]["input_schema"]["properties"]
    assert "focus on firmographics" in sent["messages"][0]["content"]


def test_anthropic_extract_does_one_repair_retry_then_succeeds():
    client, fake = _anthropic_client(
        _Response([_tool_use("record_icpfit", {"score": 900, "tier": "Z"})]),          # invalid
        _Response([_tool_use("record_icpfit", {"score": 72, "tier": "B", "summary": "ok"})]),
    )
    fit, usage = client.extract(prompts.ICP_SCORING_SYSTEM, "ctx", ICPFit)

    assert isinstance(fit, ICPFit) and fit.score == 72 and fit.tier == "B"
    assert len(fake.messages.calls) == 2
    repair_text = fake.messages.calls[1]["messages"][-1]["content"]
    assert "did not validate" in repair_text
    assert "score" in repair_text
    assert usage.tokens_in == 210  # both calls accumulated


def test_anthropic_extract_falls_back_without_raising():
    client, fake = _anthropic_client(
        _Response([_tool_use("record_icpfit", {"score": 900})]),
        _Response([_tool_use("record_icpfit", {"score": 900})]),
    )
    fit, _ = client.extract("sys", "ctx", ICPFit)
    assert isinstance(fit, ICPFit)
    assert len(fake.messages.calls) == 2


def test_anthropic_extract_falls_back_when_the_api_dies():
    client, _ = _anthropic_client(RuntimeError("boom"))
    record, usage = client.extract("sys", "ctx", CompanyRecord)
    assert isinstance(record, CompanyRecord)
    assert usage.tokens_in == 0


def test_anthropic_retries_minimally_after_a_bad_request():
    class BadRequestError(Exception):
        status_code = 400

    client, fake = _anthropic_client(
        BadRequestError("output_config: unexpected"),
        _Response([_tool_use("next_action", {"thought": "t", "tool": "finish", "args": {}})]),
    )
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.FINISH
    assert len(fake.messages.calls) == 2
    assert "output_config" not in fake.messages.calls[1]
    assert fake.messages.calls[1]["tool_choice"] == {"type": "auto"}


def test_anthropic_truncates_a_huge_context():
    client, fake = _anthropic_client(
        _Response([_tool_use("next_action", {"thought": "t", "tool": "finish", "args": {}})])
    )
    client.decide("sys", "TARGET: Acme\n" + ("filler " * 20000), TOOL_SPECS)
    sent_len = len(fake.messages.calls[0]["messages"][0]["content"])
    assert sent_len < 70_000
    assert "elided" in fake.messages.calls[0]["messages"][0]["content"]


def test_llm_scope_does_not_import_agent_enrich_or_cli():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "lea"
    owned = [
        root / "llm" / "anthropic_client.py",
        root / "llm" / "stub.py",
        root / "llm" / "factory.py",
        root / "llm" / "json_utils.py",
        root / "llm" / "openai_compat.py",
        root / "prompts.py",
    ]
    import ast

    for path in owned:
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for module in imported:
            for forbidden in ("lea.agent", "lea.enrich", "lea.cli"):
                assert not module.startswith(forbidden), f"{path.name} must not import {forbidden}"


# ---------------------------------------------------------------------------
# OpenAICompatClient — free providers. Network fully mocked with respx.
# ---------------------------------------------------------------------------

import respx  # noqa: E402

from lea.llm.openai_compat import (  # noqa: E402
    AUTH_BREAKER_THRESHOLD,
    MAX_ATTEMPTS,
    Capability,
    OpenAICompatClient,
    classify_auth_failure,
    rank_models,
    resolve_model,
    list_models,
    reset_provider_state,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


@pytest.fixture(autouse=True)
def _clean_provider_state():
    """Rung cache and auth counters are process-global: isolate every test."""
    reset_provider_state()
    yield
    reset_provider_state()


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Retry backoff must not make the suite slow.

    Patch the client's own hook rather than `time.sleep` — the module-level
    `time` object is the global one, and neutering it would reach far outside
    these tests.
    """
    monkeypatch.setattr(OpenAICompatClient, "_sleep", staticmethod(lambda attempt, retry_after: None))


def _oa_client(provider: str = "groq", **kw) -> OpenAICompatClient:
    settings = Settings(llm_provider=provider, llm_api_key=kw.pop("key", "test-key"), **kw)
    return OpenAICompatClient(settings, provider=provider)


def _tool_response(name: str, arguments: dict, prompt_tokens: int = 120, completion_tokens: int = 30) -> dict:
    """OpenAI shape: function.arguments is a JSON *string*, not an object."""
    return {
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _text_response(text: str, prompt_tokens: int = 100, completion_tokens: int = 25) -> dict:
    return {
        "model": "llama-3.3-70b-versatile",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _unsupported_tools(request) -> httpx.Response:
    return httpx.Response(
        400, json={"error": {"message": "tool_choice is not supported for this model", "type": "invalid_request_error"}}
    )


DECIDE_ARGS = {"thought": "need the official site", "tool": "web_search", "args": {"query": "Acme Robotics"}}


# --- rung a: forced function call ------------------------------------------


@respx.mock
def test_forced_function_call_is_parsed_into_an_agent_action():
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)))
    client = _oa_client()
    action, usage = client.decide(prompts.AGENT_SYSTEM, "TARGET: Acme Robotics", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert action.args == {"query": "Acme Robotics"}
    assert action.thought == "need the official site"
    assert usage.tokens_in == 120 and usage.tokens_out == 30

    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "next_action"}}
    fn = sent["tools"][0]["function"]
    assert fn["name"] == "next_action"
    assert fn["parameters"]["properties"]["tool"]["enum"] == ["web_search", "fetch_page", "detect_tech", "finish"]
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"


@respx.mock
def test_out_of_range_tool_name_is_coerced_and_noted():
    payload = {"thought": "search", "tool": "websearch", "args": {"query": "x"}}
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_tool_response("next_action", payload)))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH
    assert "coerced" in action.thought


# --- rung b: json_object ----------------------------------------------------


@respx.mock
def test_tools_unsupported_falls_back_to_json_object_mode():
    route = respx.post(GROQ_URL).mock(
        side_effect=[_unsupported_tools, httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS)))]
    )
    client = _oa_client()
    action, usage = client.decide("sys", "TARGET: Acme Robotics", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH, "a tools-less model must still produce a valid action"
    assert action.args == {"query": "Acme Robotics"}
    assert route.call_count == 2

    second = json.loads(route.calls[1].request.content)
    assert second["response_format"] == {"type": "json_object"}
    assert "tools" not in second and "tool_choice" not in second
    # The schema has to travel in the prompt now that nothing enforces it.
    assert "thought" in second["messages"][-1]["content"]
    assert client.capability == Capability.JSON_OBJECT


@respx.mock
def test_a_200_that_ignores_tool_choice_also_demotes():
    """Some models accept `tools` and then answer in prose anyway."""
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(200, json=_text_response("I think we should search.")),
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),
        ]
    )
    action, _ = _oa_client().decide("sys", "TARGET: Acme Robotics", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH
    assert route.call_count == 2


# --- rung c: prose + repair -------------------------------------------------


@respx.mock
def test_json_object_unsupported_falls_back_to_prose_and_tolerant_parsing():
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            _unsupported_tools,
            lambda r: httpx.Response(400, json={"error": {"message": "response_format is not supported"}}),
            httpx.Response(
                200,
                json=_text_response(
                    "Sure! Here is my decision:\n```json\n"
                    '{"thought": "t", "tool": "fetch_page", "args": {"url": "https://acme.com",},}\n```'
                ),
            ),
        ]
    )
    client = _oa_client()
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FETCH_PAGE, "fenced, trailing-comma JSON must still parse"
    assert action.args == {"url": "https://acme.com"}
    assert route.call_count == 3
    assert client.capability == Capability.PROSE

    last = json.loads(route.calls[2].request.content)
    assert "tools" not in last and "response_format" not in last


# --- capability caching -----------------------------------------------------


@respx.mock
def test_the_working_rung_is_cached_and_not_reprobed():
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            _unsupported_tools,
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),
        ]
    )
    client = _oa_client()
    for _ in range(3):
        action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)
        assert action.tool is ToolName.WEB_SEARCH

    assert route.call_count == 4, "the tools probe must be paid once, not once per call"
    for call in route.calls[1:]:
        assert "tools" not in json.loads(call.request.content)


@respx.mock
def test_an_endpoint_that_starts_rejecting_tools_is_demoted_and_stays_demoted():
    """A cached TOOLS rung must be overwritten, not merely left in place."""
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),  # works: cache TOOLS
            _unsupported_tools,                                                    # provider changes its mind
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),     # json_object
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),     # must stay demoted
        ]
    )
    client = _oa_client()

    client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert client.capability == Capability.TOOLS

    client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert client.capability == Capability.JSON_OBJECT, "the cached rung must be overwritten"

    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH
    assert route.call_count == 4
    assert "tools" not in json.loads(route.calls[3].request.content)


@respx.mock
def test_the_capability_cache_is_shared_across_clients_for_one_endpoint():
    respx.post(GROQ_URL).mock(
        side_effect=[
            _unsupported_tools,
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),
        ]
    )
    _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    second = _oa_client()
    assert second.capability == Capability.JSON_OBJECT
    action, _ = second.decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH


# --- rate limiting and transport errors -------------------------------------


@respx.mock
def test_429_with_retry_after_is_retried_then_succeeds():
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}, json={"error": {"message": "rate limited"}}),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH
    assert route.call_count == 2


@respx.mock
def test_persistent_429_degrades_to_finish_without_raising():
    route = respx.post(GROQ_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "1"}, json={"error": {"message": "quota"}})
    )
    action, usage = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert action.thought.startswith("LLM error:")
    assert usage.tokens_in == 0
    assert route.call_count == 4  # MAX_ATTEMPTS, then degrade


@respx.mock
def test_server_errors_are_retried_then_degrade():
    respx.post(GROQ_URL).mock(return_value=httpx.Response(503, text="upstream unavailable"))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.FINISH


@respx.mock
def test_connection_errors_degrade_cleanly():
    respx.post(GROQ_URL).mock(side_effect=httpx.ConnectError("no route to host"))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.FINISH
    assert "LLM error" in action.thought


# --- auth: intermittent 403s from a half-provisioned key ---------------------
#
# Measured live on the user's Google key: 8 identical requests -> 3x 403, 5x 200.
# A 403 that is really a provisioning race must not fail the step.

GOOGLE_403_PERMISSION = {
    "error": {"code": 403, "message": "The caller does not have permission", "status": "PERMISSION_DENIED"}
}
GOOGLE_403_DISABLED = {
    "error": {
        "code": 403,
        "message": (
            "Gemini API has not been used in project 1039824369945 before or it is disabled. "
            "Enable it by visiting https://console.developers.google.com/apis/api/..."
        ),
        "status": "PERMISSION_DENIED",
    }
}
TERMINAL_403 = {"error": {"code": 403, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}


@pytest.mark.parametrize(
    "status,message,expected",
    [
        (403, "The caller does not have permission", "transient"),
        (403, "Gemini API has not been used in project 1039824369945 before or it is disabled.", "transient"),
        (403, "The service is not enabled for this project", "transient"),
        (403, "Changes may take a few minutes to propagate", "transient"),
        (403, "API key not valid. Please pass a valid API key.", "terminal"),
        (403, "Permission denied for model gemini-3.6-flash", "terminal"),
        (403, "This API key has been revoked", "terminal"),
        (401, "Invalid authentication credentials", "terminal"),
        (401, "something we have never seen", "terminal"),
        (403, "something we have never seen", "transient"),  # when in doubt, retry
    ],
)
def test_auth_failures_are_classified(status, message, expected):
    body = json.dumps({"error": {"code": status, "message": message}})
    assert classify_auth_failure(status, body) == expected


@respx.mock
def test_a_transient_403_is_retried_and_succeeds():
    """The exact live failure: 403, 403, then a 200 with the same request."""
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(403, json=GOOGLE_403_PERMISSION),
            httpx.Response(403, json=GOOGLE_403_DISABLED),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    action, usage = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH, "an intermittent 403 must not lose the step"
    assert action.args == {"query": "Acme Robotics"}
    assert usage.tokens_in == 120
    assert route.call_count == 3


@respx.mock
def test_a_transient_403_lets_extract_keep_the_record():
    """The regression that cost industry, HQ, founded_year and every signal."""
    payload = {"input": "Vanta", "company_name": "Vanta", "industry": "Security & Compliance", "founded_year": 2018}
    respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(403, json=GOOGLE_403_PERMISSION),
            httpx.Response(200, json=_tool_response("record_companyrecord", payload)),
        ]
    )
    record, _ = _oa_client().extract(prompts.EXTRACTION_SYSTEM, "ctx", CompanyRecord)
    assert record.industry == "Security & Compliance"
    assert record.founded_year == 2018


@respx.mock
def test_a_terminal_403_fails_fast_without_burning_attempts():
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(403, json=TERMINAL_403))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert route.call_count == 1, "a revoked key must not cost four attempts on every call"


@respx.mock
def test_a_persistent_transient_403_exhausts_attempts_then_degrades():
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(403, json=GOOGLE_403_PERMISSION))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert route.call_count == MAX_ATTEMPTS


@respx.mock
def test_the_auth_breaker_stops_retrying_after_repeated_failures():
    """A dead key must not turn a 12-step run into minutes of backoff."""
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(403, json=GOOGLE_403_PERMISSION))
    client = _oa_client()

    for _ in range(AUTH_BREAKER_THRESHOLD):
        assert client.decide("sys", "TARGET: Acme", TOOL_SPECS)[0].tool is ToolName.FINISH
    spent = route.call_count
    assert spent == AUTH_BREAKER_THRESHOLD * MAX_ATTEMPTS

    # Breaker is now open: every later step costs exactly one request.
    for _ in range(5):
        action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)
        assert action.tool is ToolName.FINISH
    assert route.call_count == spent + 5


@respx.mock
def test_a_success_resets_the_auth_breaker():
    respx.post(GROQ_URL).mock(
        side_effect=[
            *[httpx.Response(403, json=GOOGLE_403_PERMISSION)] * MAX_ATTEMPTS,
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
            httpx.Response(403, json=GOOGLE_403_PERMISSION),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    client = _oa_client()
    assert client.decide("sys", "T: A", TOOL_SPECS)[0].tool is ToolName.FINISH  # 1 failure recorded
    assert client.decide("sys", "T: A", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH  # clears it
    # Still retrying, because the counter was reset by the success.
    assert client.decide("sys", "T: A", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH


@respx.mock
@pytest.mark.parametrize(
    "body,expected_word",
    [(GOOGLE_403_PERMISSION, "transient"), (TERMINAL_403, "wrong")],
)
def test_the_trace_text_says_which_kind_of_auth_failure_it_was(body, expected_word):
    respx.post(GROQ_URL).mock(return_value=httpx.Response(403, json=body))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert expected_word in action.thought.lower()
    assert "provisioning" in action.thought or "API key" in action.thought


@respx.mock
def test_auth_error_text_is_one_clean_line_without_json_braces():
    respx.post(GROQ_URL).mock(return_value=httpx.Response(403, json=GOOGLE_403_DISABLED))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    thought = action.thought
    assert "\n" not in thought, "a multi-line error breaks the rendered trace"
    assert "{" not in thought and "}" not in thought, "the raw JSON payload must not reach the trace"
    assert "has not been used in project" in thought, "but the useful part must survive"
    assert len(thought) < 400


@respx.mock
def test_google_array_wrapped_error_bodies_are_unwrapped():
    """Google's OpenAI-compat layer returns `[{"error": {...}}]`, not `{...}`.

    Seen live: an object-only parse leaked `[{ "error": { "code": 401, "mess`
    into the middle of the rendered trace.
    """
    body = [{"error": {"code": 401, "message": "API key not valid. Please pass a valid API key.", "status": "UNAUTHENTICATED"}}]
    respx.post(GROQ_URL).mock(return_value=httpx.Response(401, json=body))
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert "API key not valid" in action.thought
    assert "{" not in action.thought and "[" not in action.thought
    assert "code" not in action.thought


def test_error_message_handles_every_body_shape():
    from lea.llm.openai_compat import _error_message

    assert _error_message('{"error": {"message": "plain object"}}') == "plain object"
    assert _error_message('[{"error": {"message": "array wrapped"}}]') == "array wrapped"
    assert _error_message('{"message": "bare message"}') == "bare message"
    assert _error_message('{"error": "string error"}') == "string error"
    assert _error_message("not json at all") == "not json at all"
    assert _error_message("") == ""
    assert "\n" not in _error_message('{"error": {"message": "one\\ntwo\\nthree"}}')
    assert len(_error_message('{"error": {"message": "' + "x" * 500 + '"}}')) <= 200


@respx.mock
def test_http_error_text_is_trimmed_too():
    respx.post(GROQ_URL).mock(
        return_value=httpx.Response(500, json={"error": {"message": "internal\nerror\nwith newlines"}})
    )
    action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert "\n" not in action.thought
    assert "{" not in action.thought


# --- the failure path must stay silent on stdout/stderr ----------------------


@respx.mock
def test_no_bare_prints_on_the_failure_path(capsys):
    """Anything written directly to stdout/stderr interleaves with the rich trace."""
    respx.post(GROQ_URL).mock(return_value=httpx.Response(403, json=GOOGLE_403_PERMISSION))
    _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    _oa_client().extract("sys", "ctx", CompanyRecord)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@respx.mock
def test_per_attempt_retry_noise_is_debug_not_warning(caplog):
    """Retry chatter at WARNING lands on stderr mid-trace, because the CLI
    installs no logging handler and Python's lastResort handler kicks in."""
    respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(503, text="upstream unavailable"),
            httpx.Response(429, headers={"Retry-After": "1"}, text="slow down"),
            httpx.Response(403, json=GOOGLE_403_PERMISSION),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    with caplog.at_level("DEBUG", logger="lea.llm.openai_compat"):
        action, _ = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f"a recovered request must stay quiet, got: {[r.message for r in warnings]}"
    assert any("attempt" in r.getMessage() for r in caplog.records), "the detail still belongs at DEBUG"


@respx.mock
def test_a_step_that_ultimately_fails_logs_exactly_one_warning_line(caplog):
    respx.post(GROQ_URL).mock(return_value=httpx.Response(500, text="boom"))
    with caplog.at_level("WARNING", logger="lea.llm.openai_compat"):
        _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"one failed step, one line — got {[r.message for r in warnings]}"
    assert "\n" not in warnings[0].getMessage()


@respx.mock
def test_a_failed_auth_step_also_logs_exactly_one_warning_line(caplog):
    respx.post(GROQ_URL).mock(return_value=httpx.Response(403, json=GOOGLE_403_PERMISSION))
    with caplog.at_level("WARNING", logger="lea.llm.openai_compat"):
        _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"auth failures must not log twice — got {[r.message for r in warnings]}"
    assert "transient" in warnings[0].getMessage()


# --- extract ----------------------------------------------------------------


@respx.mock
def test_extract_returns_a_valid_company_record():
    payload = {"input": "Acme", "company_name": "Acme Robotics", "founded_year": 2019, "hq_city": "Austin"}
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_tool_response("record_companyrecord", payload)))
    record, usage = _oa_client().extract(prompts.EXTRACTION_SYSTEM, "ctx", CompanyRecord, hint="firmographics")

    assert isinstance(record, CompanyRecord)
    assert record.company_name == "Acme Robotics"
    assert record.founded_year == 2019
    assert usage.tokens_in == 120

    sent = json.loads(route.calls[0].request.content)
    fn = sent["tools"][0]["function"]
    assert fn["name"] == "record_companyrecord"
    assert "company_name" in fn["parameters"]["properties"]
    assert "firmographics" in sent["messages"][-1]["content"]


@respx.mock
def test_extract_returns_a_valid_icp_fit():
    payload = {"score": 74, "tier": "B", "summary": "decent fit"}
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_tool_response("record_icpfit", payload)))
    fit, _ = _oa_client().extract(prompts.ICP_SCORING_SYSTEM, "ctx", ICPFit)
    assert isinstance(fit, ICPFit) and fit.score == 74 and fit.tier == "B"


@respx.mock
def test_extract_does_one_repair_retry_then_succeeds():
    route = respx.post(GROQ_URL).mock(
        side_effect=[
            httpx.Response(200, json=_tool_response("record_icpfit", {"score": 900, "tier": "Z"})),
            httpx.Response(200, json=_tool_response("record_icpfit", {"score": 61, "tier": "C"})),
        ]
    )
    fit, usage = _oa_client().extract("sys", "ctx", ICPFit)

    assert isinstance(fit, ICPFit) and fit.score == 61
    assert route.call_count == 2
    repair_text = json.loads(route.calls[1].request.content)["messages"][-1]["content"]
    assert "did not validate" in repair_text
    assert "score" in repair_text
    assert usage.tokens_in == 240  # both calls accumulated


@respx.mock
def test_truncated_json_degrades_to_a_best_effort_instance_without_raising():
    route = respx.post(GROQ_URL).mock(
        return_value=httpx.Response(200, json=_text_response('{"input": "Acme", "company_name": "Acme Rob'))
    )
    record, _ = _oa_client().extract("sys", "ctx", CompanyRecord)
    assert isinstance(record, CompanyRecord)
    assert route.call_count >= 1


@respx.mock
def test_extract_never_raises_when_everything_fails():
    respx.post(GROQ_URL).mock(return_value=httpx.Response(500, text="boom"))
    record, usage = _oa_client().extract("sys", "ctx", CompanyRecord)
    assert isinstance(record, CompanyRecord)
    assert usage.tokens_in == 0


@respx.mock
def test_extract_salvages_prose_wrapped_json():
    body = 'Here is the record:\n```json\n{"input": "Acme", "company_name": "Acme Robotics"}\n```'
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_text_response(body)))
    record, _ = _oa_client().extract("sys", "ctx", CompanyRecord)
    assert record.company_name == "Acme Robotics"


@respx.mock
def test_usage_is_zero_when_the_provider_omits_it():
    body = {"choices": [{"message": {"content": json.dumps(DECIDE_ARGS)}}]}
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=body))
    _, usage = _oa_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert usage.tokens_in == 0 and usage.tokens_out == 0


# --- ollama: no key, no Authorization header --------------------------------


@respx.mock
def test_ollama_sends_no_authorization_header():
    url = "http://localhost:11434/v1/chat/completions"
    route = respx.post(url).mock(return_value=httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)))
    client = OpenAICompatClient(Settings(llm_provider="ollama"), provider="ollama")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert "authorization" not in route.calls[0].request.headers
    assert client.api_key is None


# --- model retirement: the adapter repairs a dead model id -------------------
#
# Two ids died inside a single session (gemini-2.0-flash, then gemini-2.5-flash),
# so this is tested against the real list the user's endpoint returned rather
# than invented ids.

LIVE_GOOGLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-image",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-live-preview",
    "gemini-3.1-flash-tts-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-transcribe",
    "gemini-omni-flash-preview",
    "gemini-pro-latest",
    "gemma-4-31b-it",
    "veo-3.1-fast-generate-preview",
]

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GOOGLE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/openai/models"


def _dead_model_404(message: str) -> httpx.Response:
    return httpx.Response(404, json={"error": {"code": 404, "message": message, "status": "NOT_FOUND"}})


def _google_client(model: str = "gemini-2.0-flash") -> OpenAICompatClient:
    return OpenAICompatClient(Settings(llm_provider="google", llm_api_key="k", model=model), provider="google")


def _models_route(ids=None):
    return respx.get(GOOGLE_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": f"models/{m}"} for m in (ids or LIVE_GOOGLE_MODELS)]})
    )


def test_ranking_picks_the_newest_plain_flash_from_the_live_list():
    assert rank_models(LIVE_GOOGLE_MODELS, provider="google", configured="gemini-2.0-flash")[0] == "gemini-3.5-flash"


def test_ranking_prefers_flash_then_lite_then_preview():
    assert rank_models(LIVE_GOOGLE_MODELS, provider="google", configured="gemini-2.0-flash")[:7] == [
        "gemini-3.5-flash",        # plain flash, newest
        "gemini-2.5-flash",        # plain flash, older
        "gemini-3.5-flash-lite",   # lite, newest
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",  # preview last
        "gemini-omni-flash-preview",
    ]


@pytest.mark.parametrize(
    "decoy",
    [
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-flash-native-audio-latest",
        "gemini-3.1-flash-live-preview",
        "gemini-3.1-flash-tts-preview",
        "gemini-3.5-transcribe",
        "gemini-3-pro-image",
        "veo-3.1-fast-generate-preview",
    ],
)
def test_ranking_excludes_non_text_decoys(decoy):
    """`flash-image` and `flash-tts` rank high on a naive substring match."""
    assert decoy not in rank_models(LIVE_GOOGLE_MODELS, provider="google", configured="gemini-2.0-flash")


def test_ranking_prefers_the_configured_model_when_it_is_actually_available():
    ranked = rank_models(LIVE_GOOGLE_MODELS, provider="google", configured="gemini-2.5-flash-lite")
    assert ranked[0] == "gemini-2.5-flash-lite", "never override a model that works"


def test_ranking_higher_version_wins():
    assert rank_models(["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3-flash"], provider="google")[0] == "gemini-3.5-flash"


def test_ranking_falls_back_to_generic_order_for_other_providers():
    groq = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "whisper-large-v3", "llama-guard-4-12b"]
    ranked = rank_models(groq, provider="groq", configured="llama-3.3-70b-versatile")
    assert ranked[0] == "llama-3.3-70b-versatile"
    assert "llama-guard-4-12b" not in ranked  # a safety classifier cannot run an agent


def test_ranking_handles_an_empty_or_useless_list():
    assert rank_models([], provider="google") == []
    assert rank_models(["gemini-2.5-flash-image"], provider="google") == []


@respx.mock
def test_a_404_naming_a_replacement_uses_that_replacement():
    """Google's 404 literally says 'please update to models/gemini-3.6-flash'."""
    models = _models_route()
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404(
                "models/gemini-2.5-flash is not found for API version v1beta. This model is "
                "no longer available to new users, please update to models/gemini-3.6-flash"
            ),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    client = _google_client("gemini-2.5-flash")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert client.model == "gemini-3.6-flash"
    assert json.loads(route.calls[1].request.content)["model"] == "gemini-3.6-flash"
    assert not models.called, "the provider named the successor — no need to list models"


@respx.mock
def test_a_404_without_a_replacement_falls_back_to_the_models_endpoint():
    models = _models_route()
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404("models/gemini-2.0-flash is not found for API version v1beta"),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    client = _google_client("gemini-2.0-flash")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert client.model == "gemini-3.5-flash"
    assert models.call_count == 1
    assert json.loads(route.calls[1].request.content)["model"] == "gemini-3.5-flash"


@respx.mock
def test_model_resolution_is_cached_for_the_whole_run():
    models = _models_route()
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404("models/gemini-2.0-flash is not found for API version v1beta"),
            *[httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)) for _ in range(5)],
        ]
    )
    client = _google_client("gemini-2.0-flash")
    for _ in range(5):
        assert client.decide("sys", "TARGET: Acme", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH

    assert models.call_count == 1, "one /models lookup per run, not one per call"
    assert route.call_count == 6  # the dead first attempt, then five good calls
    for call in route.calls[1:]:
        assert json.loads(call.request.content)["model"] == "gemini-3.5-flash"


@respx.mock
def test_a_second_client_reuses_the_resolved_model_without_probing():
    models = _models_route()
    respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404("models/gemini-2.0-flash is not found for API version v1beta"),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    _google_client("gemini-2.0-flash").decide("sys", "TARGET: Acme", TOOL_SPECS)

    fresh = _google_client("gemini-2.0-flash")
    assert fresh.model == "gemini-3.5-flash", "a new client starts from the repaired id"
    assert fresh.decide("sys", "TARGET: Acme", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH
    assert models.call_count == 1


@respx.mock
def test_the_substitution_logs_exactly_one_warning(caplog):
    _models_route()
    respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404("models/gemini-2.0-flash is not found for API version v1beta"),
            *[httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)) for _ in range(3)],
        ]
    )
    client = _google_client("gemini-2.0-flash")
    with caplog.at_level("WARNING", logger="lea.llm.openai_compat"):
        for _ in range(3):
            client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"tell the user once, not per call: {[r.message for r in warnings]}"
    text = warnings[0].getMessage()
    assert "gemini-2.0-flash" in text and "gemini-3.5-flash" in text
    assert "LEA_MODEL" in text


@respx.mock
def test_an_unresolvable_model_degrades_with_the_doctor_hint():
    _models_route(["gemini-2.5-flash-image", "veo-3.1-fast-generate-preview"])  # nothing usable
    route = respx.post(GOOGLE_URL).mock(
        return_value=_dead_model_404("models/gemini-2.0-flash is not found for API version v1beta")
    )
    action, _ = _google_client("gemini-2.0-flash").decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert "lea doctor --models" in action.thought
    assert "LEA_MODEL" in action.thought
    assert route.call_count == 1, "a dead model must not be retried four times"


@respx.mock
def test_a_replacement_that_is_also_dead_degrades_instead_of_looping():
    """Guards the resolution loop: never re-probe, never chain substitutions."""
    models = _models_route()
    route = respx.post(GOOGLE_URL).mock(
        return_value=_dead_model_404("this model is not found for API version v1beta")
    )
    action, _ = _google_client("gemini-2.0-flash").decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert "lea doctor --models" in action.thought
    assert models.call_count == 1, "one lookup, even when the replacement is dead too"
    assert route.call_count == 2, "the original call plus exactly one substitution attempt"


@respx.mock
def test_a_working_model_never_triggers_resolution():
    models = _models_route()
    respx.post(GOOGLE_URL).mock(return_value=httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)))
    client = _google_client("gemini-2.0-flash")
    for _ in range(3):
        client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert not models.called, "do not go shopping for models while the configured one works"
    assert client.model == "gemini-2.0-flash"


@respx.mock
def test_a_capability_404_still_demotes_instead_of_swapping_the_model():
    """A genuine tools-unsupported error must NOT be read as a dead model."""
    models = _models_route()
    respx.post(GOOGLE_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "tool_choice is not supported for this model"}}),
            httpx.Response(200, json=_text_response(json.dumps(DECIDE_ARGS))),
        ]
    )
    client = _google_client("gemini-2.0-flash")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert client.capability == Capability.JSON_OBJECT
    assert client.model == "gemini-2.0-flash", "a capability gap is not a dead model"
    assert not models.called


@respx.mock
def test_a_dead_model_404_that_also_says_not_supported_resolves_the_model():
    """The exact live Google wording — 'is not found ... or is not supported' —
    which the capability probe would otherwise swallow."""
    _models_route()
    respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404(
                "models/gemini-2.0-flash is not found for API version v1beta, "
                "or is not supported for generateContent."
            ),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    client = _google_client("gemini-2.0-flash")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert client.model == "gemini-3.5-flash"
    assert client.capability == Capability.TOOLS, "the ladder must not be demoted by a dead model"


@respx.mock
def test_extract_also_recovers_from_a_retired_model():
    _models_route()
    payload = {"input": "Vanta", "company_name": "Vanta", "industry": "Security & Compliance"}
    respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404("models/gemini-2.0-flash is not found for API version v1beta"),
            httpx.Response(200, json=_tool_response("record_companyrecord", payload)),
        ]
    )
    record, _ = _google_client("gemini-2.0-flash").extract("sys", "ctx", CompanyRecord)
    assert record.company_name == "Vanta"
    assert record.industry == "Security & Compliance"


@pytest.mark.parametrize(
    "message,expected",
    [
        ("please update to models/gemini-3.6-flash", "gemini-3.6-flash"),
        ("Update your code to use gemini-3.5-flash instead", "gemini-3.5-flash"),
        ("This model has been replaced by models/gemini-3.5-flash.", "gemini-3.5-flash"),
        ("use `llama-3.3-70b-versatile` instead", "llama-3.3-70b-versatile"),
        ("models/gemini-2.0-flash is not found for API version v1beta", None),
        ("please update to models/gemini-3.5-flash-image", None),  # a decoy suggestion
    ],
)
def test_replacement_extraction_from_error_bodies(message, expected):
    from lea.llm.openai_compat import _replacement_from_error

    body = json.dumps({"error": {"code": 404, "message": message}})
    assert _replacement_from_error(body, exclude="gemini-2.0-flash") == expected


@respx.mock
def test_resolve_model_is_exposed_for_doctor():
    models = _models_route()
    settings = Settings(llm_provider="google", llm_api_key="k", model="gemini-2.0-flash")

    assert resolve_model(settings, provider="google") == "gemini-3.5-flash"
    assert resolve_model(settings, provider="google") == "gemini-3.5-flash"
    assert models.call_count == 1, "doctor and the agent share one lookup"


@respx.mock
def test_resolve_model_returns_the_configured_id_when_it_is_served():
    _models_route()
    settings = Settings(llm_provider="google", llm_api_key="k", model="gemini-3.5-flash-lite")
    assert resolve_model(settings, provider="google") == "gemini-3.5-flash-lite"


@respx.mock
def test_resolve_model_returns_none_when_nothing_is_available():
    respx.get(GOOGLE_MODELS_URL).mock(return_value=httpx.Response(500, text="down"))
    settings = Settings(llm_provider="google", llm_api_key="k", model="gemini-2.0-flash")
    assert resolve_model(settings, provider="google") is None


def test_reset_provider_state_clears_the_model_cache():
    from lea.llm.openai_compat import _MODEL_RESOLUTION

    _MODEL_RESOLUTION[("https://x/v1", "old")] = "new"
    reset_provider_state()
    assert _MODEL_RESOLUTION == {}


# --- auth transport negotiation ---------------------------------------------
#
# Measured against Google's live endpoint: the OpenAI-compat path 400s
# x-goog-api-key and ?key= with "Missing or invalid Authorization header"
# *before* it evaluates the key, while the native path accepts both. Rather
# than encode a belief about which is right this month, negotiate it.

WRONG_TRANSPORT_400 = httpx.Response(
    400, json={"error": {"code": 400, "message": "Missing or invalid Authorization header", "status": "INVALID_ARGUMENT"}}
)
WRONG_CREDENTIAL_TYPE_401 = httpx.Response(
    401, json={"error": {"code": 401, "message": "Expected OAuth 2 access token", "status": "UNAUTHENTICATED"}}
)
BAD_KEY_401 = httpx.Response(
    401, json={"error": {"code": 401, "message": "API key not valid. Please pass a valid API key.", "status": "UNAUTHENTICATED"}}
)

SECRET = "AQ.EXAMPLE-fake-key-for-tests"


def _keyed_google_client(key: str = SECRET) -> OpenAICompatClient:
    return OpenAICompatClient(
        Settings(llm_provider="google", llm_api_key=key, model="gemini-3.5-flash"), provider="google"
    )


def _auth_of(request) -> tuple[str, str, str]:
    """(bearer, x-goog-api-key, ?key=) as actually seen on the wire."""
    return (
        request.headers.get("authorization", ""),
        request.headers.get("x-goog-api-key", ""),
        httpx.URL(str(request.url)).params.get("key", ""),
    )


@respx.mock
def test_bearer_is_tried_first_and_kept_when_accepted():
    route = respx.post(GOOGLE_URL).mock(return_value=httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)))
    client = _keyed_google_client()
    for _ in range(3):
        assert client.decide("sys", "TARGET: Acme", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH

    assert route.call_count == 3, "a working transport must never be re-probed"
    for call in route.calls:
        assert _auth_of(call.request) == (f"Bearer {SECRET}", "", "")
    assert client.auth_style == "bearer"


@respx.mock
def test_bearer_rejected_by_transport_demotes_to_the_google_header():
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[WRONG_TRANSPORT_400, httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS))]
    )
    client = _keyed_google_client()
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert route.call_count == 2
    assert _auth_of(route.calls[0].request) == (f"Bearer {SECRET}", "", "")
    assert _auth_of(route.calls[1].request) == ("", SECRET, "")
    assert client.auth_style == "header"


@respx.mock
def test_both_header_transports_rejected_falls_through_to_the_query_param():
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            WRONG_TRANSPORT_400,
            WRONG_CREDENTIAL_TYPE_401,
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    client = _keyed_google_client()
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert route.call_count == 3
    assert [_auth_of(c.request) for c in route.calls] == [
        (f"Bearer {SECRET}", "", ""),
        ("", SECRET, ""),
        ("", "", SECRET),
    ]
    assert client.auth_style == "query"


@respx.mock
def test_the_key_is_never_sent_in_two_places_at_once():
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            WRONG_TRANSPORT_400,
            WRONG_CREDENTIAL_TYPE_401,
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    _keyed_google_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    for call in route.calls:
        carriers = [v for v in _auth_of(call.request) if v]
        assert len(carriers) == 1, f"the credential must travel exactly once, saw {len(carriers)}"


@respx.mock
def test_an_invalid_key_does_not_cycle_transports():
    """Three transports against a revoked key just triples an inevitable failure."""
    route = respx.post(GOOGLE_URL).mock(return_value=BAD_KEY_401)
    client = _keyed_google_client()
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert route.call_count == 1, "fail fast on a bad key, do not negotiate"
    assert "API key" in action.thought
    assert client.auth_style == "bearer", "a bad key must not poison the transport cache"


@respx.mock
def test_a_bad_key_that_also_names_a_transport_does_not_cycle():
    """The real live 401: 'Request had invalid authentication credentials.
    Expected OAuth 2 access token.' names a credential TYPE and also says the
    credential is invalid. The bad-key reading must win over negotiation."""
    route = respx.post(GOOGLE_URL).mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "code": 401,
                    "message": (
                        "Request had invalid authentication credentials. Expected OAuth 2 "
                        "access token, login cookie or other valid authentication credential."
                    ),
                    "status": "UNAUTHENTICATED",
                }
            },
        )
    )
    client = _keyed_google_client()
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert route.call_count == 1, "an invalid credential must not be re-presented three ways"
    assert client.auth_style == "bearer"


@respx.mock
def test_the_negotiated_transport_is_cached_across_a_whole_run():
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[WRONG_TRANSPORT_400] + [httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS))] * 10
    )
    client = _keyed_google_client()
    for _ in range(10):
        assert client.decide("sys", "TARGET: Acme", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH

    assert route.call_count == 11, "one negotiation, then ten clean calls"
    for call in route.calls[1:]:
        assert _auth_of(call.request) == ("", SECRET, "")


@respx.mock
def test_a_fresh_client_reuses_the_negotiated_transport():
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[WRONG_TRANSPORT_400] + [httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS))] * 3
    )
    _keyed_google_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    fresh = _keyed_google_client()
    assert fresh.auth_style == "header"
    assert fresh.decide("sys", "TARGET: Acme", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH
    assert route.call_count == 3  # 2 for the first client, 1 for the second


@respx.mock
@pytest.mark.parametrize(
    "pin,expected",
    [("query", ("", "", SECRET)), ("header", ("", SECRET, "")), ("bearer", (f"Bearer {SECRET}", "", ""))],
)
def test_auth_style_can_be_pinned_and_skips_negotiation(monkeypatch, pin, expected):
    monkeypatch.setenv("LEA_LLM_AUTH_STYLE", pin)
    route = respx.post(GOOGLE_URL).mock(return_value=httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)))
    client = _keyed_google_client()

    assert client.auth_style == pin
    assert client.decide("sys", "TARGET: Acme", TOOL_SPECS)[0].tool is ToolName.WEB_SEARCH
    assert route.call_count == 1
    assert _auth_of(route.calls[0].request) == expected


@respx.mock
def test_a_pinned_style_is_never_demoted(monkeypatch):
    """If the operator pinned it, respect it rather than second-guessing."""
    monkeypatch.setenv("LEA_LLM_AUTH_STYLE", "bearer")
    route = respx.post(GOOGLE_URL).mock(return_value=WRONG_TRANSPORT_400)
    client = _keyed_google_client()
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert client.auth_style == "bearer"
    assert route.call_count == 1, "a pinned transport must not be probed around"
    assert _auth_of(route.calls[0].request) == (f"Bearer {SECRET}", "", "")


def test_an_unknown_pin_falls_back_to_negotiation(monkeypatch, caplog):
    from lea.llm.openai_compat import pinned_auth_style

    monkeypatch.setenv("LEA_LLM_AUTH_STYLE", "carrier-pigeon")
    with caplog.at_level("WARNING", logger="lea.llm.openai_compat"):
        assert pinned_auth_style() is None
    assert any("LEA_LLM_AUTH_STYLE" in r.getMessage() for r in caplog.records)


@respx.mock
def test_the_key_never_appears_in_logs_or_the_trace(caplog):
    """Providers sometimes echo the credential back inside the error body."""
    respx.post(GOOGLE_URL).mock(
        return_value=httpx.Response(401, json={"error": {"code": 401, "message": f"API key not valid: {SECRET}"}})
    )
    with caplog.at_level("DEBUG", logger="lea.llm.openai_compat"):
        action, _ = _keyed_google_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert SECRET not in action.thought
    assert "<redacted>" in action.thought
    for record in caplog.records:
        assert SECRET not in record.getMessage()


@respx.mock
def test_a_query_param_key_is_redacted_from_error_text(caplog):
    respx.post(GOOGLE_URL).mock(side_effect=httpx.ConnectError(f"failed connecting to {GOOGLE_URL}?key={SECRET}"))
    with caplog.at_level("DEBUG", logger="lea.llm.openai_compat"):
        action, _ = _keyed_google_client().decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert SECRET not in action.thought
    for record in caplog.records:
        assert SECRET not in record.getMessage()


@respx.mock
def test_a_keyless_endpoint_sends_no_credential_in_any_transport():
    url = "http://localhost:11434/v1/chat/completions"
    route = respx.post(url).mock(return_value=httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)))
    OpenAICompatClient(Settings(llm_provider="ollama"), provider="ollama").decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert _auth_of(route.calls[0].request) == ("", "", "")


@respx.mock
def test_transport_negotiation_does_not_consume_the_retry_budget():
    """Probing is negotiation, not a retry: a 429 after it must still get retries."""
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            WRONG_TRANSPORT_400,
            *[httpx.Response(429, headers={"Retry-After": "1"}, json={"error": {"message": "slow down"}})] * 3,
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    action, _ = _keyed_google_client().decide("sys", "TARGET: Acme", TOOL_SPECS)
    # One probe + a full MAX_ATTEMPTS budget of real retries. If the probe were
    # counted as a retry, the run would die on the last 429 instead.
    assert action.tool is ToolName.WEB_SEARCH
    assert route.call_count == 1 + MAX_ATTEMPTS


@respx.mock
def test_model_discovery_uses_the_negotiated_transport():
    """`list_models` must not re-fail on an endpoint the chat path already solved."""
    respx.post(GOOGLE_URL).mock(
        side_effect=[WRONG_TRANSPORT_400, httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS))]
    )
    models = respx.get(GOOGLE_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "models/gemini-3.5-flash"}]})
    )
    settings = Settings(llm_provider="google", llm_api_key=SECRET, model="gemini-3.5-flash")
    OpenAICompatClient(settings, provider="google").decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert list_models(settings, provider="google") == ["gemini-3.5-flash"]
    assert _auth_of(models.calls[0].request) == ("", SECRET, ""), "discovery must reuse the negotiated transport"


@respx.mock
def test_model_discovery_defaults_to_bearer_before_any_negotiation():
    models = respx.get(GOOGLE_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "models/gemini-3.5-flash"}]})
    )
    settings = Settings(llm_provider="google", llm_api_key=SECRET, model="gemini-3.5-flash")
    assert list_models(settings, provider="google") == ["gemini-3.5-flash"]
    assert _auth_of(models.calls[0].request) == (f"Bearer {SECRET}", "", "")


def test_reset_provider_state_clears_the_transport_cache():
    from lea.llm.openai_compat import _AUTH_STYLE_CACHE

    _AUTH_STYLE_CACHE["https://x/v1"] = "query"
    reset_provider_state()
    assert _AUTH_STYLE_CACHE == {}


def test_no_key_format_validation_exists_anywhere_in_scope():
    """Key formats are the provider's business. Google shipped an `AQ.` format
    that a naive `startswith("AIza")` check would have rejected outright."""
    import pathlib
    import re as _re

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "lea"
    owned = [
        root / "llm" / "openai_compat.py",
        root / "llm" / "factory.py",
        root / "llm" / "anthropic_client.py",
        root / "llm" / "stub.py",
        root / "llm" / "json_utils.py",
        root / "prompts.py",
    ]
    banned = _re.compile(r"AIza|startswith\s*\(\s*['\"](?:AIza|sk-|AQ)", _re.IGNORECASE)
    for path in owned:
        source = path.read_text()
        assert not banned.search(source), f"{path.name} must not validate key formats"
        # A key's LENGTH must never gate a request either. The one legitimate
        # use is the redaction floor, which rejects nothing — it only stops a
        # 1-char test stub from substring-shredding log text.
        for lineno, line in enumerate(source.splitlines(), 1):
            if _re.search(r"len\(\s*(?:api_)?key\s*\)", line):
                assert "MIN_REDACTABLE_KEY_LEN" in line, (
                    f"{path.name}:{lineno} checks key length outside redaction: {line.strip()}"
                )


# --- HTTP 200 that carries nothing -------------------------------------------
#
# `gemini-3.6-flash` returns 200 on every call and yields nothing: a whole run
# reports `4 steps · 0 searches · 0 pages · 0+0 tokens`. It is the preset
# default AND what Google's 404 names as the successor for retired ids, so the
# resolver can substitute *into* it. Treating 200-with-nothing as success is
# what makes that failure silent.


def _empty_response(finish_reason: str = "length", **message_extra) -> dict:
    message = {"role": "assistant", "content": None}
    message.update(message_extra)
    return {
        "model": "gemini-3.6-flash",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 800, "completion_tokens": 0},
    }


@pytest.mark.parametrize(
    "data,expected",
    [
        ({}, True),
        ({"choices": []}, True),
        ({"choices": [{"message": {"content": None}, "finish_reason": "length"}]}, True),
        ({"choices": [{"message": {"content": "   "}}]}, True),
        ({"choices": [{"message": {"content": "hello"}}]}, False),
        ({"choices": [{"message": {"content": None, "reasoning": "thinking out loud"}}]}, False),
        ({"choices": [{"message": {"content": "", "reasoning_content": "ok"}}]}, False),
        (
            {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]}}]},
            False,
        ),
    ],
)
def test_is_empty_completion_matrix(data, expected):
    from lea.llm.openai_compat import is_empty_completion

    assert is_empty_completion(data) is expected


@respx.mock
def test_a_normal_response_never_triggers_the_empty_path():
    route = respx.post(GOOGLE_URL).mock(return_value=httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)))
    action, _ = _google_client("gemini-3.5-flash").decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH
    assert route.call_count == 1, "a working model must not pay for this defence"


@respx.mock
def test_empty_content_with_finish_reason_length_retries_with_a_bigger_budget():
    """A thinking model can spend the whole budget before writing an answer."""
    from lea.llm.openai_compat import EMPTY_RETRY_MAX_TOKENS

    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            httpx.Response(200, json=_empty_response("length")),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    action, _ = _google_client("gemini-3.6-flash").decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH, "200-with-nothing must not become a lost step"
    assert route.call_count == 2
    first, second = (json.loads(c.request.content) for c in route.calls)
    assert second["max_tokens"] == EMPTY_RETRY_MAX_TOKENS
    assert second["max_tokens"] > first["max_tokens"]


@respx.mock
def test_a_persistently_empty_model_is_swapped_out():
    models = _models_route()
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            httpx.Response(200, json=_empty_response("length")),   # first budget
            httpx.Response(200, json=_empty_response("length")),   # escalated budget
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),  # new model
        ]
    )
    client = _google_client("gemini-3.6-flash")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert client.model == "gemini-3.5-flash", "must route off the model that returns nothing"
    assert json.loads(route.calls[2].request.content)["model"] == "gemini-3.5-flash"
    assert models.call_count == 1

    # The broken model must be remembered, or the resolver walks back into it.
    from lea.llm.openai_compat import is_model_usable

    base = "https://generativelanguage.googleapis.com/v1beta/openai"
    assert not is_model_usable(base, "gemini-3.6-flash")
    assert is_model_usable(base, "gemini-3.5-flash")


@respx.mock
def test_an_empty_model_with_no_alternative_degrades_with_the_doctor_hint():
    _models_route(["gemini-3.6-flash"])  # the broken model is the only one offered
    respx.post(GOOGLE_URL).mock(return_value=httpx.Response(200, json=_empty_response("length")))
    action, _ = _google_client("gemini-3.6-flash").decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.FINISH
    assert "lea doctor --models" in action.thought
    assert "LEA_MODEL" in action.thought
    assert "no content" in action.thought


@respx.mock
def test_the_resolver_never_substitutes_into_a_model_proven_empty():
    """Google's 404 names gemini-3.6-flash as the successor — but if 3.6 has
    already proven unusable, the resolver must pick something else."""
    from lea.llm.openai_compat import mark_model_unusable

    base = "https://generativelanguage.googleapis.com/v1beta/openai"
    mark_model_unusable(base, "gemini-3.6-flash")
    _models_route()
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404(
                "models/gemini-2.0-flash is not found for API version v1beta, "
                "please update to models/gemini-3.6-flash"
            ),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    client = _google_client("gemini-2.0-flash")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert client.model == "gemini-3.5-flash", "the named successor is known-broken"
    assert json.loads(route.calls[1].request.content)["model"] == "gemini-3.5-flash"


@respx.mock
def test_an_answer_hiding_in_a_reasoning_field_is_read():
    """Provider-dependent: reasoning models surface text under varying names."""
    payload = json.dumps(DECIDE_ARGS)
    respx.post(GOOGLE_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "tool_choice is not supported"}}),
            httpx.Response(200, json=_empty_response("stop", reasoning_content=payload)),
        ]
    )
    action, _ = _google_client("gemini-3.6-flash").decide("sys", "TARGET: Acme", TOOL_SPECS)
    assert action.tool is ToolName.WEB_SEARCH
    assert action.args == {"query": "Acme Robotics"}


@respx.mock
def test_extract_also_survives_an_empty_model():
    models = _models_route()
    payload = {"input": "Vanta", "company_name": "Vanta", "industry": "Security & Compliance"}
    respx.post(GOOGLE_URL).mock(
        side_effect=[
            httpx.Response(200, json=_empty_response("length")),
            httpx.Response(200, json=_empty_response("length")),
            httpx.Response(200, json=_tool_response("record_companyrecord", payload)),
        ]
    )
    record, _ = _google_client("gemini-3.6-flash").extract("sys", "ctx", CompanyRecord)
    assert record.company_name == "Vanta"
    assert record.industry == "Security & Compliance"
    assert models.call_count == 1


@respx.mock
def test_the_ranked_fallback_also_skips_unusable_models():
    """No named successor this time — the ranking itself must skip a known-bad id."""
    from lea.llm.openai_compat import mark_model_unusable

    base = "https://generativelanguage.googleapis.com/v1beta/openai"
    mark_model_unusable(base, "gemini-3.5-flash")  # the model ranking would pick
    _models_route()
    route = respx.post(GOOGLE_URL).mock(
        side_effect=[
            _dead_model_404("models/gemini-2.0-flash is not found for API version v1beta"),
            httpx.Response(200, json=_tool_response("next_action", DECIDE_ARGS)),
        ]
    )
    client = _google_client("gemini-2.0-flash")
    action, _ = client.decide("sys", "TARGET: Acme", TOOL_SPECS)

    assert action.tool is ToolName.WEB_SEARCH
    assert client.model == "gemini-2.5-flash", "skip the unusable top pick, take the next"
    assert json.loads(route.calls[1].request.content)["model"] == "gemini-2.5-flash"


def test_redaction_does_not_shred_text_for_a_short_stub_key():
    """A 1-char test key must not substring-match its way through the message."""
    from lea.llm.openai_compat import OpenAICompatClient

    client = OpenAICompatClient(Settings(llm_provider="google", llm_api_key="k"), provider="google")
    assert client._redact("max_tokens=16384 — pick another") == "max_tokens=16384 — pick another"


def test_redaction_still_removes_a_realistic_key():
    from lea.llm.openai_compat import OpenAICompatClient

    client = OpenAICompatClient(Settings(llm_provider="google", llm_api_key=SECRET), provider="google")
    out = client._redact(f"API key not valid: {SECRET}")
    assert SECRET not in out and "<redacted>" in out


def test_reset_provider_state_clears_the_unusable_set():
    from lea.llm.openai_compat import _UNUSABLE_MODELS, is_model_usable, mark_model_unusable

    mark_model_unusable("https://x/v1", "broken")
    assert not is_model_usable("https://x/v1", "broken")
    reset_provider_state()
    assert _UNUSABLE_MODELS == set()
    assert is_model_usable("https://x/v1", "broken")


# --- model discovery --------------------------------------------------------


@respx.mock
def test_list_models_returns_sorted_ids():
    respx.get(GROQ_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "gemma2-9b-it"}]})
    )
    models = list_models(Settings(llm_provider="groq", llm_api_key="k"), provider="groq")
    assert models == ["gemma2-9b-it", "llama-3.3-70b-versatile"]


@respx.mock
def test_list_models_strips_the_google_models_prefix():
    url = "https://generativelanguage.googleapis.com/v1beta/openai/models"
    respx.get(url).mock(return_value=httpx.Response(200, json={"data": [{"id": "models/gemini-2.0-flash"}]}))
    assert list_models(Settings(llm_provider="google", llm_api_key="k"), provider="google") == ["gemini-2.0-flash"]


@respx.mock
def test_list_models_returns_empty_on_any_failure():
    respx.get(GROQ_MODELS_URL).mock(return_value=httpx.Response(401, text="bad key"))
    assert list_models(Settings(llm_provider="groq", llm_api_key="k"), provider="groq") == []


@respx.mock
def test_list_models_returns_empty_on_connection_error():
    respx.get(GROQ_MODELS_URL).mock(side_effect=httpx.ConnectError("down"))
    assert list_models(Settings(llm_provider="groq", llm_api_key="k"), provider="groq") == []


def test_factory_list_models_is_empty_for_non_compat_providers():
    from lea.llm.factory import list_models as factory_list_models

    assert factory_list_models(Settings(llm_provider="stub")) == []
    assert factory_list_models(Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-x")) == []


# --- factory routing --------------------------------------------------------


@pytest.mark.parametrize("provider", ["google", "groq", "openrouter", "cerebras"])
def test_factory_routes_keyed_openai_compat_providers(provider, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = get_llm(Settings(llm_provider=provider, llm_api_key="test-key"))
    assert client.name == provider
    assert isinstance(client, OpenAICompatClient)
    assert client.base_url.startswith("http")
    assert client.model


def test_factory_routes_ollama_without_any_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = get_llm(Settings(llm_provider="ollama"))
    assert isinstance(client, OpenAICompatClient)
    assert client.name == "ollama"
    assert client.api_key is None


@pytest.mark.parametrize("provider", ["google", "groq", "openrouter", "cerebras"])
def test_factory_falls_back_to_stub_without_a_key(provider, monkeypatch):
    for preset_env in ("GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "LEA_LLM_API_KEY"):
        monkeypatch.delenv(preset_env, raising=False)
    client = get_llm(Settings(llm_provider=provider))
    assert client.name == "stub"


def test_factory_falls_back_to_stub_on_a_bad_base_url(monkeypatch, caplog):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with caplog.at_level("WARNING"):
        client = get_llm(Settings(llm_provider="groq", llm_api_key="k", llm_base_url="not-a-url"))
    assert client.name == "stub"
    assert any("base URL" in r.message for r in caplog.records)


def test_factory_honours_a_custom_endpoint_via_base_url(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(llm_provider="my-gateway", llm_base_url="https://llm.internal/v1", llm_api_key="k")
    client = get_llm(settings)
    assert isinstance(client, OpenAICompatClient)
    assert client.base_url == "https://llm.internal/v1"


def test_openai_compat_clients_satisfy_the_llm_protocol():
    from lea.llm.base import LLMClient

    client = get_llm(Settings(llm_provider="groq", llm_api_key="k"))
    assert isinstance(client, LLMClient)
    assert callable(client.decide) and callable(client.extract)


def test_auto_resolution_prefers_a_free_provider_over_the_stub(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    settings = Settings.from_env()
    assert settings.resolved_llm() == "groq"
    assert get_llm(settings).name == "groq"


def test_a_keyless_clone_still_gets_the_stub(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY",
        "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "LEA_LLM_API_KEY",
        "LEA_LLM_BASE_URL", "LEA_LLM_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    client = get_llm(Settings.from_env())
    assert client.name == "stub"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
