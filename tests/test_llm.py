"""LLM layer tests. No network, no API key, no SDK required."""

from __future__ import annotations

from datetime import date
from typing import Optional

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
