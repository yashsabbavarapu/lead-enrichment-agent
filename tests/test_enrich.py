"""Synthesis-layer tests. No network, no real LLM, no clock dependence.

The FakeLLM returns canned instances so every assertion below is about OUR
post-processing — normalization, dedupe, the score arithmetic, the hook filter —
rather than about model behavior.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from lea.config import Settings
from lea.enrich.extract import (
    EMPLOYEE_BUCKETS,
    bucket_midpoint,
    coerce_founded_year,
    extract_record,
    merge_records,
    normalize_domain,
    normalize_employee_range,
)
from lea.enrich.hooks import CLICHES, MAX_HOOK_WORDS, filter_hooks, generate_hooks, is_grounded
from lea.enrich.icp import (
    DEFAULT_ICP,
    compute_fit_score,
    infer_icp,
    load_icp,
    load_or_default_icp,
    save_icp,
    score_icp,
    tier_for,
)
from lea.io_csv import CSV_COLUMNS, read_targets, write_csv
from lea.llm.base import LLMUsage
from lea.models import (
    CompanyRecord,
    CriterionScore,
    EnrichmentResult,
    ICPCriterion,
    ICPFit,
    ICPProfile,
    OutreachHook,
    RunStats,
    Signal,
    TechStack,
)


# --------------------------------------------------------------------------
# Fixtures / fakes
# --------------------------------------------------------------------------


class FakeLLM:
    """Returns a canned instance per schema model. Records what it was asked."""

    name = "fake"
    model = "fake-1"

    def __init__(self, canned: Optional[dict[type, Any]] = None, raise_on: Optional[type] = None):
        self.canned = canned or {}
        self.raise_on = raise_on
        self.calls: list[tuple[str, str, type]] = []

    def decide(self, system: str, context: str, allowed_tools: list[dict[str, Any]]):  # pragma: no cover
        raise NotImplementedError

    def extract(self, system: str, context: str, schema_model: type, hint: Optional[str] = None):
        self.calls.append((system, context, schema_model))
        if self.raise_on is not None and schema_model is self.raise_on:
            raise RuntimeError("simulated model failure")
        if schema_model in self.canned:
            return self.canned[schema_model], LLMUsage(tokens_in=10, tokens_out=10, model="fake-1")
        # Unknown schema: best-effort empty instance, like a real stub provider.
        try:
            return schema_model(), LLMUsage()
        except Exception:
            return schema_model.model_construct(), LLMUsage()


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="stub", search_provider="fixture", offline=True)


def _messy_record() -> CompanyRecord:
    """What a sloppy-but-plausible model return looks like."""
    return CompanyRecord(
        input="Ramp",
        company_name="  Ramp  ",
        domain="HTTPS://WWW.Ramp.Com/careers/?utm=x",
        description="Corporate cards and spend management.",
        employee_range="~250 employees",
        founded_year=1492,  # implausible: must be dropped
        signals=[
            Signal(kind="funding", headline="Raised $150M Series D", date="2024-01-05",
                   source_url="https://techcrunch.com/ramp"),
            Signal(kind="funding", headline="Raised $150M Series D!", date="2024-01-05"),  # dupe
            Signal(kind="hiring", headline="Hiring 12 enterprise AEs", date="2025-06-01",
                   source_url="https://ramp.com/careers"),
            Signal(kind="product", headline="Launched procurement module"),  # undated
        ],
        tech_stack=[
            TechStack(category="analytics", name="Segment", confidence=0.4, evidence="script tag"),
            TechStack(category="Analytics", name="segment", confidence=0.9, evidence="cdn header"),
            TechStack(category="crm", name="Salesforce", confidence=0.7),
        ],
    )


# The evidence must actually support the record: extract_record verifies every
# scalar and signal against this text, so a fixture with placeholder text would
# (correctly) have its fields stripped.
EVIDENCE = [
    ("https://ramp.com", "Ramp is a corporate card company with ~250 employees. "
                         "Launched procurement module. Hiring 12 enterprise AEs."),
    ("https://techcrunch.com/ramp", "Ramp raised $150M Series D."),
    ("https://ramp.com", "duplicate url block"),
]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://www.ramp.com/", "ramp.com"),
        ("HTTP://WWW.Ramp.Com/careers/?a=b#c", "ramp.com"),
        ("ramp.com", "ramp.com"),
        ("www.sub.ramp.co.uk", "sub.ramp.co.uk"),
        ("https://ramp.com:8443/x", "ramp.com"),
        ("", None),
        (None, None),
        ("not a domain", None),
        ("localhost", None),
    ],
)
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("~250 employees", "201-500"),
        ("51-200", "51-200"),
        ("51 - 200", "51-200"),
        (7, "1-10"),
        ("10,001+", "5000+"),
        ("1k-5k", "1001-5000"),
        ("about 900 people", "501-1000"),
        ("unknown", None),
        (None, None),
        ("", None),
    ],
)
def test_normalize_employee_range(raw, expected):
    got = normalize_employee_range(raw)
    assert got == expected
    if got is not None:
        assert got in EMPLOYEE_BUCKETS


def test_employee_estimate_is_bucket_midpoint():
    assert bucket_midpoint("201-500") == 350
    assert bucket_midpoint("1-10") == 5
    assert bucket_midpoint(None) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (2019, 2019),
        ("2019", 2019),
        ("founded in 1998", 1998),
        ("1998-03-01", 1998),
        (1492, None),   # before 1800
        (3030, None),   # in the future
        ("no year here", None),
        (None, None),
        (True, None),
    ],
)
def test_coerce_founded_year(raw, expected):
    assert coerce_founded_year(raw) == expected


# --------------------------------------------------------------------------
# extract_record
# --------------------------------------------------------------------------


def test_extract_record_normalizes_everything(settings):
    llm = FakeLLM({CompanyRecord: _messy_record()})
    rec = extract_record("Ramp", EVIDENCE, llm, settings)

    assert rec.company_name == "Ramp"
    assert rec.domain == "ramp.com"
    assert rec.employee_range == "201-500"
    assert rec.employee_estimate == 350
    assert rec.founded_year is None  # 1492 rejected


def test_extract_record_dedupes_and_sorts_signals(settings):
    llm = FakeLLM({CompanyRecord: _messy_record()})
    rec = extract_record("Ramp", EVIDENCE, llm, settings)

    headlines = [s.headline for s in rec.signals]
    assert len(headlines) == 3, headlines  # the "!" duplicate collapsed
    # most recent first, undated last
    assert headlines[0] == "Hiring 12 enterprise AEs"      # 2025-06-01
    assert headlines[1] == "Raised $150M Series D"          # 2024-01-05
    assert headlines[2] == "Launched procurement module"    # undated
    assert rec.signals[2].recency_note == "undated"
    # the richer of the two duplicates (the one with a source) survived
    assert rec.signals[1].source_url == "https://techcrunch.com/ramp"


def test_extract_record_dedupes_tech_keeping_highest_confidence(settings):
    llm = FakeLLM({CompanyRecord: _messy_record()})
    rec = extract_record("Ramp", EVIDENCE, llm, settings)

    keys = {(t.category.lower(), t.name.lower()) for t in rec.tech_stack}
    assert keys == {("analytics", "segment"), ("crm", "salesforce")}
    seg = next(t for t in rec.tech_stack if t.name.lower() == "segment")
    assert seg.confidence == 0.9


def test_tech_dedupe_collapses_generic_and_cross_category_duplicates(settings):
    """Both duplications seen in the real `lea enrich "Vanta" --offline` output."""
    rec = CompanyRecord(
        input="Vanta",
        company_name="Vanta",
        tech_stack=[
            # generic/specific pair: GA4 is the real detection
            TechStack(category="analytics", name="Google Analytics",
                      evidence="gtag.js present", confidence=0.6),
            TechStack(category="analytics", name="Google Analytics 4",
                      evidence="G- measurement id", confidence=0.9),
            TechStack(category="analytics", name="Google Tag Manager", confidence=0.9),
            # same name, two categories: there is only one Cloudflare
            TechStack(category="hosting", name="Cloudflare",
                      evidence="server header", confidence=0.7),
            TechStack(category="infra", name="Cloudflare",
                      evidence="cf-ray header", confidence=0.95),
            # multi-category vendor: collapsed by the same rule, deliberately
            TechStack(category="crm", name="HubSpot", evidence="hs-script", confidence=0.9),
            TechStack(category="marketing", name="HubSpot",
                      evidence="hubspot forms", confidence=0.8),
        ],
    )
    llm = FakeLLM({CompanyRecord: rec})
    out = extract_record("Vanta", [("https://vanta.com", "t")], llm, settings)

    names = [t.name for t in out.tech_stack]
    assert names == ["Google Analytics 4", "Google Tag Manager", "Cloudflare", "HubSpot"]
    assert "Google Analytics" not in names       # generic dropped for the specific
    assert len(names) == len(set(names))         # every product listed exactly once

    by_name = {t.name: t for t in out.tech_stack}
    # the higher-confidence detection wins and names the surviving category
    assert by_name["Cloudflare"].category == "infra"
    assert by_name["Cloudflare"].confidence == 0.95
    assert by_name["HubSpot"].category == "crm"
    # evidence from a collapsed row is preserved rather than thrown away
    assert "cf-ray header" in by_name["Cloudflare"].evidence
    assert "server header" in by_name["Cloudflare"].evidence
    assert "hubspot forms" in by_name["HubSpot"].evidence
    assert "gtag.js present" in by_name["Google Analytics 4"].evidence


def test_tech_dedupe_keeps_distinct_products_that_merely_share_a_word():
    """'Google Analytics 4' must not swallow 'Google Tag Manager'."""
    from lea.enrich.extract import _dedupe_tech

    out = _dedupe_tech([
        TechStack(category="analytics", name="Google Analytics 4", confidence=0.9),
        TechStack(category="tag", name="Google Tag Manager", confidence=0.9),
        TechStack(category="hosting", name="Google Cloud", confidence=0.8),
    ])
    assert [t.name for t in out] == ["Google Analytics 4", "Google Tag Manager", "Google Cloud"]


def test_extract_record_populates_unknowns_and_sources(settings):
    llm = FakeLLM({CompanyRecord: _messy_record()})
    rec = extract_record("Ramp", EVIDENCE, llm, settings)

    # industry / hq / founded_year were never established
    assert "industry" in rec.unknowns
    assert "hq_city" in rec.unknowns
    assert "hq_country" in rec.unknowns
    assert "founded_year" in rec.unknowns
    # things we DO know must not be listed as gaps
    assert "domain" not in rec.unknowns
    assert "employee_range" not in rec.unknowns   # "~250 employees" is in the evidence
    # sources are the unique evidence URLs actually present
    assert rec.sources == ["https://ramp.com", "https://techcrunch.com/ramp"]


def test_extract_record_survives_llm_failure(settings):
    llm = FakeLLM(raise_on=CompanyRecord)
    rec = extract_record("Ghost Co", EVIDENCE, llm, settings)
    assert rec.input == "Ghost Co"
    assert rec.company_name == "Ghost Co"   # derived from the target, never invented
    # EVIDENCE is about Ramp, not Ghost Co — no host resembles the target, so the
    # domain stays unknown rather than being misattributed.
    assert rec.domain is None
    assert "domain" in rec.unknowns
    # ...and for the same reason none of it may be cited as a source about Ghost Co.
    assert rec.sources == []


def test_extract_record_repairs_a_partially_constructed_model(settings):
    """A stub provider that falls back to `model_construct()` returns an instance
    whose required fields are simply absent. That must not blow up the run."""
    partial = CompanyRecord.model_construct()
    llm = FakeLLM({CompanyRecord: partial})
    rec = extract_record("Ghost Co", EVIDENCE, llm, settings)

    assert rec.input == "Ghost Co"
    assert rec.company_name == "Ghost Co"   # derived, not left None
    assert rec.domain is None
    assert "domain" in rec.unknowns
    assert rec.sources == []                # Ramp's pages say nothing about Ghost Co


# --------------------------------------------------------------------------
# Identity sanitation — regressions found on live keyless runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # page titles are rejected outright, not cleaned up
        ("About Us - Different By Design", None),
        ("Home | Acme", None),
        ("Careers at Ramp", None),
        ("Pricing - Ramp", None),
        ("Contact Us", None),
        ("Privacy Policy | Ramp", None),
        ("Terms of Service", None),
        ("Blog - Ramp", None),
        ("Page Not Found", None),
        ("About", None),
        ("x" * 61, None),  # too long to be a name
        # real names, with brand furniture stripped
        ("Ramp | Corporate Cards & Spend Management", "Ramp"),
        ("Vanta — Automated Compliance", "Vanta"),
        ("Retool - Build internal tools remarkably fast", "Retool"),
        ("  Ramp  ", "Ramp"),
        ('"Ramp"', "Ramp"),
        ("Ramp", "Ramp"),
        ("Acme Corporation", "Acme Corporation"),
    ],
)
def test_sanitize_company_name(raw, expected):
    from lea.enrich.extract import sanitize_company_name

    assert sanitize_company_name(raw, target="Ramp") == expected


def test_company_name_matching_its_domain_survives_a_page_word():
    """'Home Depot' contains 'Home' but is the real name — the domain proves it."""
    from lea.enrich.extract import sanitize_company_name

    assert sanitize_company_name("Home Depot", "Home Depot", "homedepot.com") == "Home Depot"
    assert sanitize_company_name("Home Depot", "Home Depot", "ramp.com") is None


def test_page_title_is_replaced_by_a_derived_name(settings):
    """The exact live failure: a page <title> rendered as the company name."""
    bad = CompanyRecord(
        input="Ramp", company_name="About Us - Different By Design", domain="ramp.com"
    )
    llm = FakeLLM({CompanyRecord: bad})
    rec = extract_record("Ramp", [("https://ramp.com/about-us/", "text")], llm, settings)

    assert rec.company_name == "Ramp"
    assert "Different By Design" not in (rec.company_name or "")


def test_derived_name_falls_back_to_the_domain_label(settings):
    bad = CompanyRecord(input="https://ramp.com", company_name="Home | Acme", domain="ramp.com")
    llm = FakeLLM({CompanyRecord: bad})
    rec = extract_record("https://ramp.com", [("https://ramp.com", "t")], llm, settings)
    assert rec.company_name == "Ramp"


@pytest.mark.parametrize(
    "host",
    ["tracxn.com", "crunchbase.com", "linkedin.com", "wikipedia.org", "en.wikipedia.org",
     "pitchbook.com", "glassdoor.com", "bloomberg.com", "zoominfo.com", "g2.com",
     "facebook.com", "x.com", "medium.com", "owler.com", "apollo.io"],
)
def test_aggregator_hosts_are_rejected_as_domains(host):
    from lea.enrich.extract import choose_domain, is_aggregator

    assert is_aggregator(host, "Ramp")
    assert choose_domain(host, "Ramp", "Ramp", []) is None


def test_aggregator_denylist_is_overridden_when_it_is_the_company():
    from lea.enrich.extract import choose_domain, is_aggregator

    assert not is_aggregator("github.com", "GitHub")
    assert choose_domain("github.com", "GitHub", "GitHub", []) == "github.com"
    assert not is_aggregator("linkedin.com", "LinkedIn")


def test_domain_prefers_the_host_resembling_the_target():
    from lea.enrich.extract import choose_domain

    urls = ["https://tracxn.com/d/companies/ramp", "https://ramp.com/about", "https://vc.example/x"]
    assert choose_domain("tracxn.com", "Ramp", "Ramp", urls) == "ramp.com"
    # nothing clean available -> None, which lands in unknowns
    assert choose_domain("tracxn.com", "Ramp", "Ramp", ["https://crunchbase.com/ramp"]) is None


def test_aggregator_domain_never_reaches_the_record(settings):
    """The exact live failure: a robots-blocked aggregator cited as the domain."""
    bad = CompanyRecord(input="Ramp", company_name="Ramp", domain="tracxn.com")
    llm = FakeLLM({CompanyRecord: bad})
    evidence = [
        ("https://tracxn.com/d/companies/ramp", ""),  # the FAILED fetch
        ("https://ramp.com/about-us/", "Ramp is a spend management company."),
        ("https://ramp.com/blog/series-d", "Ramp raised $150M."),
    ]
    rec = extract_record("Ramp", evidence, llm, settings)

    assert rec.domain == "ramp.com"
    assert not any("tracxn" in s for s in rec.sources)


def test_failed_fetch_urls_are_not_cited_as_sources():
    from lea.enrich.extract import citable_urls

    ev = [
        ("https://blocked.example", ""),                   # empty body = failed fetch
        ("https://err.example", "ERROR: 403 robots-blocked"),
        ("https://good.example", "real page text"),
        ("https://good.example", "duplicate url"),
        ("", "orphan text with no url"),
    ]
    assert citable_urls(ev) == ["https://good.example"]


def test_sources_are_populated_from_evidence_actually_read(settings):
    """Regression: sources came back empty despite four pages being fetched."""
    llm = FakeLLM({CompanyRecord: CompanyRecord(input="Ramp", company_name="Ramp")})
    evidence = [
        ("https://ramp.com", "page one"),
        ("https://ramp.com/about-us/", "page two"),
        ("https://ramp.com/blog/series-d", "page three"),
        ("https://ramp.com/careers", "page four"),
    ]
    rec = extract_record("Ramp", evidence, llm, settings)

    assert rec.sources == [u for u, _ in evidence]
    assert rec.sources, "non-empty evidence must never yield empty sources"


def test_merge_records_does_not_reintroduce_an_aggregator_domain():
    base = CompanyRecord(input="Ramp", company_name="Ramp", domain="crunchbase.com")
    fresh = CompanyRecord(input="Ramp", domain="ramp.com", sources=["https://ramp.com"])
    merged = merge_records(base, fresh)
    assert merged.domain == "ramp.com"


def test_merge_records_base_wins_for_set_fields():
    base = CompanyRecord(input="Ramp", company_name="Ramp Inc", industry="Fintech",
                         sources=["https://base.example"])
    fresh = CompanyRecord(
        input="Ramp",
        company_name="Ramp (from web)",
        industry="Payments",
        hq_city="New York",
        signals=[Signal(kind="news", headline="New office")],
        sources=["https://fresh.example"],
    )
    merged = merge_records(base, fresh)

    assert merged.company_name == "Ramp Inc"      # base wins
    assert merged.industry == "Fintech"           # base wins
    assert merged.hq_city == "New York"           # gap filled from fresh
    assert [s.headline for s in merged.signals] == ["New office"]
    assert merged.sources == ["https://base.example", "https://fresh.example"]
    assert "hq_city" not in merged.unknowns       # unknowns recomputed, not stale


def test_extract_record_merges_with_base(settings):
    base = CompanyRecord(input="Ramp", company_name="Ramp Inc", industry="Fintech")
    llm = FakeLLM({CompanyRecord: _messy_record()})
    rec = extract_record("Ramp", EVIDENCE, llm, settings, base=base)

    assert rec.company_name == "Ramp Inc"
    assert rec.industry == "Fintech"
    assert rec.domain == "ramp.com"               # only the fresh pass had this
    assert "industry" not in rec.unknowns


# --------------------------------------------------------------------------
# ICP: arithmetic
# --------------------------------------------------------------------------


def _icp_4() -> ICPProfile:
    return ICPProfile(
        name="test-icp",
        summary="test",
        criteria=[
            ICPCriterion(key="company_type", description="B2B", weight=2.0),
            ICPCriterion(key="company_size", description="size", weight=1.0),
            ICPCriterion(key="tech_maturity", description="tech", weight=1.0),
            ICPCriterion(key="growth_signals", description="growth", weight=1.0),
        ],
        disqualifiers=["Consumer-only (B2C) company with no business-facing product."],
    )


def _fit_from(scores: dict[str, float], urls: Optional[list[str]] = None,
              dq: Optional[list[str]] = None) -> ICPFit:
    icp = _icp_4()
    weights = {c.key: c.weight for c in icp.criteria}
    return ICPFit(
        score=99.0,  # deliberately wrong: we must overwrite it
        tier="A",
        summary="",
        criteria=[
            # Distinct reasoning per criterion, as a real model returns. Identical
            # scores *and* identical reasoning is what marks a response degenerate.
            CriterionScore(key=k, score=v, weight=weights[k],
                           reasoning=f"Judged {k} against the evidence, landing on {v}/10.",
                           evidence_urls=list(urls) if urls else [])
            for k, v in scores.items()
        ],
        disqualifiers_hit=list(dq or []),
    )


def _known_record() -> CompanyRecord:
    return CompanyRecord(
        input="Ramp", company_name="Ramp", domain="ramp.com", industry="Fintech",
        employee_range="201-500", hq_city="New York", hq_country="USA", founded_year=2019,
        signals=[Signal(kind="funding", headline="Series D", source_url="https://a.example")],
        tech_stack=[TechStack(category="crm", name="Salesforce")],
        sources=["https://a.example"],
    )


def test_compute_fit_score_is_a_weighted_mean_rescaled():
    # weights 2,1,1,1 (total 5); scores 10,8,6,4
    # weighted = 20 + 8 + 6 + 4 = 38 ; 38/5 = 7.6 ; *10 = 76.0
    cs = [
        CriterionScore(key="a", score=10, weight=2.0),
        CriterionScore(key="b", score=8, weight=1.0),
        CriterionScore(key="c", score=6, weight=1.0),
        CriterionScore(key="d", score=4, weight=1.0),
    ]
    assert compute_fit_score(cs) == 76.0
    assert compute_fit_score([]) == 0.0


def test_score_icp_recomputes_the_total_and_ignores_the_models_number(settings):
    icp = _icp_4()
    fit_in = _fit_from(
        {"company_type": 10, "company_size": 8, "tech_maturity": 6, "growth_signals": 4},
        urls=["https://a.example"],
    )
    llm = FakeLLM({ICPFit: fit_in})
    fit = score_icp(_known_record(), icp, [("https://a.example", "text")], llm, settings)

    assert fit.score == 76.0     # NOT the 99.0 the model claimed
    assert fit.tier == "B"
    assert [c.key for c in fit.criteria] == [c.key for c in icp.criteria]
    assert [c.weight for c in fit.criteria] == [2.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "scores,expected_score,expected_tier",
    [
        ({"company_type": 10, "company_size": 10, "tech_maturity": 8, "growth_signals": 8}, 92.0, "A"),
        ({"company_type": 10, "company_size": 8, "tech_maturity": 6, "growth_signals": 4}, 76.0, "B"),
        ({"company_type": 5, "company_size": 5, "tech_maturity": 5, "growth_signals": 5}, 50.0, "C"),
        ({"company_type": 2, "company_size": 2, "tech_maturity": 2, "growth_signals": 2}, 20.0, "D"),
        # exact threshold boundaries
        ({"company_type": 8, "company_size": 8, "tech_maturity": 8, "growth_signals": 8}, 80.0, "A"),
        ({"company_type": 6, "company_size": 6, "tech_maturity": 6, "growth_signals": 6}, 60.0, "B"),
        ({"company_type": 4, "company_size": 4, "tech_maturity": 4, "growth_signals": 4}, 40.0, "C"),
    ],
)
def test_tier_thresholds(settings, scores, expected_score, expected_tier):
    llm = FakeLLM({ICPFit: _fit_from(scores, urls=["https://a.example"])})
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)
    assert fit.score == expected_score
    assert fit.tier == expected_tier


def test_tier_for_boundaries():
    assert tier_for(100) == "A"
    assert tier_for(80) == "A"
    assert tier_for(79.9) == "B"
    assert tier_for(60) == "B"
    assert tier_for(59.9) == "C"
    assert tier_for(40) == "C"
    assert tier_for(39.9) == "D"
    assert tier_for(0) == "D"


def test_disqualifier_forces_tier_d_and_caps_score(settings):
    icp = _icp_4()
    fit_in = _fit_from(
        {"company_type": 10, "company_size": 10, "tech_maturity": 10, "growth_signals": 10},
        urls=["https://a.example"],
        dq=["it is a consumer-only B2C company"],  # paraphrased; must still match
    )
    llm = FakeLLM({ICPFit: fit_in})
    fit = score_icp(_known_record(), icp, [("https://a.example", "t")], llm, settings)

    assert fit.tier == "D"
    assert fit.score <= 25.0
    assert fit.disqualifiers_hit == icp.disqualifiers
    assert "DISQUALIFIED" in fit.summary


def test_model_cannot_invent_a_disqualifier(settings):
    fit_in = _fit_from(
        {"company_type": 10, "company_size": 10, "tech_maturity": 10, "growth_signals": 10},
        urls=["https://a.example"],
        dq=["their office is painted an unpleasant colour"],
    )
    llm = FakeLLM({ICPFit: fit_in})
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)

    assert fit.disqualifiers_hit == []
    assert fit.tier == "A"
    assert fit.score == 100.0


def test_confidence_is_high_when_evidence_and_record_are_complete(settings):
    llm = FakeLLM({ICPFit: _fit_from(
        {"company_type": 8, "company_size": 8, "tech_maturity": 8, "growth_signals": 8},
        urls=["https://a.example"],
    )})
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)
    # 0.6 * (4/4 criteria cited) + 0.4 * (8/8 record fields known) == 1.0
    assert fit.confidence == 1.0


def test_empty_record_is_a_non_assessment_not_a_middling_score(settings):
    """Regression: an all-None record scored a confident-looking 50/C.

    A reader treats 50/C as a considered judgement about the company. It was not
    one — nothing was researched — so the output has to be unmistakable.
    """
    empty = CompanyRecord(input="Mystery Co", company_name="Mystery Co")
    llm = FakeLLM({ICPFit: _fit_from(
        {"company_type": 8, "company_size": 8, "tech_maturity": 8, "growth_signals": 8},
        urls=[],
    )})
    fit = score_icp(empty, _icp_4(), [], llm, settings)

    assert (fit.score, fit.tier) != (50.0, "C"), "must not look like a real middling score"
    assert fit.score == 0.0
    assert fit.tier == "D"
    assert fit.confidence == 0.0
    assert "NO ASSESSMENT" in fit.summary
    assert all(c.score == 0.0 for c in fit.criteria)
    assert fit.disqualifiers_hit == []


def test_non_assessment_does_not_call_the_model_at_all(settings):
    llm = FakeLLM({ICPFit: _fit_from({"company_type": 10}, urls=[])})
    score_icp(CompanyRecord(input="Mystery Co"), _icp_4(), [], llm, settings)
    assert llm.calls == [], "scoring an empty record must not invoke the LLM"


@pytest.mark.parametrize(
    "record",
    [
        CompanyRecord(input="X"),                                   # nothing at all
        CompanyRecord(input="X", company_name="X"),                 # a name is not research
        CompanyRecord(input="X", company_name="X", description="blurb"),
    ],
)
def test_is_unresearched_detects_empty_records(record):
    from lea.enrich.icp import is_unresearched

    assert is_unresearched(record, [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domain": "ramp.com"},
        {"industry": "Fintech"},
        {"employee_range": "51-200"},
        {"hq_country": "USA"},
        {"signals": [Signal(kind="news", headline="A thing")]},
        {"tech_stack": [TechStack(category="crm", name="Salesforce")]},
    ],
)
def test_is_unresearched_is_false_once_anything_is_known(kwargs):
    from lea.enrich.icp import is_unresearched

    rec = CompanyRecord(input="X", company_name="X", sources=["https://x.example"], **kwargs)
    assert not is_unresearched(rec, [("https://x.example", "t")])


def test_partial_record_still_scores_normally_with_low_confidence(settings):
    """A genuinely partial record is what `confidence` is for — it still scores."""
    partial = CompanyRecord(
        input="Ramp", company_name="Ramp", domain="ramp.com", sources=["https://ramp.com"]
    )
    llm = FakeLLM({ICPFit: _fit_from(
        {"company_type": 8, "company_size": 8, "tech_maturity": 8, "growth_signals": 8},
        urls=[],  # no citations
    )})
    fit = score_icp(partial, _icp_4(), [("https://ramp.com", "some text")], llm, settings)

    # The arithmetic still runs, but an uncited grade is capped: each criterion
    # at 7/10, then the total by confidence. A record we barely established must
    # not present as tier A.
    assert fit.confidence < 0.4
    assert fit.score < 80.0
    assert "Low confidence" in fit.summary
    assert "capped" in fit.summary


def test_evidence_urls_are_filtered_to_urls_actually_gathered(settings):
    """An invented citation never survives; a real one may be derived in its place."""
    llm = FakeLLM({ICPFit: _fit_from(
        {"company_type": 8, "company_size": 8, "tech_maturity": 8, "growth_signals": 8},
        urls=["https://hallucinated.example"],
    )})
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)

    for c in fit.criteria:
        assert "https://hallucinated.example" not in c.evidence_urls
        # anything present must be a URL we actually fetched
        for u in c.evidence_urls:
            assert u == "https://a.example"


def test_score_icp_falls_back_to_the_rubric_when_the_model_fails(settings):
    """No flat 50/C. The record itself is graded, and the arithmetic is exact.

    _known_record() -> company_type 9.0 (fintech + Salesforce CRM), company_size
    10.0 (known 201-500 band), tech_maturity 4.7 (1 tech / 1 category + CRM),
    growth_signals 5.5 (one UNDATED funding signal: 2.0 base + 5.0 * 0.7 undated
    recency weight). Weights 2,1,1,1:
        (9.0*2 + 10.0 + 4.7 + 5.5) / 5 * 10 = 38.2 / 5 * 10 = 76.4
    """
    llm = FakeLLM(raise_on=ICPFit)
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)

    assert fit.score == 76.4
    assert fit.tier == "B"
    assert (fit.score, fit.tier) != (50.0, "C")
    assert all(len(c.reasoning) > 40 for c in fit.criteria)
    assert "deterministic rubric" in fit.summary


def test_model_scores_are_kept_and_only_gaps_are_rubric_graded(settings):
    """A partial model response is not discarded - the judgement it made stands.

    company_type 10.0 from the model; the other three rubric-graded as above:
        (10.0*2 + 10.0 + 4.7 + 5.5) / 5 * 10 = 40.2 / 5 * 10 = 80.4
    """
    fit_in = _fit_from({"company_type": 10}, urls=["https://a.example"])
    llm = FakeLLM({ICPFit: fit_in})
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)

    assert len(fit.criteria) == 4
    assert fit.score == 80.4
    by_key = {c.key: c for c in fit.criteria}
    assert by_key["company_type"].score == 10.0
    assert "skipped this criterion" in by_key["tech_maturity"].reasoning


# --------------------------------------------------------------------------
# ICP: the deterministic rubric must actually discriminate
# --------------------------------------------------------------------------


def _vanta_like() -> CompanyRecord:
    return CompanyRecord(
        input="Vanta", company_name="Vanta", domain="vanta.com", industry="Compliance",
        description="Vanta automates security compliance for companies.",
        hq_city="San Francisco", hq_country="USA", founded_year=2018,
        signals=[
            Signal(kind="funding", date="2024-07-24", recency_note="within 90 days",
                   headline="Vanta raises $150M Series C led by Sequoia at a $2.45B valuation",
                   source_url="https://vanta.com/blog/series-c"),
            Signal(kind="hiring", headline="See open roles across engineering, sales, and security",
                   source_url="https://vanta.com/careers"),
            Signal(kind="product", date="2025-01-10", headline="Introducing Vanta AI Agent",
                   source_url="https://vanta.com/blog"),
        ],
        tech_stack=[TechStack(category=c, name=n) for c, n in [
            ("ats", "Ashby"), ("crm", "HubSpot"), ("sales", "Qualified"), ("cms", "Webflow"),
            ("analytics", "GA4"), ("hosting", "Cloudflare"), ("analytics", "Segment"),
            ("support", "Intercom"), ("marketing", "Marketo"), ("data", "Snowflake"),
        ]],
        sources=["https://vanta.com", "https://vanta.com/blog/series-c",
                 "https://vanta.com/careers", "https://vanta.com/blog"],
    )


def _ramp_like() -> CompanyRecord:
    return CompanyRecord(
        input="Ramp", company_name="Ramp", domain="ramp.com", industry="Fintech",
        description="Ramp offers corporate cards and spend management.",
        hq_city="New York", hq_country="USA", founded_year=2019,
        signals=[
            Signal(kind="hiring", headline="12 open roles for account executives",
                   source_url="https://ramp.com/careers"),
            Signal(kind="product", date="2025-02-01", headline="Introducing Stack by Ramp.",
                   source_url="https://ramp.com/blog"),
            Signal(kind="product", date="2025-03-01", headline="Ramp launches Treasury",
                   source_url="https://ramp.com/blog"),
        ],
        tech_stack=[TechStack(category=c, name=n) for c, n in [
            ("analytics", "GA4"), ("hosting", "Cloudflare"),
            ("cms", "Webflow"), ("analytics", "Segment"),
        ]],
        sources=["https://ramp.com", "https://ramp.com/careers", "https://ramp.com/blog"],
    )


def _consumer_like() -> CompanyRecord:
    return CompanyRecord(
        input="Duolingo", company_name="Duolingo", domain="duolingo.com", industry="Consumer",
        description="Duolingo is a consumer language learning app.", hq_country="USA",
        tech_stack=[TechStack(category="analytics", name="GA4")],
        sources=["https://duolingo.com"],
    )


def _supporting(rec: CompanyRecord, extra: str = "") -> list[tuple[str, str]]:
    """Evidence text that genuinely contains this record's signals and fields.

    extract_record verifies provenance, so a fixture asserting on signals has to
    supply the text those signals came from - which is what a real run does.
    """
    text = " ".join(
        [s.headline or "" for s in rec.signals or []]
        + [s.detail or "" for s in rec.signals or []]
        + [str(x) for x in (rec.hq_city, rec.hq_country, rec.founded_year,
                            rec.employee_range) if x]
        + [extra]
    )
    urls = rec.sources or ["https://example.com"]
    return [(u, text) for u in urls]


def _stub_like_llm() -> "FakeLLM":
    """Mimics the keyless StubClient: an ICPFit with no per-criterion scores."""
    return FakeLLM({ICPFit: ICPFit(score=0.0, tier="C", summary="heuristic", criteria=[])})


def _ev(rec: CompanyRecord) -> list[tuple[str, str]]:
    return [(u, "evidence text") for u in rec.sources]


def test_two_rich_records_produce_different_scores(settings):
    """The regression: every company scored an identical 50.0/C on the demo path."""
    icp = DEFAULT_ICP
    vanta = score_icp(_vanta_like(), icp, _ev(_vanta_like()), _stub_like_llm(), settings)
    ramp = score_icp(_ramp_like(), icp, _ev(_ramp_like()), _stub_like_llm(), settings)

    assert vanta.score != ramp.score, "a lead scorer that gives every lead the same score is broken"
    assert vanta.score != 50.0 and ramp.score != 50.0
    # Vanta has funding + a CRM + 10 technologies; Ramp has neither funding nor a CRM.
    assert vanta.score > ramp.score


def test_a_rich_record_never_lands_on_the_neutral_default(settings):
    for rec in (_vanta_like(), _ramp_like()):
        fit = score_icp(rec, DEFAULT_ICP, _ev(rec), _stub_like_llm(), settings)
        assert (fit.score, fit.tier) != (50.0, "C")
        assert len({c.score for c in fit.criteria}) > 1, "criteria must not be uniform"


def test_consumer_company_scores_below_b2b_and_is_disqualified(settings):
    duo = _consumer_like()
    fit = score_icp(duo, DEFAULT_ICP, _ev(duo), _stub_like_llm(), settings)
    vanta = score_icp(_vanta_like(), DEFAULT_ICP, _ev(_vanta_like()), _stub_like_llm(), settings)

    assert fit.score < vanta.score
    assert fit.tier == "D"
    assert fit.disqualifiers_hit, "an explicit consumer-only company should trip a disqualifier"


def test_every_criterion_carries_real_reasoning_and_cites_evidence(settings):
    rec = _vanta_like()
    fit = score_icp(rec, DEFAULT_ICP, _ev(rec), _stub_like_llm(), settings)

    for c in fit.criteria:
        assert len(c.reasoning) > 40, f"{c.key} reasoning is not real analysis: {c.reasoning!r}"
        assert not c.reasoning.startswith("No evidence available")
        for u in c.evidence_urls:
            assert u in rec.sources           # citations stay filtered to real URLs
    # the criteria that had record evidence cite it
    by_key = {c.key: c for c in fit.criteria}
    assert by_key["growth_signals"].evidence_urls
    assert "Series C" in by_key["growth_signals"].reasoning


def test_rubric_scores_respond_to_the_evidence_that_drives_them(settings):
    """Removing the funding signal must move growth_signals, and only it."""
    rich = _vanta_like()
    lean = _vanta_like()
    lean.signals = [s for s in lean.signals if s.kind != "funding"]

    a = {c.key: c.score for c in score_icp(rich, DEFAULT_ICP, _ev(rich), _stub_like_llm(), settings).criteria}
    b = {c.key: c.score for c in score_icp(lean, DEFAULT_ICP, _ev(lean), _stub_like_llm(), settings).criteria}

    assert b["growth_signals"] < a["growth_signals"]
    assert b["geo"] == a["geo"] and b["tech_maturity"] == a["tech_maturity"]


def test_unknown_headcount_scores_mid_not_zero(settings):
    """Neither a band nor a job board: genuinely unknown, scored mid, said plainly."""
    rec = _ramp_like()
    rec.signals = [s for s in rec.signals if s.kind != "hiring"]   # remove the board evidence
    assert rec.employee_range is None

    fit = score_icp(rec, DEFAULT_ICP, _ev(rec), _stub_like_llm(), settings)
    size = next(c for c in fit.criteria if c.key == "company_size")
    assert size.score == 5.0
    assert "genuinely unknown" in size.reasoning
    assert size.evidence_urls == []


def test_degenerate_detection(settings):
    from lea.enrich.icp import _is_degenerate

    icp = _icp_4()
    assert _is_degenerate(None, icp)
    assert _is_degenerate(ICPFit(score=0, tier="C", criteria=[]), icp)          # the stub
    nothing_matches = ICPFit(score=0, tier="C", criteria=[
        CriterionScore(key="unrelated_key", score=5.0, weight=1.0, reasoning="x")])
    assert _is_degenerate(nothing_matches, icp)                                # nothing maps on
    # identical score AND identical reasoning == default-filled
    same = ICPFit(score=0, tier="C", criteria=[
        CriterionScore(key=k, score=5.0, weight=1.0, reasoning="n/a")
        for k in ("company_type", "company_size", "tech_maturity", "growth_signals")])
    assert _is_degenerate(same, icp)
    # identical scores with distinct reasoning is a legitimate judgement
    assert not _is_degenerate(
        _fit_from({"company_type": 8, "company_size": 8, "tech_maturity": 8, "growth_signals": 8}), icp
    )


def test_deterministic_scores_are_pure_and_repeatable(settings):
    from lea.enrich.icp import deterministic_scores

    rec = _vanta_like()
    a = deterministic_scores(rec, DEFAULT_ICP)
    b = deterministic_scores(rec, DEFAULT_ICP)
    assert [(c.key, c.score) for c in a] == [(c.key, c.score) for c in b]


# --------------------------------------------------------------------------
# ICP: inference
# --------------------------------------------------------------------------


def test_infer_icp_repairs_a_sloppy_model_response(settings):
    sloppy = ICPProfile(
        name="",
        summary="",
        criteria=[
            ICPCriterion(key="Company Type!", description="b2b", weight=-3.0),
            ICPCriterion(key="companyType", description="dupe key", weight=2.0),
        ],
        disqualifiers=["  ", "Consumer only"],
    )
    llm = FakeLLM({ICPProfile: sloppy})
    prof = infer_icp(["Ramp", "Vanta", "Retool"], llm, settings, notes="mid-market")

    keys = [c.key for c in prof.criteria]
    assert len(prof.criteria) >= 4                  # padded up to the minimum
    assert len(keys) == len(set(keys))              # unique
    assert all(k == k.lower() and " " not in k for k in keys)   # snake_case
    assert all(c.weight > 0 for c in prof.criteria)             # negative weight repaired
    assert prof.derived_from == ["Ramp", "Vanta", "Retool"]
    assert prof.name
    assert prof.summary
    assert prof.disqualifiers == ["Consumer only"]


def test_infer_icp_with_no_examples_returns_the_default(settings):
    llm = FakeLLM()
    prof = infer_icp([], llm, settings)
    assert len(prof.criteria) == len(DEFAULT_ICP.criteria)
    assert prof.derived_from == []


def test_infer_icp_survives_model_failure(settings):
    llm = FakeLLM(raise_on=ICPProfile)
    prof = infer_icp(["Ramp", "Vanta"], llm, settings)
    assert len(prof.criteria) >= 4
    assert prof.derived_from == ["Ramp", "Vanta"]


# --------------------------------------------------------------------------
# ICP: persistence
# --------------------------------------------------------------------------


def test_icp_yaml_round_trips(tmp_path):
    original = _icp_4()
    p = tmp_path / "icp.yaml"
    save_icp(original, p)

    text = p.read_text(encoding="utf-8")
    assert text.startswith("#")            # human-readable header comment survives
    assert "lea icp infer" in text

    loaded = load_icp(p)
    assert loaded.name == original.name
    assert loaded.disqualifiers == original.disqualifiers
    assert [(c.key, c.weight, c.description) for c in loaded.criteria] == \
           [(c.key, c.weight, c.description) for c in original.criteria]


def test_load_or_default_icp_prefers_explicit_path(tmp_path, settings):
    p = tmp_path / "custom.yaml"
    save_icp(_icp_4(), p)
    assert load_or_default_icp(str(p), settings).name == "test-icp"


def test_load_or_default_icp_never_raises(tmp_path, settings):
    missing = tmp_path / "nope.yaml"
    assert load_or_default_icp(str(missing), settings).name == DEFAULT_ICP.name

    broken = tmp_path / "broken.yaml"
    broken.write_text("this: [is: not: valid: yaml", encoding="utf-8")
    assert load_or_default_icp(str(broken), settings).name == DEFAULT_ICP.name


def test_default_icp_is_well_formed():
    assert len(DEFAULT_ICP.criteria) >= 4
    assert all(c.weight > 0 and c.rubric for c in DEFAULT_ICP.criteria)
    assert 2 <= len(DEFAULT_ICP.disqualifiers) <= 3
    assert "PLACEHOLDER" in DEFAULT_ICP.summary


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------


def _hook_record() -> CompanyRecord:
    return CompanyRecord(
        input="Ramp",
        company_name="Ramp",
        domain="ramp.com",
        industry="Fintech",
        employee_range="201-500",
        signals=[
            Signal(kind="funding", headline="Raised a $150M Series D led by Founders Fund",
                   date="2025-04-02", source_url="https://techcrunch.com/ramp"),
            Signal(kind="hiring", headline="Opened 12 enterprise AE roles in New York",
                   date="2025-05-11", source_url="https://ramp.com/careers"),
        ],
        tech_stack=[TechStack(category="crm", name="Salesforce", evidence="job posting")],
        sources=["https://techcrunch.com/ramp", "https://ramp.com/careers"],
    )


def test_is_grounded_accepts_real_facts_and_rejects_invented_ones():
    rec = _hook_record()
    assert is_grounded("Series D led by Founders Fund", rec)
    assert is_grounded("12 enterprise AE roles", rec)
    assert not is_grounded("acquisition by Microsoft", rec)
    assert not is_grounded("IPO filed last quarter", rec)
    assert not is_grounded("", rec)


def test_filter_drops_cliches_length_and_ungrounded_claims():
    rec = _hook_record()
    good = OutreachHook(
        angle="Series D -> GTM headcount",
        hook="You closed a $150M Series D in April and are hiring 12 AEs. That is usually the "
             "month the CRM stops keeping up with the pipeline.",
        grounded_in="Raised a $150M Series D led by Founders Fund",
        source_url="https://techcrunch.com/ramp",
    )
    cliche = OutreachHook(
        angle="cliche",
        hook="Hope this finds you well! I was impressed by your Series D and wanted to touch base.",
        grounded_in="Raised a $150M Series D led by Founders Fund",
    )
    too_long = OutreachHook(
        angle="rambling",
        hook=" ".join(["word"] * (MAX_HOOK_WORDS + 5)) + " Series D Founders Fund",
        grounded_in="Raised a $150M Series D led by Founders Fund",
    )
    ungrounded = OutreachHook(
        angle="invented",
        hook="Congrats on the Microsoft acquisition, big moment for the team.",
        grounded_in="acquisition by Microsoft",
    )

    kept, dropped = filter_hooks([good, cliche, too_long, ungrounded], rec)

    assert [h.angle for h in kept] == ["Series D -> GTM headcount"]
    assert len(dropped) == 3
    assert any("cliche" in d for d in dropped)
    assert any("too long" in d for d in dropped)
    assert any("ungrounded" in d for d in dropped)


def test_every_cliche_in_the_list_is_actually_caught():
    rec = _hook_record()
    for phrase in CLICHES:
        h = OutreachHook(
            angle="x",
            hook=f"{phrase} — about your Series D.",
            grounded_in="Raised a $150M Series D led by Founders Fund",
        )
        kept, dropped = filter_hooks([h], rec)
        assert kept == [], phrase
        assert dropped, phrase


def test_generate_hooks_fallback_fires_when_everything_is_filtered(settings):
    rec = _hook_record()
    from lea.enrich.hooks import _HookBundle

    all_bad = _HookBundle(hooks=[
        OutreachHook(angle="a", hook="Hope this finds you well.", grounded_in="Series D"),
        OutreachHook(angle="b", hook="Congrats on the IPO.", grounded_in="IPO on Nasdaq"),
    ])
    llm = FakeLLM({_HookBundle: all_bad})
    hooks = generate_hooks(rec, None, [], llm, settings)

    assert 2 <= len(hooks) <= 3
    for h in hooks:
        assert h.grounded_in
        assert is_grounded(h.grounded_in, rec)
        assert len(h.hook.split()) <= MAX_HOOK_WORDS
        assert not any(c in h.hook.lower() for c in CLICHES)


def test_generate_hooks_keeps_good_ones_and_tops_up(settings):
    rec = _hook_record()
    from lea.enrich.hooks import _HookBundle

    bundle = _HookBundle(hooks=[
        OutreachHook(
            angle="Series D",
            hook="A $150M Series D and 12 open AE roles in the same quarter — the reporting "
                 "usually breaks before the headcount lands.",
            grounded_in="Raised a $150M Series D led by Founders Fund",
            source_url="https://techcrunch.com/ramp",
        ),
        OutreachHook(angle="bad", hook="Quick question about your roadmap.", grounded_in="Series D"),
    ])
    llm = FakeLLM({_HookBundle: bundle})
    hooks = generate_hooks(rec, None, [], llm, settings)

    assert len(hooks) >= 2
    assert hooks[0].angle == "Series D"
    assert hooks[0].source_url == "https://techcrunch.com/ramp"


def test_generate_hooks_never_returns_empty_even_with_a_bare_record(settings):
    bare = CompanyRecord(input="Mystery Co", company_name="Mystery Co")
    llm = FakeLLM(raise_on=None)
    hooks = generate_hooks(bare, None, [], llm, settings)
    assert len(hooks) >= 2
    assert all(h.hook.strip() for h in hooks)


def test_fallback_hooks_on_an_empty_record_assert_nothing_about_the_company(settings):
    """With nothing researched, a hook must not imply a fact or a judgement."""
    bare = CompanyRecord(input="Ramp", company_name="Ramp")
    fit = score_icp(bare, _icp_4(), [], FakeLLM(), settings)   # a non-assessment
    assert fit.score == 0.0

    hooks = generate_hooks(bare, fit, [], FakeLLM(), settings)

    assert len(hooks) >= 2
    joined = " ".join(h.hook.lower() for h in hooks)
    # no invented facts, and no leaking of the non-assessment as a verdict
    for claim in ["series", "raised", "funding", "hiring", "employees", "salesforce",
                  "tier ", "scored", "/100"]:
        assert claim not in joined, f"fallback hook asserts {claim!r} with no evidence"
    for h in hooks:
        assert h.grounded_in in {"no public signal found"}
        assert not any(c in h.hook.lower() for c in CLICHES)


def test_fallback_hooks_use_signal_specifics_not_the_raw_headline(settings):
    """Regression: 'Noticed Ramp's Introducing Stack by Ramp.. Usually that means...'"""
    rec = _vanta_like()
    hooks = generate_hooks(rec, None, [], _stub_like_llm(), settings)

    joined = " ".join(h.hook for h in hooks)
    assert ".." not in joined, "punctuation splice from pasting a headline verbatim"
    # the specifics, not the headline
    assert "$150M" in joined or "Series C" in joined
    assert "Sequoia" in joined
    assert "Vanta raises $150M Series C led by Sequoia" not in joined, "headline pasted verbatim"


def test_fallback_hooks_for_different_companies_are_not_the_same_sentence(settings):
    a = generate_hooks(_vanta_like(), None, [], _stub_like_llm(), settings)
    b = generate_hooks(_ramp_like(), None, [], _stub_like_llm(), settings)

    def skeleton(h):
        """The hook with company-specific tokens stripped — what the template is."""
        return re.sub(r"[A-Z][\w$.]*|\d+", "", h.hook)

    assert {skeleton(x) for x in a}.isdisjoint({skeleton(y) for y in b}), \
        "two different companies got the same scaffold sentence"


def test_two_hooks_for_one_company_differ_structurally(settings):
    hooks = generate_hooks(_vanta_like(), None, [], _stub_like_llm(), settings)
    assert len(hooks) >= 2
    assert len({h.angle for h in hooks}) == len(hooks)
    # no shared long tail — the old scaffold was identical after the first clause
    tails = [" ".join(h.hook.split()[-8:]) for h in hooks]
    assert len(set(tails)) == len(tails)


def test_the_old_template_scaffold_is_now_rejected_by_the_filter():
    """So the exact regression cannot come back."""
    rec = _ramp_like()
    old = OutreachHook(
        angle="hiring",
        hook="Noticed Ramp's 12 open roles for account executives. Usually that means the GTM "
             "team is about to feel the process gaps before the tooling catches up — worth a look?",
        grounded_in="12 open roles for account executives",
    )
    kept, dropped = filter_hooks([old], rec)
    assert kept == []
    assert any("cliche" in d for d in dropped)


def test_generated_fallback_hooks_survive_our_own_filter(settings):
    """The fallbacks bypass filter_hooks — they must still be clean enough to pass it."""
    for rec in (_vanta_like(), _ramp_like(), _consumer_like()):
        hooks = generate_hooks(rec, None, [], _stub_like_llm(), settings)
        kept, dropped = filter_hooks(hooks, rec)
        assert len(kept) == len(hooks), f"our own fallback was rejected: {dropped}"


def test_fallback_hooks_use_the_icp_fit_only_when_it_is_a_real_assessment(settings):
    rec = CompanyRecord(input="Ramp", company_name="Ramp", domain="ramp.com",
                        sources=["https://a.example"])
    real_fit = ICPFit(score=84.0, tier="A", summary="strong", confidence=0.8,
                      criteria=[CriterionScore(key="growth_signals", score=9, weight=1.5)])
    hooks = generate_hooks(rec, real_fit, [], FakeLLM(), settings)
    assert any("tier A" in h.hook for h in hooks)


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def test_read_targets_csv_with_header(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text(
        "Company,Note\n"
        "Ramp,fit\n"
        "  Vanta  ,fit\n"
        "\n"
        "Ramp,duplicate\n"
        "Duolingo,miss\n",
        encoding="utf-8",
    )
    assert read_targets(p) == ["Ramp", "Vanta", "Duolingo"]


def test_read_targets_picks_the_right_column(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("id,website,owner\n1,https://ramp.com,ana\n2,https://vanta.com,bo\n", encoding="utf-8")
    assert read_targets(p) == ["https://ramp.com", "https://vanta.com"]


def test_read_targets_bare_list(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text("# my list\nRamp\n\nVanta\n  Retool  \nRamp\n", encoding="utf-8")
    assert read_targets(p) == ["Ramp", "Vanta", "Retool"]


def test_read_targets_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("\n\n# nothing\n", encoding="utf-8")
    assert read_targets(p) == []


def test_shipped_examples_csv_parses():
    from pathlib import Path as _P

    p = _P(__file__).resolve().parent.parent / "examples" / "companies.csv"
    targets = read_targets(p)
    assert len(targets) >= 8
    assert "Ramp" in targets
    assert "note" not in [t.lower() for t in targets]


def _result(name: str, score: float, tier: str) -> EnrichmentResult:
    rec = _hook_record()
    rec.input = name
    rec.company_name = name
    rec.unknowns = ["founded_year"]
    rec.icp_fit = ICPFit(score=score, tier=tier, summary="ok", confidence=0.8,
                         criteria=[CriterionScore(key="company_type", score=8, weight=1.0)])
    rec.hooks = [
        OutreachHook(angle="a", hook="first hook", grounded_in="Series D"),
        OutreachHook(angle="b", hook="second hook", grounded_in="AE roles"),
    ]
    return EnrichmentResult(record=rec, trace=[], stats=RunStats(steps=7, stop_reason="gaps_closed"))


def test_write_csv_header_and_rows(tmp_path):
    out = tmp_path / "nested" / "out.csv"
    write_csv([_result("Ramp", 82.0, "A"), _result("Duolingo", 21.0, "D")], out)

    import csv as _csv

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
        fieldnames = _csv.DictReader(out.open(newline="", encoding="utf-8")).fieldnames

    assert fieldnames == CSV_COLUMNS
    assert len(rows) == 2
    r = rows[0]
    assert r["input"] == "Ramp"
    assert r["domain"] == "ramp.com"
    assert r["icp_score"] == "82.0"
    assert r["icp_tier"] == "A"
    assert r["signal_count"] == "2"
    assert r["tech_stack"] == "crm:Salesforce"
    assert r["hook_1"] == "first hook"
    assert r["hook_2"] == "second hook"
    assert r["hook_3"] == ""              # padded, not missing
    assert r["unknowns"] == "founded_year"
    assert r["sources"] == "https://techcrunch.com/ramp; https://ramp.com/careers"
    assert r["steps"] == "7"
    assert r["stop_reason"] == "gaps_closed"
    assert r["top_signal"] == "Raised a $150M Series D led by Founders Fund"


def test_write_csv_empty_results_still_writes_header(tmp_path):
    out = tmp_path / "out.csv"
    write_csv([], out)
    assert out.read_text(encoding="utf-8").strip() == ",".join(CSV_COLUMNS)


def test_write_csv_round_trips_through_read_targets(tmp_path):
    out = tmp_path / "out.csv"
    write_csv([_result("Ramp", 82.0, "A"), _result("Vanta", 71.0, "B")], out)
    # the output's `company_name` column is a valid input list for a re-run
    assert read_targets(out) == ["Ramp", "Vanta"]


# --------------------------------------------------------------------------
# Job-board evidence (Agent A's fetch_jobs) and signal recency
# --------------------------------------------------------------------------

JOBS_JSON = (
    '{"open_roles": 47, "departments": {"Engineering": 21, "Sales": 9, "Security": 6, '
    '"Marketing": 4}, "locations": ["Remote", "SF"], "ats": "ashby", '
    '"board_url": "https://jobs.ashbyhq.com/vanta"}'
)
# Exactly what Agent A's fetch_jobs `render_summary` emits — the loop flattens
# observations to (url, summary), so this string is what actually reaches us.
JOBS_PROSE = (
    "Vanta — 47 open roles on Ashby across Engineering (21), Sales (9), Security (6); "
    "locations incl. Remote, SF; 12 remote; 9 posted in the last 30 days "
    "(latest 2026-09-10). Board: https://jobs.ashbyhq.com/vanta"
)
JOBS_COLON = "Vanta: 47 open roles. Engineering: 21, Sales: 9, Security: 6."


@pytest.mark.parametrize("text", [JOBS_JSON, JOBS_PROSE, JOBS_COLON])
def test_parse_jobs_evidence_handles_every_shape_the_tool_emits(text):
    """JSON payload, Agent A's rendered summary, and a colon-separated reword."""
    from lea.enrich.extract import parse_jobs_evidence

    jobs = parse_jobs_evidence([("https://jobs.ashbyhq.com/vanta", text)])
    assert jobs is not None
    assert jobs.open_roles == 47
    assert jobs.departments["Sales"] == 9
    assert jobs.departments["Security"] == 6
    assert jobs.ats == "ashby"
    assert jobs.gtm_roles >= 9          # Sales, plus Marketing when present


def test_parse_jobs_picks_up_posting_recency_from_the_summary():
    """`latest_posted` makes the hiring signal properly dated, not just 'live'."""
    from lea.enrich.extract import parse_jobs_evidence

    jobs = parse_jobs_evidence([("https://jobs.ashbyhq.com/vanta", JOBS_PROSE)])
    assert jobs.latest_posted == "2026-09-10"
    assert jobs.posted_recently == 9


def test_job_board_signal_is_dated_from_the_latest_posting(settings):
    from lea.enrich.extract import signal_age_days

    llm = FakeLLM({CompanyRecord: CompanyRecord(input="Vanta", company_name="Vanta")})
    rec = extract_record("Vanta", [("https://jobs.ashbyhq.com/vanta", JOBS_PROSE)], llm, settings)
    sig = next(s for s in rec.signals if s.kind == "hiring")

    assert sig.date == "2026-09-10"
    assert signal_age_days(sig, TODAY) == 9        # fresh, not merely "undated"


def test_parse_jobs_evidence_ignores_ordinary_pages():
    from lea.enrich.extract import parse_jobs_evidence

    assert parse_jobs_evidence([("https://vanta.com", "We build compliance software.")]) is None
    assert parse_jobs_evidence([]) is None


def test_open_roles_never_fabricate_an_employee_range(settings):
    """The honesty rule: a hiring count is not a headcount."""
    llm = FakeLLM({CompanyRecord: CompanyRecord(input="Vanta", company_name="Vanta")})
    rec = extract_record("Vanta", [("https://jobs.ashbyhq.com/vanta", JOBS_JSON)], llm, settings)

    assert rec.employee_range is None
    assert rec.employee_estimate is None
    assert "employee_range" in rec.unknowns


def test_job_board_becomes_a_hiring_signal_with_counts(settings):
    llm = FakeLLM({CompanyRecord: CompanyRecord(input="Vanta", company_name="Vanta")})
    rec = extract_record("Vanta", [("https://jobs.ashbyhq.com/vanta", JOBS_JSON)], llm, settings)

    hiring = [s for s in rec.signals if s.kind == "hiring"]
    assert len(hiring) == 1
    sig = hiring[0]
    assert "47 open roles" in sig.headline
    assert "9 in Sales" in sig.detail
    assert sig.source_url == "https://jobs.ashbyhq.com/vanta"
    assert sig.source_url in rec.sources


def test_job_board_signal_round_trips_back_out_of_the_record(settings):
    """icp.py and hooks.py read the counts back via the signal, not a new field."""
    from lea.enrich.extract import jobs_from_record

    llm = FakeLLM({CompanyRecord: CompanyRecord(input="Vanta", company_name="Vanta")})
    rec = extract_record("Vanta", [("https://jobs.ashbyhq.com/vanta", JOBS_JSON)], llm, settings)

    back = jobs_from_record(rec)
    assert back is not None
    assert back.open_roles == 47
    assert back.departments["Sales"] == 9
    assert back.gtm_roles == 13


def test_counted_board_replaces_a_vague_hiring_signal(settings):
    vague = CompanyRecord(
        input="Vanta", company_name="Vanta",
        signals=[Signal(kind="hiring", headline="See open positions",
                        source_url="https://vanta.com/careers")],
    )
    llm = FakeLLM({CompanyRecord: vague})
    rec = extract_record("Vanta", [("https://jobs.ashbyhq.com/vanta", JOBS_JSON)], llm, settings)

    headlines = [s.headline for s in rec.signals if s.kind == "hiring"]
    assert headlines == ["47 open roles (21 in Engineering, 9 in Sales, 6 in Security, 4 in Marketing)"]
    assert "See open positions" not in headlines


def test_company_size_grades_on_open_roles_and_cites_the_board(settings):
    """WEAKNESS 1: the criterion must stop defaulting to 5.0 with no evidence."""
    llm = FakeLLM({CompanyRecord: CompanyRecord(input="Vanta", company_name="Vanta",
                                                domain="vanta.com", industry="Compliance")})
    ev = [("https://jobs.ashbyhq.com/vanta", JOBS_JSON), ("https://vanta.com", "compliance")]
    rec = extract_record("Vanta", ev, llm, settings)
    fit = score_icp(rec, DEFAULT_ICP, ev, rec and _stub_like_llm(), settings)

    size = next(c for c in fit.criteria if c.key == "company_size")
    assert size.score != 5.0, "must not default to mid when a job board was read"
    assert size.score >= 9.0                       # 47 roles == scaling through the band
    assert "47 open roles" in size.reasoning
    assert "unconfirmed" in size.reasoning          # still honest about headcount
    assert size.evidence_urls == ["https://jobs.ashbyhq.com/vanta"]


def test_reading_a_job_board_raises_confidence(settings):
    """WEAKNESS 1: confidence was stuck at 0.30-0.35 because size was never evidenced."""
    base = CompanyRecord(input="Vanta", company_name="Vanta", domain="vanta.com",
                         industry="Compliance")
    plain_ev = [("https://vanta.com", "We build compliance software.")]
    board_ev = plain_ev + [("https://jobs.ashbyhq.com/vanta", JOBS_JSON)]

    without = score_icp(
        extract_record("Vanta", plain_ev, FakeLLM({CompanyRecord: base}), settings),
        DEFAULT_ICP, plain_ev, _stub_like_llm(), settings)
    with_board = score_icp(
        extract_record("Vanta", board_ev, FakeLLM({CompanyRecord: base}), settings),
        DEFAULT_ICP, board_ev, _stub_like_llm(), settings)

    assert with_board.confidence > without.confidence


# --- recency --------------------------------------------------------------

TODAY = datetime(2026, 9, 19, tzinfo=timezone.utc)


def _days_ago(days: int) -> str:
    """A date string ``days`` before the real today.

    Code paths that take an explicit ``today`` (``signal_age_days``,
    ``_freshest_signal_phrase``) can be pinned to :data:`TODAY` and stay
    deterministic.  ``score_icp`` reads the real clock, so any fixture that
    flows through it must be relative — a hardcoded date passes on the day
    it is written and fails every day after.
    """
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


@pytest.mark.parametrize(
    "date,expected_days",
    [
        ("2026-09-10", 9),
        ("2026-09-03", 16),
        ("2026-08-19", 31),
        ("2024-07-24", 787),
        (None, None),
    ],
)
def test_signal_age_days(date, expected_days):
    from lea.enrich.extract import signal_age_days

    sig = Signal(kind="product", headline="x", date=date)
    assert signal_age_days(sig, TODAY) == expected_days


def test_recency_weight_curve():
    from lea.enrich.extract import recency_weight

    assert recency_weight(9) == 1.0
    assert recency_weight(60) == 0.9
    assert recency_weight(400) == 0.35
    assert recency_weight(800) == 0.2
    # undated counts for something, and is NOT the worst case
    assert recency_weight(None) == 0.7
    assert recency_weight(None) > recency_weight(400)
    assert recency_weight(None) < recency_weight(9)


def test_fresh_signals_outscore_stale_ones(settings):
    """WEAKNESS 2: a 9-day-old launch and a 2-year-old round must not tie."""
    fresh = CompanyRecord(
        input="Linear", company_name="Linear", domain="linear.app", industry="SaaS",
        sources=["https://linear.app/changelog"],
        signals=[
            Signal(kind="product", headline="Linear ships Agents", date=_days_ago(9),
                   source_url="https://linear.app/changelog"),
            Signal(kind="product", headline="Customer Requests GA", date=_days_ago(16),
                   source_url="https://linear.app/changelog"),
        ],
    )
    stale = CompanyRecord(
        input="Oldco", company_name="Oldco", domain="oldco.com", industry="SaaS",
        sources=["https://oldco.com/news"],
        signals=[
            Signal(kind="funding", headline="Oldco raises $40M Series B", date=_days_ago(787),
                   source_url="https://oldco.com/news"),
        ],
    )
    ev_f = [("https://linear.app/changelog", "t")]
    ev_s = [("https://oldco.com/news", "t")]

    gf = next(c for c in score_icp(fresh, DEFAULT_ICP, ev_f, _stub_like_llm(), settings).criteria
              if c.key == "growth_signals")
    gs = next(c for c in score_icp(stale, DEFAULT_ICP, ev_s, _stub_like_llm(), settings).criteria
              if c.key == "growth_signals")

    assert gf.score > gs.score, "a 9-day-old launch must beat a 2-year-old round"
    assert "9 days old" in gf.reasoning
    assert "this week" in gf.reasoning
    assert "over a year old" in gs.reasoning


def test_freshest_signal_is_surfaced_in_the_summary(settings):
    rec = CompanyRecord(
        input="Linear", company_name="Linear", domain="linear.app", industry="SaaS",
        sources=["https://linear.app/changelog"],
        signals=[Signal(kind="product", headline="Linear ships Agents", date="2026-09-10",
                        source_url="https://linear.app/changelog")],
    )
    from lea.enrich.icp import _freshest_signal_phrase

    phrase = _freshest_signal_phrase(rec, TODAY)
    assert "Linear ships Agents" in phrase
    assert "9 days old" in phrase
    # a two-year-old signal is not "freshest" material at all
    old = rec.model_copy(deep=True)
    old.signals[0].date = "2024-07-24"
    assert _freshest_signal_phrase(old, TODAY) == ""


def test_signals_are_sorted_freshest_first(settings):
    rec = CompanyRecord(
        input="Linear", company_name="Linear",
        signals=[
            Signal(kind="product", headline="Oldest", date="2026-08-19"),
            Signal(kind="product", headline="Newest", date="2026-09-10"),
            Signal(kind="product", headline="Undated"),
            Signal(kind="product", headline="Middle", date="2026-09-03"),
        ],
    )
    llm = FakeLLM({CompanyRecord: rec})
    out = extract_record("Linear", [("https://linear.app", "Oldest Newest Undated Middle")],
                         llm, settings)
    assert [s.headline for s in out.signals] == ["Newest", "Middle", "Oldest", "Undated"]


def test_hooks_lead_with_the_gtm_department_mix(settings):
    """WEAKNESS 1: '9 open sales roles' is the best hook material we have."""
    llm = FakeLLM({CompanyRecord: CompanyRecord(input="Vanta", company_name="Vanta")})
    rec = extract_record("Vanta", [("https://jobs.ashbyhq.com/vanta", JOBS_JSON)], llm, settings)
    hooks = generate_hooks(rec, None, [], _stub_like_llm(), settings)

    first = hooks[0]
    assert "9 in sales" in first.hook.lower()
    assert "47" in first.hook
    assert first.source_url == "https://jobs.ashbyhq.com/vanta"
    assert is_grounded(first.grounded_in, rec)
    assert len(first.hook.split()) <= MAX_HOOK_WORDS


def test_counted_board_also_absorbs_the_models_restatement(settings):
    """Live Gemini restated the board total as a second hiring signal."""
    dup = CompanyRecord(
        input="Vanta", company_name="Vanta",
        signals=[
            Signal(kind="hiring", headline="Vanta has 47 open roles on Ashby",
                   source_url="https://jobs.ashbyhq.com/vanta"),
            Signal(kind="hiring", headline="See open positions",
                   source_url="https://vanta.com/careers"),
            Signal(kind="hiring", headline="Hiring 3 principal engineers in Dublin",
                   source_url="https://vanta.com/careers"),
        ],
    )
    llm = FakeLLM({CompanyRecord: dup})
    ev = [("https://jobs.ashbyhq.com/vanta", JOBS_JSON),
          ("https://vanta.com/careers",
           "Vanta has 47 open roles on Ashby. See open positions. "
           "Hiring 3 principal engineers in Dublin.")]
    rec = extract_record("Vanta", ev, llm, settings)

    hiring = [s.headline for s in rec.signals if s.kind == "hiring"]
    assert len(hiring) == 2
    assert hiring[0].startswith("47 open roles (")        # ours, with the breakdown
    # a hiring signal about something else survives
    assert "Hiring 3 principal engineers in Dublin" in hiring


def test_scoring_context_hands_the_model_the_date_and_signal_ages():
    """The model must not do date arithmetic: live Gemini called June 2025
    'well within the 90-day recency window' on a run dated September 2026."""
    from lea.enrich.icp import _build_scoring_context

    rec = CompanyRecord(
        input="Ramp", company_name="Ramp",
        signals=[Signal(kind="funding", headline="Ramp valued at $13B", date="2025-03-01")],
    )
    ctx = _build_scoring_context(rec, _icp_4(), [])
    assert "TODAY'S DATE:" in ctx
    assert "SIGNAL AGES" in ctx
    assert "do not recalculate" in ctx
    assert "Ramp valued at $13B" in ctx
    assert "months old" in ctx or "years old" in ctx


# --------------------------------------------------------------------------
# Funding re-classification: real headlines rarely say "raised"
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headline,expected",
    [
        # the genuine TechCrunch headline that was landing as a product signal
        ("Ramp nearly doubles valuation to $32B in secondary share sale", "secondary"),
        ("Acme announces a tender offer for employees", "secondary"),
        ("Stripe employees sell shares in a $6.5B secondary sale", "secondary"),
        # valuation phrasing with no raise verb anywhere
        ("Figma valued at $12.5 billion after investor round", "valuation"),
        ("Anthropic's valuation hits $60B post-money", "valuation"),
        # primary capital, in the many ways it gets written
        ("Vanta raises $150M Series C led by Sequoia", "primary"),
        ("Acme closes $40M Series B", "primary"),
        ("Acme secures $12M led by Index Ventures", "primary"),
        ("Startup lands $5M seed round", "primary"),
        ("Acme banks $30M in growth financing", "primary"),
        ("Acme nets $8M investment from Accel", "primary"),
        ("Deel picks up $50M in Series D funding", "primary"),
        # and the things that must NOT trip it
        ("Introducing Stack by Ramp.", None),
        ("Cleared 50+ products for launch last year", None),
        ("We serve a $13B market opportunity", None),
        ("Hiring 12 account executives", None),
        ("Launched our first International market", None),
        ("Our platform saved customers $200 million last year", None),
    ],
)
def test_classify_funding_covers_real_world_phrasing(headline, expected):
    from lea.enrich.extract import classify_funding

    assert classify_funding(headline) == expected


def test_valuation_headline_is_reclassified_from_product_to_funding(settings):
    """The exact regression: a $32B valuation event filed as a product launch."""
    rec = CompanyRecord(
        input="Ramp", company_name="Ramp",
        signals=[
            Signal(kind="product",
                   headline="Ramp nearly doubles valuation to $32B in secondary share sale",
                   source_url="https://techcrunch.com/ramp"),
            Signal(kind="product", headline="Introducing Stack by Ramp.",
                   source_url="https://ramp.com/blog"),
        ],
    )
    llm = FakeLLM({CompanyRecord: rec})
    out = extract_record("Ramp", _supporting(rec), llm, settings)

    by_head = {s.headline: s for s in out.signals}
    assert by_head["Ramp nearly doubles valuation to $32B in secondary share sale"].kind == "funding"
    # a genuine product launch is left alone
    assert by_head["Introducing Stack by Ramp."].kind == "product"


def test_secondary_sale_is_labelled_as_a_liquidity_event(settings):
    """Honest taxonomy: it scores as funding but says no new capital arrived."""
    from lea.enrich.extract import SECONDARY_NOTE, is_secondary_funding

    rec = CompanyRecord(
        input="Ramp", company_name="Ramp",
        signals=[Signal(kind="product",
                        headline="Ramp nearly doubles valuation to $32B in secondary share sale",
                        source_url="https://techcrunch.com/ramp")],
    )
    llm = FakeLLM({CompanyRecord: rec})
    out = extract_record("Ramp", _supporting(rec), llm, settings)
    sig = out.signals[0]

    assert sig.kind == "funding"
    assert SECONDARY_NOTE in sig.detail
    assert "rather than new primary capital" in sig.detail
    assert is_secondary_funding(sig)


def test_secondary_scores_below_an_equivalent_primary_round(settings):
    """No new capital reached the company, so it creates less new budget."""
    def fit_for(headline: str) -> float:
        rec = CompanyRecord(
            input="Co", company_name="Co", domain="co.com", industry="SaaS",
            sources=["https://news.example"],
            signals=[Signal(kind="funding", headline=headline, date=_days_ago(9),
                            source_url="https://news.example")],
        )
        from lea.enrich.extract import reclassify_signals
        rec.signals = reclassify_signals(rec.signals)
        fit = score_icp(rec, DEFAULT_ICP, [("https://news.example", "t")],
                        _stub_like_llm(), settings)
        return next(c.score for c in fit.criteria if c.key == "growth_signals")

    primary = fit_for("Co raises $150M Series C led by Sequoia")
    secondary = fit_for("Co nearly doubles valuation to $32B in secondary share sale")
    assert secondary < primary
    assert secondary > 2.0, "still a real momentum signal, not discarded"


def test_a_secondary_hook_does_not_congratulate_them_on_a_raise(settings):
    """Saying 'you raised' about money they never received costs the reply."""
    rec = CompanyRecord(
        input="Ramp", company_name="Ramp", domain="ramp.com",
        sources=["https://techcrunch.com/ramp"],
        signals=[Signal(kind="funding", date=_days_ago(9),
                        headline="Ramp nearly doubles valuation to $32B in secondary share sale",
                        source_url="https://techcrunch.com/ramp")],
    )
    from lea.enrich.extract import reclassify_signals
    rec.signals = reclassify_signals(rec.signals)

    hooks = generate_hooks(rec, None, [], _stub_like_llm(), settings)
    joined = " ".join(h.hook.lower() for h in hooks)
    assert "you raised" not in joined
    assert "raised" not in joined
    assert "secondary" in joined
    assert "$32b" in joined


def test_hiring_signals_are_never_reclassified_as_funding():
    """A job board quoting salary bands is not a funding round."""
    from lea.enrich.extract import reclassify_signals

    sigs = reclassify_signals([
        Signal(kind="hiring", headline="47 open roles",
               detail="Comp band $150,000 to $200,000 for Series C stage."),
    ])
    assert sigs[0].kind == "hiring"


# --------------------------------------------------------------------------
# Department mix -> gtm_segment
# --------------------------------------------------------------------------


def _record_with_board(total: int, departments: dict) -> CompanyRecord:
    import json as _json
    from lea.enrich.extract import jobs_signal, parse_jobs_evidence

    payload = _json.dumps({"open_roles": total, "departments": departments,
                           "ats": "ashby", "board_url": "https://jobs.ashbyhq.com/co",
                           "posted_last_30d": 12, "latest_posted": "2026-09-15"})
    jobs = parse_jobs_evidence([("https://jobs.ashbyhq.com/co", payload)])
    return CompanyRecord(
        input="Co", company_name="Co", domain="co.com", industry="SaaS",
        sources=["https://jobs.ashbyhq.com/co"], signals=[jobs_signal(jobs)],
    )


def test_gtm_heavy_hiring_scores_above_the_same_total_in_engineering(settings):
    """63 of 148 roles in Sales is a far louder GTM signal than 148 roles."""
    ev = [("https://jobs.ashbyhq.com/co", "t")]
    sales_heavy = _record_with_board(148, {"Sales": 63, "Engineering": 85})
    eng_only = _record_with_board(148, {"Engineering": 148})

    def gtm(rec):
        fit = score_icp(rec, DEFAULT_ICP, ev, _stub_like_llm(), settings)
        return next(c for c in fit.criteria if c.key == "gtm_segment")

    a, b = gtm(sales_heavy), gtm(eng_only)
    assert a.score > b.score
    assert "63 of 148 open roles in go-to-market" in a.reasoning
    assert "43%" in a.reasoning                       # the share, not just the count
    assert "last 30 days" in a.reasoning              # posted_last_30d is used


# --------------------------------------------------------------------------
# Provenance verification — the anti-hallucination guarantee
#
# The blind spot these cover: the earlier tests asserted that WE never invent a
# value. They never asserted that we reject a value the MODEL invents, which is
# the path a live model actually took (employee_range "201-500" appearing in no
# fixture, reasoning claiming it was "stated", unknowns published empty).
# --------------------------------------------------------------------------

THIN_EVIDENCE = [("https://vanta.com", "Vanta automates compliance. We are hiring.")]

FABRICATED = dict(
    employee_range="201-500", employee_estimate=500, hq_city="San Francisco",
    hq_country="United States", founded_year=2018,
)


def _fabricating_llm(**overrides):
    rec = CompanyRecord(input="Vanta", company_name="Vanta", domain="vanta.com",
                        **{**FABRICATED, **overrides})
    return FakeLLM({CompanyRecord: rec})


@pytest.mark.parametrize("field", ["employee_range", "employee_estimate", "hq_city",
                                   "hq_country", "founded_year"])
def test_a_value_absent_from_the_evidence_is_dropped_not_published(settings, field):
    rec = extract_record("Vanta", THIN_EVIDENCE, _fabricating_llm(), settings)
    assert getattr(rec, field) is None, f"{field} was invented and survived"


def test_unknowns_reflect_post_verification_state_not_what_the_model_filled_in(settings):
    """Regression: unknowns came back empty because the model filled everything."""
    rec = extract_record("Vanta", THIN_EVIDENCE, _fabricating_llm(), settings)

    assert rec.unknowns, "unknowns must never be empty just because the model answered"
    for f in ("employee_range", "hq_city", "hq_country", "founded_year"):
        assert f in rec.unknowns


def test_values_the_evidence_does_support_are_kept(settings):
    """The gate must not be so strict it throws away real findings."""
    ev = [("https://vanta.com",
           "Vanta was founded in 2018, is headquartered in San Francisco, United States, "
           "and has around 350 employees.")]
    rec = extract_record("Vanta", ev, _fabricating_llm(), settings)

    assert rec.employee_range == "201-500"
    assert rec.hq_city == "San Francisco"
    assert rec.hq_country == "United States"
    assert rec.founded_year == 2018
    assert "employee_range" not in rec.unknowns


def test_employee_estimate_is_the_band_midpoint_not_the_models_arithmetic(settings):
    """Regression: the model returned 500 as the midpoint of 201-500."""
    ev = [("https://vanta.com", "Vanta has around 350 employees.")]
    rec = extract_record("Vanta", ev, _fabricating_llm(), settings)

    assert rec.employee_range == "201-500"
    assert rec.employee_estimate == 350, "midpoint of 201-500 is 350, not the model's 500"


def test_country_aliases_are_accepted(settings):
    ev = [("https://vanta.com", "Vanta is based in San Francisco, USA.")]
    rec = extract_record("Vanta", ev, _fabricating_llm(), settings)
    assert rec.hq_country == "United States"      # evidence said "USA"


def test_an_invented_signal_is_dropped(settings):
    """The most damaging possible output: a funding round that never happened."""
    llm = _fabricating_llm(signals=[
        Signal(kind="funding", headline="Vanta raises $500M Series Z from SoftBank",
               source_url="https://vanta.com"),
        Signal(kind="hiring", headline="Vanta automates compliance and is hiring",
               source_url="https://vanta.com"),
    ])
    rec = extract_record("Vanta", THIN_EVIDENCE, llm, settings)

    headlines = [s.headline for s in rec.signals]
    assert not any("Series Z" in h for h in headlines), "invented funding round survived"
    assert any("hiring" in h for h in headlines), "a supported signal was wrongly dropped"


def test_no_evidence_at_all_means_nothing_is_claimed(settings):
    rec = extract_record("Vanta", [], _fabricating_llm(), settings)
    for f in ("employee_range", "hq_city", "hq_country", "founded_year"):
        assert getattr(rec, f) is None
    assert rec.signals == []


def test_the_scorer_cannot_grade_a_criterion_on_a_dropped_value(settings):
    """The model scored company_size 10/10 on a headcount it invented."""
    rec = extract_record("Vanta", THIN_EVIDENCE, _fabricating_llm(), settings)
    assert rec.employee_range is None

    inflated = ICPFit(
        score=96.8, tier="A", summary="",
        criteria=[CriterionScore(
            key=c.key, score=10.0, weight=c.weight, evidence_urls=[],
            reasoning="Headcount is stated as 201-500 employees on the company record.")
            for c in DEFAULT_ICP.criteria],
    )
    fit = score_icp(rec, DEFAULT_ICP, THIN_EVIDENCE, FakeLLM({ICPFit: inflated}), settings)

    assert fit.score < 80.0, "an uncited, fabricated basis must not present as tier A"
    assert fit.tier != "A"
    assert all(c.score <= 7.0 for c in fit.criteria), "uncited criteria are capped"


def test_score_and_confidence_cannot_contradict_each_other(settings):
    """Regression: 96.8 alongside confidence 0.40."""
    rec = extract_record("Vanta", THIN_EVIDENCE, _fabricating_llm(), settings)
    inflated = ICPFit(
        score=96.8, tier="A", summary="",
        # distinct reasoning per criterion, so this is a real (if uncited)
        # model judgement rather than a degenerate default-filled response
        criteria=[CriterionScore(key=c.key, score=10.0, weight=c.weight, evidence_urls=[],
                                 reasoning=f"Excellent fit on {c.key}, a perfect match.")
                  for c in DEFAULT_ICP.criteria],
    )
    fit = score_icp(rec, DEFAULT_ICP, THIN_EVIDENCE, FakeLLM({ICPFit: inflated}), settings)

    ceiling = 40.0 + 60.0 * fit.confidence
    assert fit.score <= ceiling + 0.05, "score must not outrun the evidence behind it"
    assert fit.confidence < 0.5
    assert fit.score < 80.0
    assert "capped" in fit.summary


def test_a_well_evidenced_record_is_not_capped(settings):
    """The ceiling must not punish a genuinely well-sourced assessment."""
    ev = [("https://vanta.com",
           "Vanta was founded in 2018 in San Francisco, United States, has around "
           "350 employees, uses HubSpot, and raised $150M Series C led by Sequoia.")]
    rec = extract_record("Vanta", ev, _fabricating_llm(
        industry="Compliance",
        signals=[Signal(kind="funding", headline="Vanta raised $150M Series C led by Sequoia",
                        date="2026-09-01", source_url="https://vanta.com")],
        tech_stack=[TechStack(category="crm", name="HubSpot")],
    ), settings)

    cited = ICPFit(
        score=0, tier="C", summary="",
        criteria=[CriterionScore(key=c.key, score=9.0, weight=c.weight,
                                 evidence_urls=["https://vanta.com"],
                                 reasoning=f"Strong on {c.key}, per the cited source.")
                  for c in DEFAULT_ICP.criteria],
    )
    fit = score_icp(rec, DEFAULT_ICP, ev, FakeLLM({ICPFit: cited}), settings)

    assert fit.score == 90.0          # 9.0/10 across the board, uncapped
    assert fit.tier == "A"
    assert "capped" not in fit.summary


@pytest.mark.parametrize(
    "evidence_text,claimed,ok",
    [
        ("Vanta currently has 500 employees",  "201-500",   True),   # exact, in band
        ("a team of 350 people",               "201-500",   True),
        ("We are 5,000+ employees",            "5000+",     True),   # open-ended, correct band
        ("We are 5,000+ employees",            "1001-5000", False),  # ">=5000" is not "<=5000"
        ("We are 500+ employees",              "201-500",   False),  # ">=500" cannot mean "<=500"
        ("We are 500+ employees",              "501-1000",  True),
        ("Headcount is 51-200",                "51-200",    True),   # band written out
        ("We serve 300 customers",             "201-500",   False),  # customers are not staff
        ("Vanta automates compliance",         "201-500",   False),
    ],
)
def test_headcount_verification_reads_the_evidence_precisely(evidence_text, claimed, ok):
    from lea.enrich.extract import _verify_employee_range

    assert _verify_employee_range(claimed, evidence_text.lower()) is ok


def test_a_stated_headcount_beats_the_band_midpoint(settings):
    """When the evidence gives a real number, use it rather than the midpoint."""
    ev = [("https://vanta.com", "Vanta currently has 500 employees.")]
    rec = extract_record("Vanta", ev, _fabricating_llm(), settings)
    assert rec.employee_range == "201-500"
    assert rec.employee_estimate == 500, "the stated figure, not the 350 midpoint"


# --------------------------------------------------------------------------
# Derived citations + order-preserving penalty
#
# The regression: a model that reasons well but emits no evidence_urls had every
# criterion clamped to an identical 7.0, so the LLM path could not rank at all
# and scored worse than the keyless one on the headline feature.
# --------------------------------------------------------------------------

GEMINI_LIKE = {"company_type": 9.0, "gtm_segment": 10.0, "company_size": 6.0,
               "tech_maturity": 9.0, "growth_signals": 9.0, "geo": 10.0}


def _uncited_llm(scores=None, summary="Vanta is an exemplary fit."):
    """A model with specific, correct reasoning that cites nothing — the live case."""
    scores = scores or GEMINI_LIKE
    return FakeLLM({ICPFit: ICPFit(
        score=96.8, tier="A", summary=summary,
        criteria=[CriterionScore(key=k, score=v, weight=1.0, evidence_urls=[],
                                 reasoning=f"Vanta scores {v} on {k} for documented reasons.")
                  for k, v in scores.items()],
    )})


def _rich_record() -> CompanyRecord:
    rec = _vanta_like()
    rec.employee_range = "201-500"
    return rec


def test_citations_are_derived_when_the_model_emits_none(settings):
    """Provenance is known in code, so it must not depend on model compliance."""
    rec = _rich_record()
    fit = score_icp(rec, DEFAULT_ICP, _ev(rec), _uncited_llm(), settings)

    cited = [c for c in fit.criteria if c.evidence_urls]
    assert len(cited) >= 4, "most criteria should carry a derived citation"
    for c in cited:
        for u in c.evidence_urls:
            assert u in rec.sources, "a derived citation must be a URL we fetched"


def test_uncited_criteria_are_no_longer_clamped_to_one_value(settings):
    """The exact regression: six criteria, all exactly 7.0, ranking destroyed."""
    rec = _rich_record()
    fit = score_icp(rec, DEFAULT_ICP, _ev(rec), _uncited_llm(), settings)

    scores = [c.score for c in fit.criteria]
    assert len(set(scores)) >= 3, f"criteria collapsed to {set(scores)}"
    assert scores.count(7.0) < len(scores)
    # the model's ordering survives
    by_key = {c.key: c.score for c in fit.criteria}
    assert by_key["gtm_segment"] > by_key["company_size"]


def test_the_shrink_penalises_but_preserves_ordering():
    """An uncited 9 must still outrank an uncited 5."""
    from lea.enrich.icp import UNCITED_NEUTRAL, UNCITED_SHRINK

    def shrunk(x):
        return round(UNCITED_NEUTRAL + (x - UNCITED_NEUTRAL) * UNCITED_SHRINK, 1)

    assert shrunk(10.0) < 10.0 and shrunk(9.0) < 9.0        # penalised
    assert shrunk(10.0) > shrunk(9.0) > shrunk(6.0)          # ordered
    assert shrunk(5.0) == 5.0                                # unknown is the fixed point
    assert shrunk(2.0) > 2.0                                 # uncited pessimism also softened


def test_the_llm_path_ranks_as_well_as_the_keyless_one(settings):
    """Sanity bar: the model path must not be worse than the stub at ranking."""
    rec = _rich_record()
    ev = _ev(rec)
    llm_fit = score_icp(rec, DEFAULT_ICP, ev, _uncited_llm(), settings)
    stub_fit = score_icp(rec, DEFAULT_ICP, ev, _stub_like_llm(), settings)

    assert len({c.score for c in llm_fit.criteria}) >= 3
    assert llm_fit.tier == stub_fit.tier == "A"
    assert abs(llm_fit.score - stub_fit.score) < 15
    assert llm_fit.confidence >= 0.8, "derived citations must lift confidence"


def test_thin_evidence_still_scores_visibly_below_rich_evidence(settings):
    """The penalty must remain real, not be engineered away."""
    rich = _rich_record()
    thin = CompanyRecord(input="Mystery", company_name="Mystery", domain="m.com",
                         industry="SaaS", sources=["https://m.com"])

    rich_fit = score_icp(rich, DEFAULT_ICP, _ev(rich), _uncited_llm(), settings)
    thin_fit = score_icp(thin, DEFAULT_ICP, [("https://m.com", "Mystery builds SaaS.")],
                         _uncited_llm(), settings)

    assert thin_fit.score < rich_fit.score - 15
    assert thin_fit.confidence < rich_fit.confidence
    assert thin_fit.tier != "A"
    # and it still ranks its own criteria rather than flattening them
    assert len({c.score for c in thin_fit.criteria}) >= 3


def test_a_bounded_score_leads_with_the_bound_not_the_enthusiasm(settings):
    """'Exemplary fit' in front of 61/B is the 96.8-at-0.40 contradiction inverted."""
    thin = CompanyRecord(input="Mystery", company_name="Mystery", domain="m.com",
                         industry="SaaS", sources=["https://m.com"])
    fit = score_icp(thin, DEFAULT_ICP, [("https://m.com", "Mystery builds SaaS.")],
                    _uncited_llm(summary="Mystery is an exemplary fit."), settings)

    assert fit.summary.startswith("Bounded by evidence:")
    assert f"tier {fit.tier}" in fit.summary
    assert fit.summary.index("Bounded by evidence") < fit.summary.index("exemplary")


# --------------------------------------------------------------------------
# Relevance gate — from the real out/linear.json failure
#
# A live run searched "Linear issue tracker software company"; the word "issue"
# hijacked the results and these five pages were published as sources. A
# reviewer clicking a citation and landing on J.K. Rowling's website is a
# credibility event, so the gate is biased toward dropping.
# --------------------------------------------------------------------------

LINEAR_JUNK = [
    ("https://www.jkrowling.com/opinions/j-k-rowling-writes-about-her-reasons-for-"
     "speaking-out-on-sex-and-gender-issues/",
     "J.K. Rowling writes about her reasons for speaking out on sex and gender issues."),
    ("https://developer.atlassian.com/cloud/jira/platform/rest/v2/api-group-issues/",
     "Jira Cloud platform REST API. This resource represents Jira issues."),
    ("https://github.com/whatwg/html/issues/11523",
     "whatwg/html issue 11523: linear-gradient interpolation in canvas rendering. "
     "The linear interpolation path is wrong."),
    ("https://www.facebook.com/bbclearningenglish.multimedia/videos/4-ways-to-use-issue/"
     "343528391359677/",
     "4 ways to use issue - BBC Learning English video."),
    ("https://support.google.com/pixelphone/thread/413513162/network-connection-issue-"
     "with-pixel-10-pro-xl-512gb",
     "Network connection issue with Pixel 10 Pro XL 512GB. Thread replies."),
]

LINEAR_GENUINE = [
    ("https://linear.app/", "Linear is a purpose-built tool for planning and building "
                            "products. Loops for product management. Priority inbox."),
    ("https://lps.linear.app/", "Landing pages."),
    # Thin but non-empty: the real run cited it, so something was read. A truly
    # empty body is a failed fetch and is excluded by `citable_urls`, separately.
    ("https://orbit.linear.app/", "Orbit"),
    ("https://linear.app/brand", "Brand assets"),
    ("https://linear.app/integrations/dixa", "Dixa integration"),
    ("https://linear.app/quality/01", "Quality"),
]


@pytest.mark.parametrize("url,body", LINEAR_JUNK)
def test_search_junk_is_never_cited(url, body):
    from lea.enrich.extract import is_relevant_source

    assert not is_relevant_source(url, body, "Linear", "Linear", "linear.app")


@pytest.mark.parametrize("url,body", LINEAR_GENUINE)
def test_the_companys_own_pages_are_always_kept(url, body):
    """Thin pages on the company's own domain are genuine and never need to prove it."""
    from lea.enrich.extract import is_relevant_source

    assert is_relevant_source(url, body, "Linear", "Linear", "linear.app")


def test_a_legitimate_third_party_article_is_kept():
    """A funding article may not carry the name in its URL, so content is what counts."""
    from lea.enrich.extract import is_relevant_source

    assert is_relevant_source(
        "https://www.forbes.com/sites/x/2025/01/01/software-roundup/",
        "The roundup mentions Linear among the fastest-growing software companies.",
        "Linear", "Linear", "linear.app")
    assert is_relevant_source(
        "https://techcrunch.com/2024/07/23/vanta-series-c-funding/",
        "Vanta has raised a $150M Series C led by Sequoia, the company said today.",
        "Vanta", "Vanta", "vanta.com")


def test_a_common_word_name_is_not_matched_in_lowercase_prose():
    """'linear-gradient' and 'linear algebra' must not qualify as mentions."""
    from lea.enrich.extract import is_relevant_source

    assert not is_relevant_source(
        "https://example.com/math", "A linear map between linear spaces is linear.",
        "Linear", "Linear", "linear.app")


def test_the_gate_strips_junk_from_a_published_record(settings):
    """End-to-end on the real failure: 11 sources in, 6 genuine ones out."""
    evidence = LINEAR_JUNK + LINEAR_GENUINE
    base = CompanyRecord(
        input="Linear", company_name="Linear", domain="linear.app",
        signals=[
            Signal(kind="product", headline="Loops for product management",
                   source_url="https://linear.app/"),
            Signal(kind="news", headline="Gender issues opinion piece",
                   source_url="https://www.jkrowling.com/opinions/j-k-rowling-writes-"
                              "about-her-reasons-for-speaking-out-on-sex-and-gender-issues/"),
        ],
    )
    rec = extract_record("Linear", evidence, FakeLLM({CompanyRecord: base}), settings)

    assert len(rec.sources) == 6
    assert all("linear.app" in u for u in rec.sources)
    for junk_url, _ in LINEAR_JUNK:
        assert junk_url not in rec.sources

    # a signal whose only source was junk keeps its claim but loses the link,
    # rather than carrying a citation that goes somewhere unrelated
    by_head = {s.headline: s for s in rec.signals}
    assert by_head["Loops for product management"].source_url == "https://linear.app/"
    if "Gender issues opinion piece" in by_head:
        assert by_head["Gender issues opinion piece"].source_url is None


def test_criteria_cannot_cite_an_irrelevant_page(settings):
    rec = CompanyRecord(input="Linear", company_name="Linear", domain="linear.app",
                        industry="SaaS", sources=["https://linear.app/"])
    junk_url = LINEAR_JUNK[0][0]
    llm = FakeLLM({ICPFit: ICPFit(
        score=90, tier="A", summary="Great fit.",
        criteria=[CriterionScore(key=c.key, score=9.0, weight=1.0,
                                 evidence_urls=[junk_url],
                                 reasoning=f"Judged {c.key} from the sources.")
                  for c in DEFAULT_ICP.criteria])})

    fit = score_icp(rec, DEFAULT_ICP, LINEAR_JUNK + [("https://linear.app/", "Linear is a tool.")],
                    llm, settings)
    for c in fit.criteria:
        assert junk_url not in c.evidence_urls


def test_a_mostly_irrelevant_run_is_reported_as_poor_research(settings):
    rec = CompanyRecord(input="Linear", company_name="Linear", domain="linear.app",
                        industry="SaaS", sources=["https://linear.app/"])
    evidence = LINEAR_JUNK + [("https://linear.app/", "Linear is a tool for teams.")]

    fit = score_icp(rec, DEFAULT_ICP, evidence, _stub_like_llm(), settings)

    assert "Research quality was poor" in fit.summary
    assert "5 of 6" in fit.summary
    # and a clean run says nothing of the sort
    clean = score_icp(rec, DEFAULT_ICP, [("https://linear.app/", "Linear is a tool.")],
                      _stub_like_llm(), settings)
    assert "Research quality was poor" not in clean.summary
    assert fit.confidence < clean.confidence


# --------------------------------------------------------------------------
# Common-word brand names vs sentence-initial capitalisation
#
# Case-sensitivity alone is not enough: "Linear algebra is the branch of
# mathematics..." capitalises Linear for grammatical reasons, not because it is
# a proper noun. Wikipedia, docs and academic pages start sentences with the
# word constantly, and those are exactly what ranks for a bare common-word
# query. Position and the right-hand neighbour decide, not case alone.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,body,name,domain",
    [
        # the reported failure
        ("https://en.wikipedia.org/wiki/Linear_algebra",
         "Linear algebra is the branch of mathematics concerning linear equations.",
         "Linear", "linear.app"),
        # siblings, all realistic for names in examples/companies.csv
        ("https://stats.example/lr",
         "Linear regression is a statistical method for modelling relationships.",
         "Linear", "linear.app"),
        ("https://math.example/sets",
         "Notion of a set is fundamental in mathematics and logic.",
         "Notion", "notion.so"),
        ("https://traffic.gov/ramp",
         "Ramp metering is a traffic management technique used on motorways.",
         "Ramp", "ramp.com"),
        ("https://en.wikipedia.org/wiki/Figma_(disambiguation)",
         "Figma may also refer to a place name in northern Italy.",
         "Figma", "figma.com"),
        # title-cased phrase, mid-sentence: the adjective reading
        ("https://uni.example/course",
         "Introduction to Linear Algebra, a first course in mathematics.",
         "Linear", "linear.app"),
    ],
)
def test_a_common_word_at_a_sentence_boundary_is_not_a_brand_mention(url, body, name, domain):
    from lea.enrich.extract import is_relevant_source

    assert not is_relevant_source(url, body, name, name, domain)


@pytest.mark.parametrize(
    "body",
    [
        "Linear has raised a Series C led by Accel.",             # subject + verb
        "Linear, the issue tracking startup, announced a tier.",  # appositive
        "Linear's roadmap for 2026 includes agents.",             # possessive
        "The roundup mentions Linear among fast-growing tools.",  # mid-sentence capital
        "Linear customers include Vercel and Ramp.",              # weak, but corroborated
        "Teams use Linear for planning and building products.",   # mid-sentence capital
    ],
)
def test_a_genuine_brand_mention_is_still_recognised(body):
    from lea.enrich.extract import is_relevant_source

    assert is_relevant_source("https://news.example/a", body, "Linear", "Linear", "linear.app")


def test_a_name_in_the_url_alone_does_not_make_a_page_relevant():
    """wikipedia.org/wiki/Linear_algebra carries the slug and is still junk."""
    from lea.enrich.extract import is_relevant_source

    assert not is_relevant_source(
        "https://en.wikipedia.org/wiki/Linear_algebra",
        "The branch of mathematics concerning systems of equations and matrices.",
        "Linear", "Linear", "linear.app")
    # but it is enough when the body is too short to judge at all
    assert is_relevant_source(
        "https://techcrunch.com/2024/07/23/vanta-series-c-funding/", "",
        "Vanta", "Vanta", "vanta.com")


def test_mention_strength_classification():
    from lea.enrich.extract import _mentions

    assert _mentions("Linear is a tool.", "Linear") == ["strong"]
    assert _mentions("Linear algebra is hard.", "Linear") == ["weak"]
    assert _mentions("We use Linear daily.", "Linear") == ["strong"]
    assert _mentions("Introduction to Linear Algebra", "Linear") == ["weak"]
    assert _mentions("linear-gradient is a CSS function.", "Linear") == []


@pytest.mark.parametrize(
    "url,name,domain,expected",
    [
        # A public ATS board is hosted on the ATS's domain, not the company's,
        # but fetch_jobs resolved it FOR this company. It is as genuine a source
        # as the company's own /careers page.
        ("https://jobs.ashbyhq.com/vanta", "Vanta", "vanta.com", True),
        ("https://boards.greenhouse.io/figma", "Figma", "figma.com", True),
        ("https://jobs.lever.co/ramp", "Ramp", "ramp.com", True),
        ("https://apply.workable.com/linear/", "Linear", "linear.app", True),
        # ...but only for THIS company
        ("https://jobs.ashbyhq.com/someone-else", "Vanta", "vanta.com", False),
        ("https://boards.greenhouse.io/competitor", "Figma", "figma.com", False),
    ],
)
def test_ats_boards_are_relevant_only_for_their_own_company(url, name, domain, expected):
    from lea.enrich.extract import is_relevant_source

    assert is_relevant_source(url, "12 open roles", name, name, domain) is expected


def test_agent_a_real_job_board_summary_survives_the_gate():
    """The exact string fetch_jobs emits: name is sentence-initial before an em dash."""
    from lea.enrich.extract import is_relevant_source

    summary = ("Vanta — 93 open roles on Ashby across Revenue (36), Software Engineering "
               "(27), Security (6). Board: https://jobs.ashbyhq.com/vanta")
    assert is_relevant_source("https://jobs.ashbyhq.com/vanta", summary,
                              "Vanta", "Vanta", "vanta.com")
