"""Synthesis-layer tests. No network, no real LLM, no clock dependence.

The FakeLLM returns canned instances so every assertion below is about OUR
post-processing — normalization, dedupe, the score arithmetic, the hook filter —
rather than about model behavior.
"""

from __future__ import annotations

import re
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


EVIDENCE = [
    ("https://ramp.com", "Ramp is a corporate card company."),
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
    assert "employee_range" not in rec.unknowns
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
    assert rec.sources == ["https://ramp.com", "https://techcrunch.com/ramp"]


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
    assert rec.sources == ["https://ramp.com", "https://techcrunch.com/ramp"]


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

    assert fit.score == 80.0          # the arithmetic still runs
    assert fit.tier == "A"
    assert fit.confidence < 0.4       # but the caveat is loud
    assert "Low confidence" in fit.summary


def test_evidence_urls_are_filtered_to_urls_actually_gathered(settings):
    llm = FakeLLM({ICPFit: _fit_from(
        {"company_type": 8, "company_size": 8, "tech_maturity": 8, "growth_signals": 8},
        urls=["https://hallucinated.example"],
    )})
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)
    for c in fit.criteria:
        assert c.evidence_urls == []


def test_score_icp_falls_back_to_the_rubric_when_the_model_fails(settings):
    """No flat 50/C. The record itself is graded, and the arithmetic is exact.

    _known_record() -> company_type 9.0 (fintech + Salesforce CRM), company_size
    10.0 (201-500), tech_maturity 4.7 (1 tech / 1 category + CRM), growth_signals
    7.0 (undated funding signal). Weights 2,1,1,1:
        (9.0*2 + 10.0 + 4.7 + 7.0) / 5 * 10 = 39.7 / 5 * 10 = 79.4
    """
    llm = FakeLLM(raise_on=ICPFit)
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)

    assert fit.score == 79.4
    assert fit.tier == "B"
    assert (fit.score, fit.tier) != (50.0, "C")
    assert all(len(c.reasoning) > 40 for c in fit.criteria)   # real analysis, not a stub line
    assert "deterministic rubric" in fit.summary


def test_model_scores_are_kept_and_only_gaps_are_rubric_graded(settings):
    """A partial model response is not discarded — the judgement it did make stands.

    company_type 10.0 comes from the model; the other three are rubric-graded as
    above: (10.0*2 + 10.0 + 4.7 + 7.0) / 5 * 10 = 41.7 / 5 * 10 = 83.4
    """
    fit_in = _fit_from({"company_type": 10}, urls=["https://a.example"])
    llm = FakeLLM({ICPFit: fit_in})
    fit = score_icp(_known_record(), _icp_4(), [("https://a.example", "t")], llm, settings)

    assert len(fit.criteria) == 4
    assert fit.score == 83.4
    by_key = {c.key: c for c in fit.criteria}
    assert by_key["company_type"].score == 10.0                      # the model's
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
    rec = _ramp_like()
    assert rec.employee_range is None
    fit = score_icp(rec, DEFAULT_ICP, _ev(rec), _stub_like_llm(), settings)
    size = next(c for c in fit.criteria if c.key == "company_size")
    assert size.score == 5.0
    assert "could not be established" in size.reasoning


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
