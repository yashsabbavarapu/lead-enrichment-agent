"""The ICP engine: infer a rubric, score against it, persist it.

Two rules govern this module.

1. The LLM judges, it does not count. It produces a 0-10 per criterion with
   written reasoning and cited URLs. The weighted roll-up to 0-100, the tier
   thresholds and the disqualifier override are all computed here, in Python.
   A model that quietly adds wrong is a much worse bug than one that reasons
   badly, because nothing about the output looks off.
2. A score built on unknowns has to say so. `confidence` falls when criteria
   were graded without evidence.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from lea.config import Settings
from lea.enrich.default_icp import DEFAULT_ICP
from lea.llm.base import LLMClient
from lea.models import CompanyRecord, CriterionScore, ICPCriterion, ICPFit, ICPProfile

__all__ = [
    "infer_icp",
    "score_icp",
    "load_icp",
    "save_icp",
    "load_or_default_icp",
    "compute_fit_score",
    "tier_for",
    "is_unresearched",
    "deterministic_scores",
    "DEFAULT_ICP",
    "TIER_THRESHOLDS",
    "DISQUALIFIED_SCORE_CAP",
]

TIER_THRESHOLDS: list[tuple[str, float]] = [("A", 80.0), ("B", 60.0), ("C", 40.0)]
DISQUALIFIED_SCORE_CAP = 25.0
MIN_CRITERIA = 4

DEFAULT_ICP_FILENAME = "icp.yaml"

_YAML_HEADER = """\
# ICP profile for lead-enrichment-agent.
#
# Edit this file by hand — it is the whole point of keeping it as YAML.
#   criteria[].key      snake_case, unique; shown as the row label in the score table
#   criteria[].weight   relative importance; only the ratio matters, they are
#                       normalized at scoring time (so 2.0 vs 1.0 == 4.0 vs 2.0)
#   criteria[].rubric   plain-language anchors for 0 and 10. The more concrete
#                       these are, the more stable the scores across companies.
#   disqualifiers       any hit forces tier D and caps the score, regardless of
#                       how well the company scores elsewhere.
#
# Regenerate from real customers with:
#   lea icp infer --examples "Customer A, Customer B, Customer C" --out icp.yaml
"""


# --------------------------------------------------------------------------
# Prompts (lazy, with fallbacks so this module works before Agent B lands)
# --------------------------------------------------------------------------

_FALLBACK_INFERENCE_SYSTEM = (
    "You are a GTM strategist. Given a list of a vendor's example customers, infer the "
    "Ideal Customer Profile they share. Output a weighted rubric: 4-7 criteria, each with "
    "a snake_case `key`, a one-sentence `description`, a positive `weight` reflecting "
    "relative importance, and a `rubric` giving concrete anchors for what a 0 and a 10 "
    "look like. Also list 2-3 `disqualifiers`: hard stops that rule an account out no "
    "matter how well it scores otherwise. Criteria must be observable from public web "
    "evidence — no internal CRM fields. Write a `summary` naming what these customers "
    "actually have in common."
)

_FALLBACK_SCORING_SYSTEM = (
    "You are scoring one company against an ICP rubric. For EACH criterion return a "
    "score from 0 to 10, one or two sentences of `reasoning` citing the specific fact you "
    "used, and `evidence_urls` listing the source URLs that fact came from. If the "
    "evidence does not establish a criterion, score it 5, say 'no evidence' in the "
    "reasoning, and leave evidence_urls empty — do not guess. List any disqualifier you "
    "can positively confirm in `disqualifiers_hit`, quoting it verbatim. "
    "Do NOT compute an overall score; that is calculated separately."
)


def _inference_prompt() -> str:
    try:  # pragma: no cover - import shim
        from lea.prompts import ICP_INFERENCE_SYSTEM

        return ICP_INFERENCE_SYSTEM
    except Exception:
        return _FALLBACK_INFERENCE_SYSTEM


def _scoring_prompt() -> str:
    try:  # pragma: no cover - import shim
        from lea.prompts import ICP_SCORING_SYSTEM

        return ICP_SCORING_SYSTEM
    except Exception:
        return _FALLBACK_SCORING_SYSTEM


# --------------------------------------------------------------------------
# Repair helpers
# --------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def snake_case(raw: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(raw or "")).lower()
    s = _NON_ALNUM.sub("_", s).strip("_")
    return s or "criterion"


def _repair_criteria(criteria: list[ICPCriterion]) -> list[ICPCriterion]:
    """Positive weights, snake_case unique keys, no empty descriptions."""
    out: list[ICPCriterion] = []
    seen: set[str] = set()
    for c in criteria or []:
        if c is None:
            continue
        key = snake_case(getattr(c, "key", "") or getattr(c, "description", "")[:40])
        if key in seen:
            n = 2
            while f"{key}_{n}" in seen:
                n += 1
            key = f"{key}_{n}"
        seen.add(key)

        try:
            weight = float(c.weight)
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0 or weight != weight or weight == float("inf"):
            weight = 1.0

        out.append(
            ICPCriterion(
                key=key,
                description=(c.description or key.replace("_", " ")).strip(),
                weight=round(weight, 3),
                rubric=(c.rubric or "").strip(),
            )
        )
    return out


def _pad_criteria(criteria: list[ICPCriterion]) -> list[ICPCriterion]:
    """A rubric with two criteria is not a rubric. Top up from the default."""
    have = {c.key for c in criteria}
    for c in DEFAULT_ICP.criteria:
        if len(criteria) >= MIN_CRITERIA:
            break
        if c.key not in have:
            criteria.append(c.model_copy(deep=True))
            have.add(c.key)
    return criteria


def _clean_disqualifiers(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for d in items or []:
        s = str(d or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out[:6]


# --------------------------------------------------------------------------
# infer
# --------------------------------------------------------------------------


def infer_icp(
    examples: list[str],
    llm: LLMClient,
    settings: Settings,
    notes: Optional[str] = None,
) -> ICPProfile:
    """Derive a weighted ICP rubric from 3-5 example customers.

    The headline feature: instead of hand-writing a scoring rubric, name accounts
    you already closed and let the model articulate what they have in common.
    Whatever comes back is mechanically repaired before it is trusted.
    """
    names = [str(e).strip() for e in (examples or []) if str(e).strip()]
    seen: set[str] = set()
    names = [n for n in names if not (n.lower() in seen or seen.add(n.lower()))]

    if not names:
        prof = DEFAULT_ICP.model_copy(deep=True)
        prof.summary = "No example customers supplied. " + prof.summary
        return prof

    context_lines = [
        "EXAMPLE CUSTOMERS (accounts this vendor has already won):",
        *[f"  {i}. {n}" for i, n in enumerate(names, 1)],
        "",
        "Infer the Ideal Customer Profile these accounts share.",
    ]
    if notes:
        context_lines += ["", "ADDITIONAL CONTEXT FROM THE USER:", str(notes).strip()]
    context = "\n".join(context_lines)

    profile: Optional[ICPProfile] = None
    try:
        profile, _usage = llm.extract(
            _inference_prompt(),
            context,
            ICPProfile,
            hint="Derive a weighted ICP rubric with 4-7 criteria and 2-3 disqualifiers.",
        )
    except Exception:
        profile = None

    if not isinstance(profile, ICPProfile):
        profile = DEFAULT_ICP.model_copy(deep=True)
        profile.summary = (
            "Could not infer a profile from the supplied examples; falling back to the "
            "default placeholder. " + profile.summary
        )

    criteria = _pad_criteria(_repair_criteria(list(profile.criteria)))
    disqualifiers = _clean_disqualifiers(list(profile.disqualifiers))
    if not disqualifiers:
        disqualifiers = list(DEFAULT_ICP.disqualifiers[:2])

    name = (profile.name or "").strip()
    if not name or name == DEFAULT_ICP.name:
        name = "inferred-" + snake_case("-".join(names[:2]))[:40]

    summary = (profile.summary or "").strip()
    if not summary:
        summary = "Inferred from: " + ", ".join(names) + "."

    return ICPProfile(
        name=name,
        summary=summary,
        criteria=criteria,
        disqualifiers=disqualifiers,
        derived_from=names,
    )


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def tier_for(score: float) -> str:
    for tier, threshold in TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "D"


def compute_fit_score(criteria: list[CriterionScore]) -> float:
    """Weighted mean of 0-10 criterion scores, rescaled to 0-100.

    This is the one number a buyer acts on, so it is computed here and never by
    the model. Weights are normalized, so they need not sum to anything.
    """
    usable = [c for c in criteria if c is not None and (c.weight or 0) > 0]
    total_weight = sum(float(c.weight) for c in usable)
    if total_weight <= 0:
        return 0.0
    weighted = sum(float(c.weight) * max(0.0, min(10.0, float(c.score))) for c in usable)
    return round((weighted / total_weight) * 10.0, 1)


_DQ_STOP = {
    "with", "that", "this", "they", "their", "from", "have", "been", "when",
    "which", "there", "where", "such", "into", "then", "than", "also", "does",
    "will", "would", "about", "other", "these", "those", "being",
}


def _dq_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{4,}", text.lower()) if t not in _DQ_STOP}


def _match_disqualifier(claimed: str, declared: list[str]) -> Optional[str]:
    """Map a model-claimed disqualifier back onto one the profile actually declares.

    Anti-hallucination: the model cannot invent a new hard stop at scoring time.
    """
    c = str(claimed or "").strip().lower()
    if not c:
        return None
    for d in declared:
        dl = d.strip().lower()
        if c == dl or c in dl or dl in c:
            return d
    # Token-overlap fallback — the model paraphrases constantly, and a declared
    # disqualifier is usually a long sentence while the claim is a short phrase.
    # Normalizing by the SMALLER set makes "it is a consumer-only B2C company"
    # match "Consumer-only (B2C) company with no business-facing product."
    c_tokens = _dq_tokens(c)
    best, best_overlap = None, 0.0
    for d in declared:
        d_tokens = _dq_tokens(d.lower())
        if not d_tokens or not c_tokens:
            continue
        shared = c_tokens & d_tokens
        if len(shared) < 2:  # one shared word is a coincidence, not a match
            continue
        overlap = len(shared) / min(len(c_tokens), len(d_tokens))
        if overlap > best_overlap:
            best, best_overlap = d, overlap
    return best if best_overlap >= 0.5 else None


_PRIORITY_FIELDS_FOR_CONFIDENCE = [
    "company_name",
    "domain",
    "industry",
    "employee_range",
    "hq_country",
    "founded_year",
]


def _record_known_fraction(record: CompanyRecord) -> float:
    fields = _PRIORITY_FIELDS_FOR_CONFIDENCE
    known = 0
    for f in fields:
        v = getattr(record, f, None)
        if v is not None and (not isinstance(v, str) or v.strip()):
            known += 1
    extras = 2
    if record.signals:
        known += 1
    if record.tech_stack:
        known += 1
    return known / float(len(fields) + extras)


def _compute_confidence(scored: list[CriterionScore], record: CompanyRecord) -> float:
    """How much of this score rests on actual cited evidence.

    60% from the share of criteria that cited a source URL, 40% from how much of
    the record itself was established. Three unknowns and the number says so.
    """
    if not scored:
        return 0.0
    with_evidence = sum(1 for c in scored if c.evidence_urls)
    evidence_frac = with_evidence / float(len(scored))
    known_frac = _record_known_fraction(record)
    return round(0.6 * evidence_frac + 0.4 * known_frac, 2)


def is_unresearched(record: CompanyRecord, evidence: list[tuple[str, str]]) -> bool:
    """Did the research establish anything at all about this company?

    Nothing to go on means no domain, no industry, no signals, no tech, no
    headcount and no country — or no citable evidence whatsoever. A company name
    alone does not count: the agent was handed that.
    """
    record_empty = not any(
        [
            record.domain,
            (record.industry or "").strip() if isinstance(record.industry, str) else record.industry,
            record.signals,
            record.tech_stack,
            record.employee_range,
            record.hq_country,
        ]
    )
    no_evidence = not (evidence or record.sources)
    return bool(record_empty or no_evidence)


def _freshest_signal_phrase(record: CompanyRecord, today: Optional[datetime] = None) -> str:
    """The single most actionable recent signal, for the summary line.

    The tier is a ranking; this is the reason to pick up the phone this week, so
    it belongs on the same line rather than buried in the signal table.
    """
    from lea.enrich.extract import describe_age, signal_age_days

    best: Optional[tuple[int, str]] = None
    for sig in record.signals or []:
        age = signal_age_days(sig, today)
        if age is None or age > 120:
            continue
        if best is None or age < best[0]:
            best = (age, (sig.headline or "").strip())
    if best is None or not best[1]:
        return ""
    return f'Freshest signal ({describe_age(best[0])}): "{best[1][:80]}".'


def _non_assessment(record: CompanyRecord, icp: ICPProfile) -> ICPFit:
    """The output when research turned up nothing.

    Deliberately NOT the neutral 50/C. A middling score reads as a considered
    judgement about the company, and someone will work the list believing it was
    one. Zero at tier D with an explicit summary cannot be misread, and it sorts
    to the bottom, which is where an unresearched company belongs.
    """
    name = record.company_name or record.input or "this company"
    return ICPFit(
        score=0.0,
        tier="D",
        summary=(
            f"NO ASSESSMENT — research did not establish enough about {name} to score it: "
            f"no domain, industry, headcount, signals or tech stack were confirmed. This is "
            f"the absence of a score, not a low one, and nothing here is a judgement about "
            f"the company. Re-run with a search API key, or check the name spelling."
        ),
        criteria=[
            CriterionScore(
                key=c.key,
                score=0.0,
                weight=c.weight,
                reasoning="Not assessed — no evidence was gathered for this criterion.",
                evidence_urls=[],
            )
            for c in icp.criteria
        ],
        disqualifiers_hit=[],
        confidence=0.0,
    )


# --------------------------------------------------------------------------
# Deterministic rubric scorer
#
# The keyless stub cannot produce a per-criterion ICPFit — it returns an empty
# criteria list — so without this every company scored an identical 50/C and the
# headline feature demoed as a constant. This grades each criterion from what is
# actually in the record, names the evidence it used, and cites a URL where one
# exists. It is not as good as a real model. It is legible, and it discriminates.
# --------------------------------------------------------------------------

_B2B_WORDS = {
    "b2b", "saas", "software", "fintech", "infrastructure", "devtools", "developer",
    "security", "compliance", "cybersecurity", "analytics", "data", "cloud", "api",
    "platform", "enterprise", "martech", "adtech", "hr tech", "hrtech", "payroll",
    "procurement", "logistics", "insurtech", "legaltech", "observability",
    "identity", "payments", "spend management", "crm", "erp", "workflow",
}
_CONSUMER_WORDS = {
    "consumer", "b2c", "retail", "apparel", "clothing", "grocery", "restaurant",
    "gaming", "video game", "dating", "fitness app", "streaming", "social network",
    "language learning", "edtech consumer", "direct-to-consumer", "dtc", "ecommerce brand",
}
_CRM_NAMES = {"salesforce", "hubspot", "pipedrive", "zoho", "dynamics", "attio", "close"}
_SALES_TOOL_NAMES = {
    "qualified", "outreach", "salesloft", "drift", "marketo", "pardot", "intercom",
    "clearbit", "6sense", "gong", "apollo", "zoominfo", "chili piper", "mutiny",
    "marketo engage", "demandbase", "default", "navattic",
}
_ATS_NAMES = {"ashby", "greenhouse", "lever", "workable", "rippling", "bamboohr", "workday"}
_DATA_NAMES = {
    "snowflake", "databricks", "bigquery", "redshift", "dbt", "fivetran", "segment",
    "looker", "tableau", "amplitude", "mixpanel", "ga4", "google analytics", "heap",
    "posthog", "hex", "mode",
}
_INFRA_NAMES = {
    "cloudflare", "vercel", "netlify", "aws", "gcp", "azure", "kubernetes",
    "datadog", "sentry", "fastly", "akamai", "heroku",
}

_STRONG_GEO = {
    "us", "usa", "united states", "canada", "uk", "united kingdom", "ireland",
    "germany", "france", "netherlands", "spain", "italy", "belgium", "switzerland",
    "austria", "portugal", "luxembourg",
}
_MEDIUM_GEO = {
    "australia", "new zealand", "sweden", "norway", "denmark", "finland",
    "iceland", "israel", "singapore",
}

_SIZE_SCORES = {
    "51-200": 10.0, "201-500": 10.0,
    "11-50": 6.0, "501-1000": 6.0,
    "1001-5000": 3.0,
    "1-10": 1.0, "5000+": 1.0,
}


class _Facts:
    """Everything the deterministic scorers need, computed once per record."""

    def __init__(self, record: CompanyRecord, today: Optional[datetime] = None):
        from lea.enrich.extract import jobs_from_record

        self.record = record
        self.today = today
        self.jobs = jobs_from_record(record)
        self.text = " ".join(
            str(x or "").lower()
            for x in (record.industry, record.sub_industry, record.description)
        )
        self.tech_names = {(t.name or "").strip().lower() for t in record.tech_stack or []}
        self.tech_categories = {
            (t.category or "").strip().lower() for t in record.tech_stack or [] if t.category
        }
        self.signals_by_kind: dict[str, list] = {}
        for s in record.signals or []:
            self.signals_by_kind.setdefault(s.kind, []).append(s)
        self.default_url = (record.sources or [None])[0]

    def has(self, names: set[str]) -> list[str]:
        """Which of `names` are in the tech stack (substring match both ways)."""
        found = []
        for n in self.tech_names:
            if not n:
                continue
            if n in names or any(k in n or n in k for k in names):
                found.append(n)
        return sorted(set(found))

    def kind(self, k: str) -> list:
        return self.signals_by_kind.get(k, [])

    def urls_for(self, *kinds: str) -> list[str]:
        out = [s.source_url for k in kinds for s in self.kind(k) if s.source_url]
        seen, uniq = set(), []
        for u in out:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq or ([self.default_url] if self.default_url else [])

    def any_word(self, words: set[str]) -> Optional[str]:
        for w in words:
            if w in self.text:
                return w
        return None


def _clamp(x: float) -> float:
    return round(max(0.0, min(10.0, x)), 1)


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one}" if n == 1 else f"{n} {many or one + 's'}"


def _join_names(names: list[str]) -> str:
    """Title-case detected tech for readability: ['hubspot'] -> 'HubSpot'-ish."""
    pretty = [n if any(ch.isupper() for ch in n) else n.title() for n in names]
    if len(pretty) <= 2:
        return " and ".join(pretty)
    return ", ".join(pretty[:-1]) + " and " + pretty[-1]


def _sc_company_type(f: _Facts) -> tuple[float, str, list[str]]:
    consumer = f.any_word(_CONSUMER_WORDS)
    if consumer:
        return (
            2.0,
            f"Reads as a consumer business: '{consumer}' appears in the industry/description. "
            f"No evidence of a business-facing product.",
            [f.default_url] if f.default_url else [],
        )
    b2b = f.any_word(_B2B_WORDS)
    crm = f.has(_CRM_NAMES)
    sales = f.has(_SALES_TOOL_NAMES)

    score = 8.0 if b2b else 5.0
    parts = []
    if b2b:
        parts.append(f"industry/description mentions '{b2b}', consistent with a B2B product")
    else:
        parts.append("industry and description do not clearly establish B2B vs consumer")
    if crm:
        score += 1.0
        parts.append(f"a business CRM ({', '.join(crm)}) is in the stack, which consumer apps rarely run")
    if sales:
        score += 0.5
        parts.append(f"B2B sales tooling detected ({', '.join(sales)})")
    return _clamp(score), ". ".join(p[0].upper() + p[1:] for p in parts) + ".", (
        [f.default_url] if f.default_url else []
    )


def _sc_gtm_segment(f: _Facts) -> tuple[float, str, list[str]]:
    crm = f.has(_CRM_NAMES)
    sales = f.has(_SALES_TOOL_NAMES)
    ats = f.has(_ATS_NAMES)
    hiring = f.kind("hiring")

    score = 4.0
    parts = []
    if crm:
        score += 2.5
        parts.append(f"runs {_join_names(crm)} as its CRM, so there is a pipeline someone owns")
    if sales:
        score += min(3.0, 1.5 * len(sales))
        parts.append(
            f"uses dedicated sales/marketing tooling ({_join_names(sales)}), which implies "
            f"an active outbound or inbound motion"
        )
    if ats:
        score += 1.0
        parts.append(f"runs an applicant tracking system ({_join_names(ats)}), indicating structured hiring")
    if hiring:
        score += 1.0
        parts.append(f'is actively hiring — "{hiring[0].headline[:70]}"')
    if not parts:
        return (
            5.0,
            "No CRM, sales tooling or hiring activity was detected, so the GTM motion could "
            "not be established either way. Scored mid pending evidence.",
            [f.default_url] if f.default_url else [],
        )
    return _clamp(score), "The company " + "; ".join(parts) + ".", f.urls_for("hiring")


def _sc_company_size(f: _Facts) -> tuple[float, str, list[str]]:
    """Size, graded on whatever evidence exists — headcount, else open roles.

    Open roles are a stage/momentum proxy, not a headcount: a company running 47
    live reqs is demonstrably not a five-person shop, but we do not know whether
    it is 200 people or 600, and we never write a fabricated `employee_range`
    from it. The reasoning says which evidence it used and what it cannot know.
    """
    rec = f.record
    if rec.employee_range and rec.employee_range in _SIZE_SCORES:
        score = _SIZE_SCORES[rec.employee_range]
        est = f" (~{rec.employee_estimate} employees)" if rec.employee_estimate else ""
        return (
            score,
            f"Headcount band {rec.employee_range}{est}. The rubric's sweet spot is 51-500, "
            f"where GTM process pain is real but procurement is still light.",
            [f.default_url] if f.default_url else [],
        )

    jobs = f.jobs
    if jobs is not None and jobs.open_roles > 0:
        n = jobs.open_roles
        board = f"their {jobs.ats.title()} board" if jobs.ats else "their public job board"
        if n >= 150:
            score, reads = 3.0, ("hiring at a scale that implies well over 1,000 staff, "
                                 "above the band this rubric targets")
        elif n >= 60:
            score, reads = 6.0, ("consistent with a late-stage company of roughly 500+, "
                                 "near the top of the target band")
        elif n >= 20:
            score, reads = 9.5, ("the hiring pattern of a company scaling through the "
                                 "51-500 band this rubric targets")
        elif n >= 6:
            score, reads = 8.5, ("steady hiring, consistent with an established team "
                                 "inside the target band")
        elif n >= 1:
            score, reads = 6.0, ("light hiring, consistent with an early or deliberately "
                                 "small team")
        else:
            score, reads = 4.0, "no live reqs, so no momentum to read"
        gtm = jobs.gtm_roles
        gtm_bit = f", {gtm} of them in GTM functions" if gtm else ""
        return (
            _clamp(score),
            f"{n} open roles on {board}{gtm_bit} — {reads}. A company running {n} live reqs "
            f"is demonstrably not a handful of people. Exact headcount is still unconfirmed, "
            f"so this is graded on hiring evidence rather than a headcount band.",
            [jobs.board_url] if jobs.board_url else ([f.default_url] if f.default_url else []),
        )

    return (
        5.0,
        "Neither a headcount band nor a public job board could be found, so size is "
        "genuinely unknown and scored mid rather than penalised. It is a gap, not a "
        "negative — see `confidence`.",
        [],
    )


def _sc_tech_maturity(f: _Facts) -> tuple[float, str, list[str]]:
    if not f.tech_names:
        return (
            5.0,
            "No technologies were detected, so stack maturity is unknown. Scored mid "
            "rather than zero: absence of detection is not absence of tooling.",
            [],
        )
    crm, data, infra = f.has(_CRM_NAMES), f.has(_DATA_NAMES), f.has(_INFRA_NAMES)
    breadth = len(f.tech_categories)
    score = 2.0 + min(5.0, 1.2 * breadth)
    parts = [
        f"{_plural(len(f.tech_names), 'technology', 'technologies')} across "
        f"{_plural(breadth, 'category', 'categories')}"
    ]
    if crm:
        score += 1.5
        parts.append(f"a modern CRM ({_join_names(crm)})")
    if data:
        score += 1.5
        parts.append(f"analytics/data tooling ({_join_names(data[:3])})")
    if infra:
        score += 1.0
        parts.append(f"cloud-native infrastructure ({_join_names(infra[:3])})")
    return (
        _clamp(score),
        "Detected " + "; ".join(parts) + ". Breadth of best-of-breed SaaS is the proxy "
        "for buy-not-build behaviour.",
        [f.default_url] if f.default_url else [],
    )


def _sc_growth_signals(f: _Facts) -> tuple[float, str, list[str]]:
    """Trigger events, discounted by age.

    A nine-day-old product launch is a reason to call this week; a funding round
    from two years ago is history. Both used to score the same. Each signal's
    contribution is now multiplied by `recency_weight(age)`, and the reasoning
    names which signals drove the score and how old they were.
    """
    from lea.enrich.extract import describe_age, recency_weight, signal_age_days

    base_points = {"funding": 5.0, "hiring": 2.5, "product": 1.0,
                   "leadership": 0.5, "partnership": 0.5, "news": 0.5}
    caps = {"product": 2.0, "leadership": 1.0, "partnership": 1.0, "news": 1.0}

    score = 2.0
    spent: dict[str, float] = {}
    scored: list[tuple[float, str, Optional[int]]] = []

    for sig in f.record.signals or []:
        pts = base_points.get(sig.kind)
        if pts is None:
            continue
        age = signal_age_days(sig, f.today)
        weighted = pts * recency_weight(age)
        cap = caps.get(sig.kind)
        if cap is not None:
            room = max(0.0, cap - spent.get(sig.kind, 0.0))
            weighted = min(weighted, room)
            spent[sig.kind] = spent.get(sig.kind, 0.0) + weighted
        elif sig.kind in spent:
            continue                      # one funding/hiring signal is enough
        else:
            spent[sig.kind] = weighted
        if weighted <= 0:
            continue
        score += weighted
        scored.append((weighted, f'{sig.kind}: "{(sig.headline or "")[:70]}"', age))

    if not scored:
        return (
            2.0,
            "No funding, hiring, product or leadership signal was found in the evidence. "
            "Without a trigger event there is no urgency to sell into.",
            [],
        )

    scored.sort(key=lambda t: -t[0])
    described = "; ".join(f"{label} ({describe_age(age)})" for _w, label, age in scored[:3])
    freshest = min((a for _w, _l, a in scored if a is not None), default=None)
    if freshest is not None and freshest <= 30:
        tail = (f"The freshest is {describe_age(freshest)}, which is current enough to open "
                f"a conversation on this week.")
    elif freshest is not None and freshest > 365:
        tail = ("Everything here is over a year old, so it is context rather than a reason "
                "to call now — scored down accordingly.")
    else:
        tail = "Signals are weighted by age, so older events count for less."

    return (_clamp(score), f"Trigger events — {described}. {tail}",
            f.urls_for("funding", "hiring", "product"))


def _sc_geo(f: _Facts) -> tuple[float, str, list[str]]:
    rec = f.record
    country = (rec.hq_country or "").strip().lower()
    city = (rec.hq_city or "").strip()
    if not country and not city:
        return (5.0, "Headquarters location was not established. Scored mid pending evidence.", [])
    loc = ", ".join(x for x in [city, rec.hq_country] if x)
    if country in _STRONG_GEO:
        return (10.0, f"Headquartered in {loc} — a core market we sell into today.",
                [f.default_url] if f.default_url else [])
    if country in _MEDIUM_GEO:
        return (6.0, f"Headquartered in {loc} — a serviceable secondary market.",
                [f.default_url] if f.default_url else [])
    if country:
        return (3.0, f"Headquartered in {loc}, outside the core and secondary markets.",
                [f.default_url] if f.default_url else [])
    return (6.0, f"City is {city} but the country was not established; scored on the city alone.",
            [f.default_url] if f.default_url else [])


def _sc_generic(f: _Facts, crit: ICPCriterion) -> tuple[float, str, list[str]]:
    """A criterion this scorer has no specific heuristic for."""
    known = sum(
        1 for v in [f.record.domain, f.record.industry, f.record.employee_range,
                    f.record.hq_country, f.record.signals, f.record.tech_stack] if v
    )
    score = 3.0 + known  # 3..9
    return (
        _clamp(score),
        f"No deterministic heuristic exists for '{crit.key}', so this is scored from overall "
        f"record completeness ({known}/6 evidence areas established). A real model would "
        f"judge this criterion directly — set ANTHROPIC_API_KEY.",
        [f.default_url] if f.default_url else [],
    )


#: key/description keywords -> scorer. First match wins.
_SCORER_ROUTES: list[tuple[tuple[str, ...], Any]] = [
    (("size", "headcount", "employee", "scale"), _sc_company_size),
    (("tech", "stack", "maturity", "tooling", "infrastructure"), _sc_tech_maturity),
    (("growth", "signal", "trigger", "funding", "momentum", "hiring"), _sc_growth_signals),
    (("geo", "region", "location", "country", "market_fit", "territory"), _sc_geo),
    (("gtm", "segment", "motion", "sales", "revenue", "buyer"), _sc_gtm_segment),
    (("type", "b2b", "model", "business", "vertical", "industry"), _sc_company_type),
]


def _route(crit: ICPCriterion):
    hay = f"{crit.key} {crit.description}".lower()
    key = crit.key.lower()
    for words, fn in _SCORER_ROUTES:        # key match is the strong signal
        if any(w in key for w in words):
            return fn
    for words, fn in _SCORER_ROUTES:        # fall back to the description
        if any(w in hay for w in words):
            return fn
    return None


def deterministic_scores(
    record: CompanyRecord,
    icp: ICPProfile,
    allowed_urls: Optional[set[str]] = None,
) -> list[CriterionScore]:
    """Grade every criterion from the record itself, with cited reasoning."""
    f = _Facts(record)
    out: list[CriterionScore] = []
    for crit in icp.criteria:
        fn = _route(crit)
        try:
            score, reasoning, urls = fn(f) if fn else _sc_generic(f, crit)
        except Exception:
            score, reasoning, urls = 5.0, "Scoring this criterion failed; treated as unknown.", []
        if allowed_urls is not None:
            urls = [u for u in urls if u in allowed_urls]
        out.append(
            CriterionScore(
                key=crit.key,
                score=score,
                weight=crit.weight,
                reasoning=reasoning,
                evidence_urls=[u for u in urls if u],
            )
        )
    return out


def deterministic_disqualifiers(record: CompanyRecord) -> list[str]:
    """Hard stops the record positively establishes, as claim strings.

    Deliberately one conservative rule. These claims are still passed through
    `_match_disqualifier`, so nothing fires unless the profile actually declares
    a matching disqualifier — the same gate the model's claims go through.
    """
    f = _Facts(record)
    claims: list[str] = []
    consumer = f.any_word(_CONSUMER_WORDS)
    if consumer and not f.any_word(_B2B_WORDS) and not f.has(_CRM_NAMES):
        claims.append(
            f"consumer-only B2C company with no business-facing product "
            f"(matched '{consumer}', and no B2B product or CRM evidence)"
        )
    return claims


def _is_degenerate(raw: Optional[ICPFit], icp: ICPProfile) -> bool:
    """Did the model actually judge anything, or just fill in defaults?

    The keyless stub returns an empty criteria list; a bad model response can also
    come back all-identical or with no reasoning. Any of those means the numbers
    carry no information and must not be presented as a score.
    """
    if raw is None or not raw.criteria:
        return True
    wanted = {c.key for c in icp.criteria}
    matched = [c for c in raw.criteria if snake_case(c.key) in wanted]
    if not matched:
        # Nothing the model returned maps onto this rubric. A partial response is
        # fine — `_align_scores` grades the missing criteria from the record and
        # keeps the judgements the model did make.
        return True
    if not any((c.reasoning or "").strip() for c in matched):
        return True
    scores = {round(float(c.score), 2) for c in matched}
    reasons = {(c.reasoning or "").strip().lower() for c in matched}
    if len(matched) > 1 and len(scores) <= 1 and len(reasons) <= 1:
        # Same number and the same sentence for every criterion: a default-filled
        # response, not a judgement. Identical scores with *distinct* reasoning is
        # a legitimate (if unusual) outcome and is left alone.
        return True
    return False


def _align_scores(
    raw: list[CriterionScore],
    icp: ICPProfile,
    allowed_urls: set[str],
    record: Optional[CompanyRecord] = None,
) -> list[CriterionScore]:
    """Force the model's criterion list onto the profile's: same keys, same weights.

    Weights come from the profile, never from the model — otherwise the model can
    silently reweight the rubric it was asked to apply. Evidence URLs are filtered
    to ones actually gathered, so a citation cannot be invented. A criterion the
    model skipped is graded deterministically rather than parked at a flat 5.
    """
    by_key: dict[str, CriterionScore] = {}
    for c in raw or []:
        if c is None:
            continue
        by_key.setdefault(snake_case(c.key), c)

    fallback: dict[str, CriterionScore] = {}
    if record is not None:
        fallback = {c.key: c for c in deterministic_scores(record, icp, allowed_urls)}

    out: list[CriterionScore] = []
    for crit in icp.criteria:
        got = by_key.get(crit.key) or by_key.get(snake_case(crit.key))
        if got is None:
            sub = fallback.get(crit.key)
            if sub is not None:
                sub = sub.model_copy(deep=True)
                sub.reasoning = "The model skipped this criterion; graded from the record. " + sub.reasoning
                out.append(sub)
            else:
                out.append(
                    CriterionScore(
                        key=crit.key,
                        score=5.0,
                        weight=crit.weight,
                        reasoning="The model did not score this criterion; treated as unknown (5/10).",
                        evidence_urls=[],
                    )
                )
            continue
        try:
            score = float(got.score)
        except (TypeError, ValueError):
            score = 5.0
        score = max(0.0, min(10.0, score))
        urls = [u for u in (got.evidence_urls or []) if u in allowed_urls]
        # Narrow override: the model punted to the neutral 5/10 with nothing to
        # cite, while the rubric scorer has an actual citation. Observed live as
        # "Employee range is unknown... using the fallback unknown score of 5",
        # sometimes alongside a headcount recalled from training data. A cited
        # grade beats an uncited non-answer. Deliberately does NOT fire when the
        # model made a real judgement (any score but the 5.0 fallback) - an
        # uncited good judgement is still a judgement.
        sub = fallback.get(crit.key)
        if (
            not urls
            and score == 5.0
            and sub is not None
            and sub.evidence_urls
            and sub.score != 5.0
        ):
            sub = sub.model_copy(deep=True)
            sub.reasoning = (
                "Superseded the model's uncited grade "
                f"({score:.0f}/10) with cited evidence. " + sub.reasoning
            )
            out.append(sub)
            continue

        reasoning = (got.reasoning or "").strip() or "No reasoning returned."
        if not urls and "no evidence" not in reasoning.lower():
            reasoning = reasoning.rstrip(".") + ". (No source URL cited for this criterion.)"
        out.append(
            CriterionScore(
                key=crit.key,
                score=score,
                weight=crit.weight,  # profile is authoritative
                reasoning=reasoning,
                evidence_urls=urls,
            )
        )
    return out


def _build_scoring_context(record: CompanyRecord, icp: ICPProfile, evidence: list[tuple[str, str]]) -> str:
    lines: list[str] = ["ICP PROFILE: " + (icp.name or "default")]
    if icp.summary:
        lines.append(icp.summary)
    lines.append("")
    lines.append("CRITERIA TO SCORE (use these exact keys):")
    for c in icp.criteria:
        lines.append(f"- {c.key} (weight {c.weight}): {c.description}")
        if c.rubric:
            lines.append(f"    rubric: {c.rubric}")
    if icp.disqualifiers:
        lines.append("")
        lines.append("DISQUALIFIERS (report only ones you can positively confirm):")
        for d in icp.disqualifiers:
            lines.append(f"- {d}")

    # Give the model the date and the pre-computed signal ages. Asking it to do
    # date arithmetic produced "June 2025 ... well within the 90-day recency
    # window" on a run dated September 2026. Same principle as never letting it
    # total the score: compute it here, hand it the answer.
    from lea.enrich.extract import describe_age, signal_age_days

    today = datetime.now(timezone.utc)
    lines += ["", f"TODAY'S DATE: {today.date().isoformat()}"]
    if record.signals:
        lines.append("SIGNAL AGES (already computed - do not recalculate):")
        for sig in record.signals:
            age = signal_age_days(sig, today)
            lines.append(f'- "{(sig.headline or "")[:70]}" -> {describe_age(age)}')

    lines += ["", "COMPANY RECORD:", record.model_dump_json(
        indent=2,
        exclude={"icp_fit", "hooks"},
    )]

    if evidence:
        lines += ["", "SOURCE EVIDENCE (cite these URLs in evidence_urls):"]
        for url, text in evidence:
            snippet = (text or "").strip()
            if len(snippet) > 2000:
                snippet = snippet[:2000] + " ...[truncated]"
            lines.append(f"--- source: {url} ---")
            lines.append(snippet)
    return "\n".join(lines)


def score_icp(
    record: CompanyRecord,
    icp: ICPProfile,
    evidence: list[tuple[str, str]],
    llm: LLMClient,
    settings: Settings,
) -> ICPFit:
    """Score one record against one profile. Never raises."""
    evidence = list(evidence or [])
    if not icp.criteria:
        icp = DEFAULT_ICP

    # Refuse to score a company we learned nothing about. Calling the LLM here
    # would produce a fluent, confident-looking rubric built on an empty record.
    if is_unresearched(record, evidence):
        return _non_assessment(record, icp)

    allowed_urls = {str(u).strip() for (u, _t) in evidence if u} | {
        str(u).strip() for u in (record.sources or []) if u
    }

    raw_fit: Optional[ICPFit] = None
    try:
        raw_fit, _usage = llm.extract(
            _scoring_prompt(),
            _build_scoring_context(record, icp, evidence),
            ICPFit,
            hint="Score each criterion 0-10 with reasoning and evidence_urls. Do not compute the total.",
        )
    except Exception:
        raw_fit = None

    if _is_degenerate(raw_fit if isinstance(raw_fit, ICPFit) else None, icp):
        # The model judged nothing (the keyless stub returns no criteria at all).
        # Grading from the record beats presenting an identical 50/C for everyone.
        scored = deterministic_scores(record, icp, allowed_urls)
        claimed_dq = deterministic_disqualifiers(record)
        model_summary = ""
        graded_by = "rubric"
    else:
        assert isinstance(raw_fit, ICPFit)
        scored = _align_scores(list(raw_fit.criteria), icp, allowed_urls, record)
        claimed_dq = list(raw_fit.disqualifiers_hit or [])
        model_summary = (raw_fit.summary or "").strip()
        graded_by = "model"

    # --- the arithmetic, done here, deliberately -------------------------
    score = compute_fit_score(scored)

    # Only disqualifiers the profile actually declares can fire.
    hits: list[str] = []
    for claimed in claimed_dq:
        matched = _match_disqualifier(claimed, icp.disqualifiers)
        if matched and matched not in hits:
            hits.append(matched)

    if hits:
        score = min(score, DISQUALIFIED_SCORE_CAP)
        tier = "D"
    else:
        tier = tier_for(score)

    confidence = _compute_confidence(scored, record)

    summary = model_summary
    if not summary:
        ranked = sorted(scored, key=lambda c: (c.score * c.weight), reverse=True)
        top = ranked[:2]
        weak = [c for c in ranked if c.score <= 4.0][-1:]
        summary = (
            f"Tier {tier} ({score:.0f}/100). Strongest: "
            + ", ".join(f"{c.key} {c.score:.0f}/10" for c in top)
            + "."
        )
        if weak:
            summary += f" Weakest: {weak[0].key} {weak[0].score:.0f}/10."
        fresh = _freshest_signal_phrase(record)
        if fresh:
            summary += f" {fresh}"
        if graded_by == "rubric":
            summary += (
                " Graded by the deterministic rubric (no LLM judgement available) — "
                "set ANTHROPIC_API_KEY for reasoned per-criterion scoring."
            )
    if hits:
        summary = f"DISQUALIFIED — {hits[0]} " + summary
    if confidence < 0.4:
        summary += (
            f" Low confidence ({confidence:.2f}): most criteria were graded without "
            "direct evidence, so treat this as a provisional score."
        )

    return ICPFit(
        score=score,
        tier=tier,  # type: ignore[arg-type]
        summary=summary.strip(),
        criteria=scored,
        disqualifiers_hit=hits,
        confidence=confidence,
    )


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def _profile_to_dict(profile: ICPProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "summary": profile.summary,
        "criteria": [
            {
                "key": c.key,
                "description": c.description,
                "weight": float(c.weight),
                "rubric": c.rubric,
            }
            for c in profile.criteria
        ],
        "disqualifiers": list(profile.disqualifiers),
        "derived_from": list(profile.derived_from),
        "created_at": profile.created_at,
    }


def save_icp(profile: ICPProfile, path: str | os.PathLike[str]) -> Path:
    """Write a human-editable YAML profile, with an explanatory header block."""
    p = Path(path)
    if p.parent and str(p.parent):
        p.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        _profile_to_dict(profile),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )
    p.write_text(_YAML_HEADER + "\n" + body, encoding="utf-8")
    return p


def load_icp(path: str | os.PathLike[str]) -> ICPProfile:
    """Read a YAML profile. Raises if the file is missing or unparseable —
    an explicitly requested profile failing silently would be worse."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected a YAML mapping, got {type(data).__name__}")
    profile = ICPProfile.model_validate(data)
    profile.criteria = _repair_criteria(profile.criteria)
    return profile


def load_or_default_icp(path: Optional[str | os.PathLike[str]], settings: Settings) -> ICPProfile:
    """Resolve a profile: explicit path -> ./icp.yaml -> the shipped default.

    Contractually never raises. A broken icp.yaml degrades to the default rather
    than taking down a batch run halfway through.
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.append(Path.cwd() / DEFAULT_ICP_FILENAME)
        try:
            candidates.append(Path(settings.out_dir).parent / DEFAULT_ICP_FILENAME)
        except Exception:
            pass

    for c in candidates:
        try:
            if c.is_file():
                return load_icp(c)
        except Exception:
            continue
    return DEFAULT_ICP.model_copy(deep=True)
