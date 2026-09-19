"""Evidence -> CompanyRecord.

The LLM proposes; this module disposes. Everything the model returns gets
normalized, deduped, range-checked and re-sourced before it becomes a record,
because "the model said so" is not a data quality strategy.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable, Optional

from lea.config import Settings
from lea.llm.base import LLMClient
from lea.models import CompanyRecord, Signal, TechStack

__all__ = [
    "extract_record",
    "merge_records",
    "normalize_record",
    "normalize_domain",
    "normalize_employee_range",
    "bucket_midpoint",
    "coerce_founded_year",
    "sanitize_company_name",
    "derive_company_name",
    "choose_domain",
    "is_aggregator",
    "registrable_label",
    "citable_urls",
    "AGGREGATOR_DOMAINS",
    "MAX_COMPANY_NAME_CHARS",
    "PRIORITY_FIELDS",
    "EMPLOYEE_BUCKETS",
]

# --------------------------------------------------------------------------
# Fallback prompt, used until lea.prompts lands (Agent B) or if it fails to import.
# --------------------------------------------------------------------------

_FALLBACK_EXTRACTION_SYSTEM = (
    "You are a B2B data analyst. Read the evidence below and fill the requested JSON "
    "schema describing the company. Rules: (1) never invent a fact that is not present "
    "in the evidence; (2) leave a field null rather than guessing; (3) every signal and "
    "tech_stack entry must carry the source_url / evidence it came from; (4) prefer the "
    "most recent evidence when sources disagree; (5) keep `description` to two sentences."
)


def _fallback_context(target: str, evidence_blocks: list[tuple[str, str]]) -> str:
    parts = [f"TARGET COMPANY: {target}", ""]
    for i, (url, text) in enumerate(evidence_blocks, 1):
        parts.append(f"--- EVIDENCE {i} | source: {url} ---")
        parts.append((text or "").strip())
        parts.append("")
    if len(evidence_blocks) == 0:
        parts.append("(no evidence gathered)")
    return "\n".join(parts)


def _prompt_bits() -> tuple[str, object]:
    """Lazy import of Agent B's prompts with a local fallback."""
    try:  # pragma: no cover - trivial import shim
        from lea.prompts import EXTRACTION_SYSTEM, build_extraction_context

        return EXTRACTION_SYSTEM, build_extraction_context
    except Exception:
        return _FALLBACK_EXTRACTION_SYSTEM, _fallback_context


# --------------------------------------------------------------------------
# Normalizers
# --------------------------------------------------------------------------

EMPLOYEE_BUCKETS: list[str] = [
    "1-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1000",
    "1001-5000",
    "5000+",
]

_BUCKET_BOUNDS: list[tuple[str, int, int]] = [
    ("1-10", 1, 10),
    ("11-50", 11, 50),
    ("51-200", 51, 200),
    ("201-500", 201, 500),
    ("501-1000", 501, 1000),
    ("1001-5000", 1001, 5000),
    ("5000+", 5001, 20000),
]

_BUCKET_MIDPOINTS: dict[str, int] = {
    "1-10": 5,
    "11-50": 30,
    "51-200": 125,
    "201-500": 350,
    "501-1000": 750,
    "1001-5000": 3000,
    "5000+": 7500,
}

PRIORITY_FIELDS: list[str] = [
    "company_name",
    "domain",
    "industry",
    "employee_range",
    "hq_city",
    "hq_country",
    "founded_year",
]

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_NUM_RE = re.compile(r"\d[\d,\.]*")


def normalize_domain(raw: Optional[str]) -> Optional[str]:
    """`https://WWW.Ramp.com/careers/` -> `ramp.com`."""
    if not raw:
        return None
    d = str(raw).strip().lower()
    if not d:
        return None
    d = _SCHEME_RE.sub("", d)
    d = d.split("/", 1)[0]
    d = d.split("?", 1)[0].split("#", 1)[0]
    d = d.split("@")[-1]          # strip any user info / email local part
    d = d.split(":", 1)[0]        # strip port
    while d.startswith("www."):
        d = d[4:]
    d = d.strip(". ")
    if not d or "." not in d or " " in d:
        return None
    return d


def _parse_int(token: str) -> Optional[int]:
    m = _NUM_RE.search(token)
    if not m:
        return None
    cleaned = m.group(0).replace(",", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _bucket_for(n: int) -> str:
    for name, lo, hi in _BUCKET_BOUNDS:
        if lo <= n <= hi:
            return name
    return "5000+" if n > 5000 else "1-10"


def normalize_employee_range(raw: Optional[str | int]) -> Optional[str]:
    """Map anything headcount-ish onto a canonical bucket.

    Handles "~250 employees", "500-1000", "10,001+", "51 - 200", "1k-5k", ints.
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return _bucket_for(raw) if raw > 0 else None

    s = str(raw).strip().lower()
    if not s:
        return None
    if s in {"unknown", "n/a", "na", "none", "-"}:
        return None

    # exact canonical hit first
    compact = s.replace(" ", "")
    for b in EMPLOYEE_BUCKETS:
        if compact == b:
            return b

    # k-suffix expansion: "1k-5k" -> "1000-5000"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", lambda m: str(int(float(m.group(1)) * 1000)), s)

    plus = "+" in s or "more than" in s or "over" in s
    nums = [_parse_int(t) for t in re.findall(r"\d[\d,\.]*", s)]
    nums = [n for n in nums if n is not None and n >= 0]
    if not nums:
        return None

    if len(nums) >= 2:
        lo, hi = min(nums[0], nums[1]), max(nums[0], nums[1])
        # Pick the bucket the range's midpoint sits in — a "500-1000" lands in 501-1000.
        mid = (lo + hi) / 2
        return _bucket_for(int(round(mid)))

    n = nums[0]
    if n <= 0:
        return None
    if plus and n >= 5000:
        return "5000+"
    return _bucket_for(n)


def bucket_midpoint(bucket: Optional[str]) -> Optional[int]:
    if not bucket:
        return None
    return _BUCKET_MIDPOINTS.get(bucket.strip())


def coerce_founded_year(raw: object) -> Optional[int]:
    """Accept 1998, "1998", "founded in 1998", "1998-03-01". Reject nonsense."""
    if raw is None:
        return None
    current = datetime.now(timezone.utc).year
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        y = int(raw)
    else:
        m = re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", str(raw))
        if not m:
            return None
        y = int(m.group(1))
    if 1800 <= y <= current:
        return y
    return None


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def _norm_headline(h: str) -> str:
    s = _PUNCT_RE.sub(" ", (h or "").lower())
    return _WS_RE.sub(" ", s).strip()


_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %Y",
    "%b %Y",
    "%Y-%m",
    "%Y",
]


def _date_key(raw: Optional[str]) -> Optional[tuple[int, int, int]]:
    """Best-effort sortable date. None means 'undated' and sorts last."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    for fmt in _DATE_PATTERNS:
        try:
            dt = datetime.strptime(s, fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue
    # loose: pull a year, optionally a month name
    ym = re.search(r"\b(1[89]\d{2}|20\d{2})\b", s)
    if not ym:
        return None
    year = int(ym.group(1))
    month = 0
    for i, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    ):
        if name in s.lower():
            month = i
            break
    return (year, month, 0)


def _dedupe_signals(signals: Iterable[Signal]) -> list[Signal]:
    """Keep the first occurrence per normalized headline, preferring one with a
    date/source over a bare duplicate."""
    best: dict[str, Signal] = {}
    order: list[str] = []
    for s in signals or []:
        if not s or not getattr(s, "headline", "").strip():
            continue
        key = _norm_headline(s.headline)
        if not key:
            continue
        if key not in best:
            best[key] = s
            order.append(key)
            continue
        incumbent = best[key]
        # richer entry wins: has a date, then has a source, then longer detail
        def rank(x: Signal) -> tuple[int, int, int]:
            return (1 if x.date else 0, 1 if x.source_url else 0, len(x.detail or ""))

        if rank(s) > rank(incumbent):
            best[key] = s
    return [best[k] for k in order]


def _sort_signals(signals: list[Signal]) -> list[Signal]:
    """Most recent first; undated last, stable within each group."""
    dated: list[tuple[tuple[int, int, int], int, Signal]] = []
    undated: list[Signal] = []
    for i, s in enumerate(signals):
        k = _date_key(s.date)
        if k is None:
            s.recency_note = s.recency_note or "undated"
            undated.append(s)
        else:
            dated.append((k, i, s))
    dated.sort(key=lambda t: (t[0][0], t[0][1], t[0][2], -t[1]), reverse=True)
    return [t[2] for t in dated] + undated


def _dedupe_tech(items: Iterable[TechStack]) -> list[TechStack]:
    """One entry per (category, name); highest confidence wins."""
    best: dict[tuple[str, str], TechStack] = {}
    order: list[tuple[str, str]] = []
    for t in items or []:
        if not t or not getattr(t, "name", "").strip():
            continue
        key = ((t.category or "other").strip().lower(), t.name.strip().lower())
        if key not in best:
            best[key] = t
            order.append(key)
        elif (t.confidence or 0) > (best[key].confidence or 0):
            best[key] = t
    return [best[k] for k in order]


def _unique_urls(urls: Iterable[Optional[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if not u:
            continue
        u = str(u).strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _is_set(value: object) -> bool:
    """Is this field confidently populated? Empty string / list / None are not."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _compute_unknowns(rec: CompanyRecord) -> list[str]:
    missing = [f for f in PRIORITY_FIELDS if not _is_set(getattr(rec, f, None))]
    if not rec.signals:
        missing.append("signals")
    if not rec.tech_stack:
        missing.append("tech_stack")
    if not _is_set(rec.description):
        missing.append("description")
    return missing


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------

_SCALAR_FIELDS = [
    "company_name",
    "domain",
    "description",
    "industry",
    "sub_industry",
    "employee_range",
    "employee_estimate",
    "hq_city",
    "hq_country",
    "founded_year",
]


def merge_records(a: CompanyRecord, b: CompanyRecord) -> CompanyRecord:
    """Merge `b` into `a`. `a` wins for any scalar it already has set.

    Lists are unioned and re-deduped; `unknowns` is recomputed from the result so
    a merge can only ever shrink the gap list, never leave a stale one.
    """
    a = _ensure_record(a, getattr(b, "input", "") or "")
    b = _ensure_record(b, a.input)
    merged = a.model_copy(deep=True)

    for f in _SCALAR_FIELDS:
        if not _is_set(getattr(merged, f, None)):
            val = getattr(b, f, None)
            if _is_set(val):
                setattr(merged, f, val)

    if not _is_set(merged.input):
        merged.input = b.input

    merged.signals = _sort_signals(_dedupe_signals(list(merged.signals) + list(b.signals)))
    merged.tech_stack = _dedupe_tech(list(merged.tech_stack) + list(b.tech_stack))
    merged.sources = _unique_urls(list(merged.sources) + list(b.sources))

    if merged.icp_fit is None and b.icp_fit is not None:
        merged.icp_fit = b.icp_fit
    if not merged.hooks and b.hooks:
        merged.hooks = list(b.hooks)

    # Keep derived fields coherent after the merge.
    merged.employee_range = normalize_employee_range(merged.employee_range)
    if merged.employee_estimate is None:
        merged.employee_estimate = bucket_midpoint(merged.employee_range)
    merged.domain = choose_domain(
        merged.domain, merged.input, merged.company_name, merged.sources
    )
    merged.founded_year = coerce_founded_year(merged.founded_year)
    merged.unknowns = _compute_unknowns(merged)
    return merged


# --------------------------------------------------------------------------
# Identity sanitation
#
# company_name and domain are what get rendered as the headline of the report,
# and they are also the two fields the model is most likely to get *plausibly*
# wrong: a page <title> instead of a name, an aggregator profile instead of the
# company's own site. Both failures look fine until a human reads the output.
# --------------------------------------------------------------------------

#: Hosts that are never a company's own domain. Module-level so it is editable
#: and testable. Overridden when the host actually matches the target, so that
#: GitHub still resolves to github.com.
AGGREGATOR_DOMAINS: set[str] = {
    "wikipedia.org", "linkedin.com", "crunchbase.com", "tracxn.com",
    "pitchbook.com", "glassdoor.com", "bloomberg.com", "zoominfo.com",
    "apollo.io", "owler.com", "g2.com", "capterra.com", "facebook.com",
    "twitter.com", "x.com", "youtube.com", "medium.com", "github.com",
    "gitlab.com", "reddit.com", "instagram.com", "tiktok.com", "indeed.com",
    "builtin.com", "angel.co", "wellfound.com", "producthunt.com",
    "forbes.com", "techcrunch.com", "businesswire.com", "prnewswire.com",
    "dnb.com", "zippia.com", "rocketreach.co", "leadiq.com", "craft.co",
}

#: Words that mark a string as page furniture rather than a company name.
_PAGE_TITLE_WORDS = {
    "about", "home", "homepage", "careers", "career", "pricing", "blog",
    "contact", "login", "signup", "privacy", "terms", "faq", "support",
    "docs", "documentation", "press", "welcome", "overview", "index",
    "not found", "page not found",
}

#: Left-hand title segments too generic to be a brand.
_GENERIC_SEGMENTS = {
    "about", "about us", "home", "homepage", "welcome", "us", "overview",
    "company", "our company", "who we are", "index", "untitled", "news",
}

MAX_COMPANY_NAME_CHARS = 60

#: Two-part public suffixes, so "ramp.co.uk" yields the label "ramp".
_MULTI_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "co.nz", "co.za", "co.in",
    "com.au", "com.br", "com.mx", "com.sg", "com.tr", "co.il", "co.kr",
}

_TITLE_SPLIT_RE = re.compile(r"\s*[|•·—–]\s*|\s+-\s+|\s*::\s*")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("", (text or "").lower())


def registrable_label(host: Optional[str]) -> str:
    """`sub.ramp.co.uk` -> `ramp`. Best-effort; no public-suffix-list dependency."""
    h = normalize_domain(host) or ""
    if not h:
        return ""
    parts = h.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _MULTI_SUFFIXES:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def _matches_target(host: str, *names: Optional[str]) -> bool:
    """Does this host plausibly belong to the company we asked about?"""
    label = _slug(registrable_label(host))
    if len(label) < 3:
        return False
    for n in names:
        s = _slug(n or "")
        if len(s) < 3:
            continue
        if s == label or s in label or label in s:
            return True
    return False


def is_aggregator(host: Optional[str], *names: Optional[str]) -> bool:
    """True if `host` is a directory/profile site rather than the company's own."""
    h = normalize_domain(host)
    if not h:
        return False
    hit = any(h == d or h.endswith("." + d) for d in AGGREGATOR_DOMAINS)
    if not hit:
        return False
    return not _matches_target(h, *names)


def choose_domain(
    candidate: Optional[str],
    target: str,
    company_name: Optional[str] = None,
    evidence_urls: Optional[list[str]] = None,
) -> Optional[str]:
    """Pick the company's own domain, never a directory site's.

    Order: a clean candidate that is not an aggregator; else the evidence URL
    whose host resembles the target; else the target if it was itself a domain;
    else nothing. Returning None is correct and lands in `unknowns` — reporting
    tracxn.com as the company's domain is not.
    """
    names = (target, company_name)
    cand = normalize_domain(candidate)
    if cand and not is_aggregator(cand, *names):
        return cand

    hosts: list[str] = []
    for u in evidence_urls or []:
        h = normalize_domain(u)
        if h and h not in hosts:
            hosts.append(h)

    for h in hosts:  # a host that looks like the company beats any other
        if _matches_target(h, *names) and not is_aggregator(h, *names):
            return h
    from_target = normalize_domain(target)
    if from_target and not is_aggregator(from_target, *names):
        return from_target
    # Deliberately no "just take the first clean host" fallback. Evidence
    # gathered while researching X can easily include a host belonging to
    # someone else, and attributing it to X is the same class of error as
    # citing an aggregator. Unknown is the honest answer.
    return None


def _has_page_word(text: str) -> bool:
    low = _WS_RE.sub(" ", " " + re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()) + " ")
    return any(f" {w} " in low for w in _PAGE_TITLE_WORDS)


def sanitize_company_name(
    raw: Optional[str],
    target: str = "",
    domain: Optional[str] = None,
) -> Optional[str]:
    """Strip brand furniture; reject anything that is really a page title.

    `"Ramp | Corporate Cards"` -> `"Ramp"`.
    `"About Us - Different By Design"` -> None (a <title>, not a name).
    """
    if not raw:
        return None
    s = _WS_RE.sub(" ", str(raw).strip().strip("\"'").strip())
    if not s:
        return None

    # A name matching its own domain is right even if it contains a page word
    # ("Home Depot" / homedepot.com), so check that before rejecting.
    if domain and _slug(s) and _slug(s) == _slug(registrable_label(domain)):
        return s if len(s) <= MAX_COMPANY_NAME_CHARS else None

    if _has_page_word(s):
        return None

    parts = [p.strip() for p in _TITLE_SPLIT_RE.split(s) if p.strip()]
    if len(parts) > 1:
        head = parts[0]
        if head.lower() in _GENERIC_SEGMENTS or len(head) < 2:
            return None
        s = head

    if len(s) > MAX_COMPANY_NAME_CHARS or s.lower() in _GENERIC_SEGMENTS:
        return None
    return s


def derive_company_name(target: str, domain: Optional[str]) -> Optional[str]:
    """Fallback name: the target as typed, else the domain's registrable label."""
    t = (target or "").strip()
    if t and not _SCHEME_RE.match(t) and "/" not in t and not normalize_domain(t):
        return t  # a plain name like "Ramp"
    label = registrable_label(domain or t)
    if label:
        return label.replace("-", " ").title()
    return t or None


_ERROR_PREFIXES = ("error", "fetch failed", "failed to fetch", "http error", "[error")


def citable_urls(evidence: list[tuple[str, str]]) -> list[str]:
    """The URLs we actually got usable text from.

    A fetch that 403'd or was robots-blocked still arrives in the evidence list,
    with an empty or error body. Citing it as a source would misrepresent where
    the data came from, so it is not citable.
    """
    out: list[str] = []
    seen: set[str] = set()
    for url, text in evidence or []:
        u = str(url or "").strip()
        if not u or u in seen:
            continue
        body = str(text or "").strip()
        if not body or body.lower().startswith(_ERROR_PREFIXES):
            continue
        seen.add(u)
        out.append(u)
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _ensure_record(obj: object, target: str) -> CompanyRecord:
    """Coerce whatever the LLM layer handed back into a fully-valid record.

    A provider that falls back to `model_construct()` returns an instance with
    required fields simply absent, so plain attribute access raises. Rebuild by
    field name and re-validate rather than trusting the object's shape.
    """
    if not isinstance(obj, CompanyRecord):
        return CompanyRecord(input=target)
    data: dict[str, object] = {}
    for name in CompanyRecord.model_fields:
        try:
            data[name] = getattr(obj, name)
        except AttributeError:
            continue
    if not data.get("input"):
        data["input"] = target
    try:
        return CompanyRecord.model_validate(data)
    except Exception:
        return CompanyRecord(input=target)


def normalize_record(
    rec: CompanyRecord,
    target: str,
    evidence_urls: Optional[list[str]] = None,
) -> CompanyRecord:
    """Apply every post-processing rule to a record. Pure; safe to call twice."""
    rec = _ensure_record(rec, target).model_copy(deep=True)

    rec.input = rec.input or target

    # Identity first: the domain constrains what counts as a valid name, and the
    # name is only trusted after the domain has been established.
    candidate_urls = list(evidence_urls) if evidence_urls is not None else list(rec.sources or [])
    rec.domain = choose_domain(rec.domain, target, rec.company_name, candidate_urls)

    name = sanitize_company_name(rec.company_name, target, rec.domain)
    if name is None:
        name = derive_company_name(target, rec.domain)
    rec.company_name = name or None

    rec.employee_range = normalize_employee_range(rec.employee_range)
    est = rec.employee_estimate
    if isinstance(est, (int, float)) and est and est > 0:
        rec.employee_estimate = int(est)
        # A raw headcount is better evidence than a guessed bucket — re-derive.
        if rec.employee_range is None:
            rec.employee_range = _bucket_for(int(est))
    else:
        rec.employee_estimate = bucket_midpoint(rec.employee_range)

    rec.founded_year = coerce_founded_year(rec.founded_year)

    for field in ("hq_city", "hq_country", "industry", "sub_industry"):
        v = getattr(rec, field, None)
        if isinstance(v, str):
            setattr(rec, field, v.strip() or None)
    rec.description = (rec.description or "").strip()

    rec.signals = _sort_signals(_dedupe_signals(rec.signals))
    rec.tech_stack = _dedupe_tech(rec.tech_stack)

    if evidence_urls is not None:
        rec.sources = _unique_urls(evidence_urls)
    else:
        rec.sources = _unique_urls(rec.sources)

    rec.unknowns = _compute_unknowns(rec)
    return rec


def extract_record(
    target: str,
    evidence: list[tuple[str, str]],
    llm: LLMClient,
    settings: Settings,
    base: Optional[CompanyRecord] = None,
) -> CompanyRecord:
    """Turn gathered evidence into a normalized CompanyRecord.

    `evidence` is a list of (source_url, text) blocks, in the order the agent
    gathered them. Never raises: a failed extraction degrades to an all-unknowns
    record rather than killing the run.
    """
    evidence = [(u or "", t or "") for (u, t) in (evidence or [])]
    system, build_ctx = _prompt_bits()

    try:
        context = build_ctx(target, evidence)  # type: ignore[operator]
    except Exception:
        context = _fallback_context(target, evidence)

    try:
        rec, _usage = llm.extract(
            system,
            context,
            CompanyRecord,
            hint=f"Extract the record for: {target}",
        )
    except Exception:
        rec = None

    if not isinstance(rec, CompanyRecord):
        rec = CompanyRecord(input=target)

    # Only evidence we actually read is citable — a robots-blocked or 404'd fetch
    # still appears in the list and must not end up in `sources`.
    evidence_urls = citable_urls(evidence)
    rec = normalize_record(rec, target, evidence_urls=evidence_urls)

    if base is not None:
        # base wins for anything it confidently holds.
        rec = merge_records(base, rec)
        rec.sources = _unique_urls(list(base.sources) + evidence_urls)
        rec.unknowns = _compute_unknowns(rec)

    return rec
