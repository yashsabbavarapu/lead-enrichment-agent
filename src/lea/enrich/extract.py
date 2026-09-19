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
    "apply_relevance_gate",
    "is_relevant_source",
    "relevant_evidence",
    "verify_record",
    "VERIFIED_FIELDS",
    "classify_funding",
    "reclassify_signals",
    "is_secondary_funding",
    "SECONDARY_NOTE",
    "parse_jobs_evidence",
    "jobs_from_record",
    "jobs_signal",
    "signal_age_days",
    "recency_weight",
    "describe_age",
    "JobsData",
    "GTM_DEPARTMENTS",
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


# --------------------------------------------------------------------------
# Job-board evidence
#
# Agent A's `fetch_jobs` tool reads public ATS boards and returns
#   {"open_roles": int, "departments": {name: count}, "locations": [...],
#    "titles": [...], "ats": "ashby", "board_url": "..."}
# but the agent loop flattens every observation to (source_url, text) before it
# reaches us, so that payload arrives as prose. We parse it back out tolerantly:
# JSON first, then the summary text. If the tool ships a different shape, the
# text path still recovers the two numbers that matter.
#
# Honesty rule: open roles are NEVER converted into an `employee_range`. A
# hiring count is evidence about stage and momentum, not a headcount, and
# inventing a band from it is exactly the plausible-fabrication failure this
# project exists to avoid.
# --------------------------------------------------------------------------

_ATS_HOSTS = ("ashby", "greenhouse", "lever.co", "workable", "jobs.", "boards.", "careers")
_OPEN_ROLES_RE = re.compile(r"(\d{1,4})\s+open\s+(?:roles?|positions?|jobs?)", re.I)
# Agent A's `fetch_jobs` renders "Engineering (21), Sales (9)"; we also accept
# "Engineering: 21" and "21 in Engineering" so a summary reword does not break us.
# Each word must be capitalised, so "roles on Ashby across Engineering (21)"
# yields "Engineering" rather than swallowing the prose before it.
_DEPT_PAREN_RE = re.compile(
    r"([A-Z][\w&/+'.]*(?:\s*[&/]?\s*[A-Z][\w&/+'.]*)*)\s*\((\d{1,4})\)")
_DEPT_COLON_RE = re.compile(r"([A-Z][\w&/+'. ]{1,28}?)\s*[:\u2013-]\s*(\d{1,4})\b")
_DEPT_IN_RE = re.compile(r"(\d{1,4})\s+in\s+([A-Z][\w&/+'.]*(?:\s[A-Z][\w&/+'.]*)*)")
_LATEST_POSTED_RE = re.compile(r"latest\s+(\d{4}-\d{2}-\d{2})", re.I)
_POSTED_30D_RE = re.compile(r"(\d{1,4})\s+posted\s+in\s+the\s+last\s+(\d{1,3})\s+days", re.I)
_ATS_NAME_RE = re.compile(r"\b(ashby|greenhouse|lever|workable)\b", re.I)
#: Host fragments, because "jobs.ashbyhq.com" has no word boundary after "ashby".
_ATS_HOST_NAMES = (("ashbyhq", "ashby"), ("greenhouse.io", "greenhouse"),
                   ("boards-api.greenhouse", "greenhouse"), ("lever.co", "lever"),
                   ("workable.com", "workable"))


def _ats_name(text: str, url: str) -> str:
    """Which ATS is this? Prefer the host, which cannot be a coincidence."""
    low = (url or "").lower()
    for fragment, name in _ATS_HOST_NAMES:
        if fragment in low:
            return name
    m = _ATS_NAME_RE.search(text or "")
    return m.group(1).lower() if m else ""

#: Departments a GTM seller actually cares about.
GTM_DEPARTMENTS = ("sales", "revenue", "marketing", "growth", "customer success",
                   "business development", "partnerships", "revops", "gtm")


class JobsData:
    """Parsed public-job-board evidence. A cheap value object, not a model."""

    __slots__ = ("open_roles", "departments", "locations", "ats", "board_url",
                 "latest_posted", "posted_recently")

    def __init__(self, open_roles: int = 0, departments: Optional[dict[str, int]] = None,
                 locations: Optional[list[str]] = None, ats: str = "",
                 board_url: str = "", latest_posted: str = "",
                 posted_recently: int = 0) -> None:
        self.open_roles = int(open_roles or 0)
        self.departments = {str(k): int(v) for k, v in (departments or {}).items() if str(k).strip()}
        self.locations = list(locations or [])
        self.ats = str(ats or "")
        self.board_url = str(board_url or "")
        self.latest_posted = str(latest_posted or "")
        self.posted_recently = int(posted_recently or 0)

    @property
    def gtm_roles(self) -> int:
        return sum(n for d, n in self.departments.items()
                   if any(g in d.lower() for g in GTM_DEPARTMENTS))

    def gtm_breakdown(self, k: int = 3) -> list[tuple[str, int]]:
        hits = [(d, n) for d, n in self.departments.items()
                if any(g in d.lower() for g in GTM_DEPARTMENTS)]
        return sorted(hits, key=lambda kv: (-kv[1], kv[0]))[:k]

    def top_departments(self, k: int = 3) -> list[tuple[str, int]]:
        return sorted(self.departments.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"JobsData(open_roles={self.open_roles}, departments={self.departments})"


def _jobs_from_json(text: str) -> Optional[JobsData]:
    if "{" not in (text or ""):
        return None
    try:
        import json
        start, end = text.index("{"), text.rindex("}") + 1
        blob = json.loads(text[start:end])
    except Exception:
        return None
    if not isinstance(blob, dict) or "open_roles" not in blob:
        return None
    try:
        return JobsData(
            open_roles=blob.get("open_roles") or 0,
            departments=blob.get("departments") or {},
            locations=blob.get("locations") or [],
            ats=blob.get("ats") or "",
            board_url=blob.get("board_url") or "",
            latest_posted=blob.get("latest_posted") or "",
            posted_recently=blob.get("posted_last_30d") or 0,
        )
    except Exception:
        return None


def _jobs_from_text(text: str, url: str) -> Optional[JobsData]:
    total_m = _OPEN_ROLES_RE.search(text or "")
    if not total_m:
        return None
    total = int(total_m.group(1))
    tail = text[total_m.end():][:600]      # the breakdown follows the total

    departments: dict[str, int] = {}
    for name, count in _DEPT_PAREN_RE.findall(tail):
        departments.setdefault(name.strip(" ,;.-"), int(count))
    if not departments:
        for name, count in _DEPT_COLON_RE.findall(tail):
            departments.setdefault(name.strip(" ,;.-"), int(count))
    if not departments:
        for count, name in _DEPT_IN_RE.findall(tail):
            departments.setdefault(name.strip(" ,;.-"), int(count))
    departments = {
        k: v for k, v in departments.items()
        if k and k.lower() not in {"open", "roles", "total", "and", "in", "remote", "latest"}
        and 0 < v <= total
    }

    latest_m = _LATEST_POSTED_RE.search(text)
    recent_m = _POSTED_30D_RE.search(text)
    return JobsData(
        open_roles=total,
        departments=departments,
        ats=_ats_name(text[:400], url),
        board_url=url,
        latest_posted=latest_m.group(1) if latest_m else "",
        posted_recently=int(recent_m.group(1)) if recent_m else 0,
    )


def parse_jobs_evidence(evidence: list[tuple[str, str]]) -> Optional[JobsData]:
    """Find and parse the job-board block in an evidence list, if there is one."""
    best: Optional[JobsData] = None
    for url, text in evidence or []:
        if not text:
            continue
        looks_like_board = any(h in (url or "").lower() for h in _ATS_HOSTS) or bool(
            _OPEN_ROLES_RE.search(text))
        if not looks_like_board:
            continue
        got = _jobs_from_json(text) or _jobs_from_text(text, url or "")
        if got is None:
            continue
        if not got.board_url:
            got.board_url = url or ""
        if best is None or (got.open_roles, len(got.departments)) > (
                best.open_roles, len(best.departments)):
            best = got
    return best


def jobs_signal(jobs: JobsData) -> Signal:
    """A hiring Signal carrying the counts in `detail`.

    The record model is frozen, so `detail` is where this structured fact lives.
    The format is deliberately both readable in the CLI and parseable back by
    `jobs_from_record`.
    """
    breakdown = ", ".join(f"{n} in {d}" for d, n in jobs.top_departments(4))
    where = f" on their {jobs.ats.title()} board" if jobs.ats else " on their public job board"
    detail = f"{jobs.open_roles} open roles{where}"
    if breakdown:
        detail += f" - {breakdown}"
    return Signal(
        kind="hiring",
        headline=f"{jobs.open_roles} open roles" + (f" ({breakdown})" if breakdown else ""),
        detail=detail + ".",
        date=jobs.latest_posted or None,
        source_url=jobs.board_url or None,
        recency_note=(
            f"{jobs.posted_recently} posted in the last 30 days"
            if jobs.posted_recently else "live job board"
        ),
    )


_READBACK_TOTAL_RE = re.compile(r"(\d{1,4})\s+open\s+roles", re.I)
_READBACK_DEPT_RE = re.compile(r"(\d{1,4})\s+in\s+([A-Z][\w&/+']*(?:\s[A-Z][\w&/+']*)*)")


def jobs_from_record(record: CompanyRecord) -> Optional[JobsData]:
    """Recover job-board facts from the hiring signal `jobs_signal` wrote.

    Lets the ICP scorer and hook writer reuse the department mix without adding
    a field to the frozen record model.
    """
    for s in record.signals or []:
        if s.kind != "hiring":
            continue
        blob = f"{s.headline or ''} {s.detail or ''} {s.recency_note or ''}"
        m = _READBACK_TOTAL_RE.search(blob)
        if not m:
            continue
        depts: dict[str, int] = {}
        for n, name in _READBACK_DEPT_RE.findall(blob):
            depts.setdefault(name.strip(), int(n))
        recent = _POSTED_30D_RE.search(blob)
        return JobsData(open_roles=int(m.group(1)), departments=depts,
                        ats=_ats_name(blob, s.source_url or ""),
                        board_url=s.source_url or "",
                        latest_posted=s.date or "",
                        posted_recently=int(recent.group(1)) if recent else 0)
    return None


# --------------------------------------------------------------------------
# Signal recency
# --------------------------------------------------------------------------

_RECENCY_NOTE_DAYS = [
    (re.compile(r"\blive\s+job\s+board\b", re.I), lambda m: 0),
    (re.compile(r"\bwithin\s+(\d+)\s+days?\b", re.I), lambda m: int(m.group(1))),
    (re.compile(r"\bwithin\s+a\s+week\b", re.I), lambda m: 7),
    (re.compile(r"\bwithin\s+a\s+month\b", re.I), lambda m: 30),
    (re.compile(r"\b(\d+)\s+days?\s+ago\b", re.I), lambda m: int(m.group(1))),
    (re.compile(r"\bover\s+a\s+year\s+ago\b", re.I), lambda m: 500),
    (re.compile(r"\bthis\s+(?:week|month)\b", re.I), lambda m: 14),
]


def signal_age_days(signal: Signal, today: Optional[datetime] = None) -> Optional[int]:
    """How old is this signal, in days? None means genuinely undated.

    Prefers the parsed `date`, falls back to phrases in `recency_note`. A live
    job board counts as today: it is a snapshot of right now.
    """
    now = today or datetime.now(timezone.utc)
    key = _date_key(signal.date) if signal.date else None
    if key is not None:
        y, m, d = key
        try:
            when = datetime(y, m or 1, d or 1, tzinfo=timezone.utc)
            return max(0, (now - when).days)
        except ValueError:
            pass
    for rx, fn in _RECENCY_NOTE_DAYS:
        m = rx.search(signal.recency_note or "")
        if m:
            try:
                return max(0, int(fn(m)))
            except Exception:
                continue
    return None


def recency_weight(age_days: Optional[int]) -> float:
    """How much a signal still counts, by age.

    Undated sits at 0.7 — deliberately above the 0.55 a dated 6-12 month signal
    gets, and below the 0.9+ of anything under 90 days. An undated signal might
    be from last week; discounting it to half would penalise the company for our
    extractor failing to find a date, which is the same "missing evidence read as
    bad evidence" mistake `confidence` exists to avoid.
    """
    if age_days is None:
        return 0.7
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.9
    if age_days <= 180:
        return 0.75
    if age_days <= 365:
        return 0.55
    if age_days <= 730:
        return 0.35
    return 0.2


def describe_age(age_days: Optional[int]) -> str:
    """Human phrasing for a signal's age, for the reasoning column."""
    if age_days is None:
        return "undated"
    if age_days <= 1:
        return "today"
    if age_days <= 30:
        return f"{age_days} days old"
    if age_days <= 75:
        return f"{max(1, round(age_days / 7))} weeks old"
    if age_days <= 400:
        return f"{max(1, round(age_days / 30))} months old"
    return f"{age_days / 365:.1f} years old"


# --------------------------------------------------------------------------
# Relevance gate
#
# Live search sometimes returns pages with nothing to do with the target. A run
# for "Linear" searched `Linear issue tracker software company`, the word
# "issue" hijacked the results, and the published record cited J.K. Rowling's
# website, Jira's REST docs and a Pixel support thread. A reviewer clicking a
# citation and landing there is a credibility event; a missing citation is
# invisible. So this gate is deliberately biased toward dropping.
#
# The test is content, not URL: a legitimate funding article lives at
# techcrunch.com and may not carry the company name in its path. The bar is low
# - one clear branded mention is enough - because this is a junk filter, not a
# precision instrument.
# --------------------------------------------------------------------------

#: Words that make a nearby name-mention look like it is about a company.
_COMPANY_CONTEXT = (
    "inc", "inc.", "llc", "ltd", "corp", "company", "startup", "platform",
    "software", "saas", "founded", "founder", "co-founder", "ceo", "raises",
    "raised", "funding", "series", "valuation", "customers", "product",
    "launch", "hiring", "team", "app", "acquires", "acquired", "headquarters",
    "teams", "customers", "products", "founders", "employees", "pricing",
    "integrations", "enterprise", "workspace", "dashboard", "subscription",
    "roles", "job", "jobs", "careers", "openings", "recruiting", "board",
)


def _host_of(url: str) -> str:
    return normalize_domain(url) or ""


def _is_own_domain(url: str, domain: Optional[str]) -> bool:
    """Is this URL on the company's own site (including any subdomain)?

    `lps.linear.app`, `orbit.linear.app` and `linear.app/brand` are all genuine
    even when the page body is thin, so they never need to prove themselves.
    """
    if not domain:
        return False
    host, own = _host_of(url), normalize_domain(domain)
    if not host or not own:
        return False
    return host == own or host.endswith("." + own) or own.endswith("." + host)


def _name_variants(*names: Optional[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        n = (n or "").strip()
        if len(n) < 2:
            continue
        if n not in out:
            out.append(n)
        # "Acme Inc." also matches as "Acme"
        head = re.split(r"[,(]| - ", n)[0].strip()
        head = re.sub(r"\b(inc|llc|ltd|corp|corporation|co|gmbh|plc)\.?$", "", head,
                      flags=re.I).strip()
        if len(head) >= 2 and head not in out:
            out.append(head)
    return out


#: Tokens that make a preceding name read as the SUBJECT of a sentence - the
#: company doing something - rather than an adjective modifying a noun.
_SUBJECT_FOLLOWERS = {
    "is", "was", "are", "were", "has", "have", "had", "will", "can", "does",
    "did", "announced", "announces", "raised", "raises", "launched", "launches",
    "said", "says", "offers", "provides", "builds", "helps", "acquired",
    "acquires", "closed", "secured", "plans", "adds", "ships", "integrates",
    "supports", "lets", "makes", "uses", "joined", "hired", "expanded",
    "became", "remains", "continues", "reported", "confirmed", "unveiled",
    "introduced", "partnered", "teamed", "sells", "serves", "powers", "enables",
    "today", "now", "also", "recently", "just", "and", "or", "which", "that",
}

#: Tokens that, following the name, mark it as a company even when capitalised.
_COMPANY_SUFFIXES = {
    "inc", "inc.", "corp", "corp.", "llc", "ltd", "ltd.", "co", "co.", "gmbh",
    "plc", "labs", "software", "technologies", "technology", "systems",
    "solutions", "group", "holdings", "ventures", "ai", "io", "app", "platform",
    "hq", "team",
}

_SENTENCE_ENDERS = set(".!?\n\r\u2022\u2013\u2014|*-:;\u2026")


def _mention_strength(text: str, match: "re.Match[str]") -> str:
    """How much does this occurrence look like a brand rather than a word?

    Sentence-initial capitalisation is grammar, not evidence: "Linear algebra is
    the branch of mathematics..." capitalises Linear for the same reason every
    sentence capitalises its first word. Wikipedia, docs and academic pages do
    this constantly, and those are exactly what ranks for a bare common-word
    query. So position and the right-hand neighbour decide, not case alone.
    """
    before = text[: match.start()].rstrip()
    sentence_initial = not before or before[-1] in _SENTENCE_ENDERS

    after = text[match.end():].lstrip()
    if after[:2] in ("'s", "\u2019s") or after[:1] in (",", ";"):
        return "strong"                      # possessive or appositive: a subject
    follower = re.match(r"[A-Za-z][A-Za-z.'\u2019-]*", after)
    word = follower.group(0) if follower else ""
    low = word.lower().rstrip(".,;:")

    if low in _SUBJECT_FOLLOWERS or low in _COMPANY_SUFFIXES:
        return "strong"
    if word[:1].isupper():
        # "Linear Algebra", "Introduction to Linear Regression" - part of a
        # larger title-cased phrase, which is how the adjective reads.
        return "weak"
    if sentence_initial:
        # Capitalised only because a sentence started. Needs corroboration.
        return "weak"
    return "strong"                          # mid-sentence capital: a proper noun


def _mentions(text: str, name: str) -> list[str]:
    """Strength of every standalone, capitalised occurrence of `name`."""
    if not name:
        return []
    pattern = r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])"
    return [_mention_strength(text or "", m) for m in re.finditer(pattern, text or "")]


def _branded_mentions(text: str, name: str) -> int:
    """Occurrences that actually read as the brand. Kept for callers/tests."""
    return sum(1 for st in _mentions(text, name) if st == "strong")


def _is_ats_board_for(url: str, names: list[str]) -> bool:
    """A known applicant-tracking host whose path carries the company slug."""
    low = (url or "").lower()
    if not any(fragment in low for fragment, _n in _ATS_HOST_NAMES):
        return False
    path = low.split("://", 1)[-1]
    path = path[path.find("/"):] if "/" in path else ""
    for n in names:
        slug = _SLUG_RE.sub("", n.lower())
        if len(slug) >= 3 and slug in _SLUG_RE.sub("", path):
            return True
    return False


def is_relevant_source(
    url: str,
    text: str,
    target: str,
    company_name: Optional[str] = None,
    domain: Optional[str] = None,
) -> bool:
    """May this page be cited as a source about this company?"""
    if not url:
        return False
    if _is_own_domain(url, domain):
        return True

    body = text or ""
    names = _name_variants(company_name, target)

    # A public ATS board is hosted on the ATS's domain, not the company's, but
    # `fetch_jobs` resolved it FOR this company and the slug is in the path.
    # jobs.ashbyhq.com/vanta is as genuine a source as vanta.com/careers.
    if _is_ats_board_for(url, names):
        return True

    # The company's domain named anywhere in the page or the URL is decisive.
    own = normalize_domain(domain) if domain else None
    haystack_url = (url or "").lower()
    if own and (own in body.lower() or own in haystack_url):
        return True

    # A name in the URL path is NOT sufficient on its own: "wikipedia.org/wiki/
    # Linear_algebra" and "traffic.gov/ramp" both carry the slug while having
    # nothing to do with the company. It only decides when the body is too
    # short to judge, which covers a thin fetch of a genuinely relevant page.
    if len(body.strip()) < 40:
        for n in names:
            slug = _SLUG_RE.sub("-", n.lower()).strip("-")
            if len(slug) >= 4 and slug in haystack_url:
                return True

    for n in names:
        strengths = _mentions(body, n)
        if any(st == "strong" for st in strengths):
            return True
        if strengths:
            # Only weak (sentence-initial or title-cased) matches. These need
            # corroboration from the surrounding prose, which is what separates
            # "Linear customers include..." from "Linear algebra is...".
            pattern = r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])"
            for m in re.finditer(pattern, body):
                window = body[max(0, m.start() - 90): m.end() + 90].lower()
                padded = f" {re.sub(r'[^a-z0-9]+', ' ', window).strip()} "
                if any(f" {w} " in padded for w in _COMPANY_CONTEXT):
                    return True
    return False


def relevant_evidence(
    evidence: list[tuple[str, str]],
    target: str,
    company_name: Optional[str] = None,
    domain: Optional[str] = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Split evidence into what may be cited and the URLs that may not."""
    keep: list[tuple[str, str]] = []
    rejected: list[str] = []
    for url, text in evidence or []:
        if is_relevant_source(url, text, target, company_name, domain):
            keep.append((url, text))
        elif url and url not in rejected:
            rejected.append(url)
    return keep, rejected


# --------------------------------------------------------------------------
# Provenance verification
#
# Normalization checks that a value is well FORMED. This checks that somebody
# actually SAID it. A live model returned employee_range="201-500" for a company
# whose evidence contains no headcount anywhere, wrote reasoning claiming it was
# "stated", and the pipeline published it with an empty `unknowns` list.
#
# Prompting a model not to invent is a hope. Checking is a guarantee, and the
# guarantee is the product. Every scalar below must be traceable to a string in
# the gathered evidence or it is dropped and recorded as unknown.
# --------------------------------------------------------------------------

VERIFIED_FIELDS = (
    "employee_range", "employee_estimate", "hq_city", "hq_country", "founded_year",
)

_COUNTRY_ALIASES = {
    "united states": {"united states", "usa", "u.s.", "u.s.a.", "us", "america"},
    "usa": {"united states", "usa", "u.s.", "u.s.a.", "us", "america"},
    "united kingdom": {"united kingdom", "uk", "u.k.", "britain", "england", "scotland"},
    "uk": {"united kingdom", "uk", "u.k.", "britain", "england", "scotland"},
    "netherlands": {"netherlands", "holland", "the netherlands"},
    "germany": {"germany", "deutschland"},
}


def _blob(evidence: list[tuple[str, str]]) -> str:
    return " \n ".join((t or "") for _u, t in (evidence or [])).lower()


def _verify_year(year: Optional[int], blob: str) -> bool:
    return bool(year) and str(year) in blob


def _verify_place(name: Optional[str], blob: str) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    if not n:
        return False
    if n in blob:
        return True
    for alias_set in (_COUNTRY_ALIASES.get(n),):
        if alias_set and any(a in blob for a in alias_set):
            return True
    return False


def _verify_employee_range(bucket: Optional[str], blob: str) -> bool:
    """The band itself, or any headcount that falls inside it, must appear."""
    if not bucket:
        return False
    # Verification runs before normalization, so the incoming value may still be
    # raw ("~250 employees"). Canonicalise first, then check provenance.
    b = normalize_employee_range(bucket) or ""
    if not b or _BUCKET_MIDPOINTS.get(b) is None:
        return False
    bounds = next(((lo, hi) for name, lo, hi in _BUCKET_BOUNDS if name == b), None)
    if bounds is None:
        return False
    lo, hi = bounds

    # the band written out, in the ways people write it
    digits = re.findall(r"\d+", b)
    if len(digits) >= 2:
        a, z = digits[0], digits[1]
        for sep in ("-", "\u2013", "\u2014", " to ", " - ", "-"):
            if f"{a}{sep}{z}".lower() in blob:
                return True
        if re.search(rf"\b{a}\s*(?:-|\u2013|to)\s*{z}\b", blob):
            return True
    elif b.endswith("+") and re.search(rf"\b{digits[0]}\s*\+", blob):
        return True

    # or a concrete headcount inside the band, stated as a headcount
    for m in re.finditer(
        r"(\d[\d,]{0,8})\s*(\+?)\s*(?:employees|people|staff|team members|headcount)", blob
    ):
        try:
            n = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if m.group(2) == "+":
            # "500+ employees" means AT LEAST 500, so a band that tops out at
            # 500 is not supported by it - only a band reaching beyond is.
            if lo >= n or hi > n:
                return True
            continue
        if lo <= n <= hi:
            return True
    for m in re.finditer(r"(?:team|company|workforce)\s+of\s+(?:about\s+|around\s+|~)?(\d[\d,]{0,8})", blob):
        try:
            n = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if lo <= n <= hi:
            return True
    return False


def _verify_headcount_number(n: Optional[int], blob: str) -> bool:
    if not n:
        return False
    return bool(re.search(rf"\b{n:,}\b".replace(",", "[,]?"), blob)) or str(n) in blob


def _signal_is_supported(sig: Signal, blob: str) -> bool:
    """Is this signal's headline actually rooted in the evidence text?

    Token overlap rather than exact match: extractors legitimately rephrase, but
    a wholly invented event shares almost nothing with the source.
    """
    words = {w for w in re.findall(r"[a-z0-9$][a-z0-9$.%-]{2,}", (sig.headline or "").lower())}
    words -= {
        "the", "and", "for", "with", "its", "has", "have", "from", "that", "this",
        "their", "our", "new", "now", "are", "was", "will", "company", "announced",
    }
    if not words:
        return False
    hits = sum(1 for w in words if w in blob)
    return (hits / len(words)) >= 0.6


def verify_record(
    rec: CompanyRecord, evidence: list[tuple[str, str]]
) -> tuple[CompanyRecord, list[str]]:
    """Drop every scalar the evidence does not actually support.

    Returns the cleaned record and the list of dropped field names. Signals that
    cannot be traced to the evidence are dropped too - an invented funding round
    is the single most damaging thing this tool could emit.
    """
    out = rec.model_copy(deep=True)
    blob = _blob(evidence)
    dropped: list[str] = []
    if not blob.strip():
        # No evidence text at all: nothing here can be verified, so nothing is
        # claimed. This is the honest reading, not a degenerate one.
        for f in VERIFIED_FIELDS:
            if _is_set(getattr(out, f, None)):
                setattr(out, f, None)
                dropped.append(f)
        if out.signals:
            dropped.append("signals")
            out.signals = []
        return out, dropped

    if out.employee_range and not _verify_employee_range(out.employee_range, blob):
        dropped.append("employee_range")
        out.employee_range = None

    # An estimate is only as good as its source: either the exact number appears,
    # or it is re-derived from a band that did. Never the model's own arithmetic
    # (which returned 500 as the midpoint of 201-500).
    if out.employee_estimate is not None:
        if not _verify_headcount_number(out.employee_estimate, blob):
            derived = bucket_midpoint(out.employee_range)
            if derived is None:
                dropped.append("employee_estimate")
            out.employee_estimate = derived

    if out.hq_city and not _verify_place(out.hq_city, blob):
        dropped.append("hq_city")
        out.hq_city = None
    if out.hq_country and not _verify_place(out.hq_country, blob):
        dropped.append("hq_country")
        out.hq_country = None
    if out.founded_year and not _verify_year(out.founded_year, blob):
        dropped.append("founded_year")
        out.founded_year = None

    kept = [s for s in (out.signals or []) if _signal_is_supported(s, blob)]
    if len(kept) != len(out.signals or []):
        dropped.append("unsupported_signals")
    out.signals = kept
    return out, dropped


# --------------------------------------------------------------------------
# Funding re-classification
#
# Whoever produced the record - the stub's keyword classifier or a real model -
# can file a capital event under the wrong kind. Real headlines rarely say
# "raised": the genuine Ramp article reads "nearly doubles valuation to $32B in
# secondary share sale", which has no raise verb at all and was landing as a
# `product` signal. Since this module is the last pass before the record is
# rendered, correcting the kind belongs here rather than in any one producer.
# --------------------------------------------------------------------------

_FUND_AMOUNT_RE = re.compile(
    r"\$\s?\d[\d,.]*\s*(?:[MBK]\b|million|billion|bn\b)", re.I)
_FUND_ROUND_RE = re.compile(
    r"\b(?:pre-seed|seed|series\s+[a-j]|bridge|growth|mezzanine)\s*(?:round|financing)?\b", re.I)
_FUND_SECONDARY_RE = re.compile(
    r"\b(secondary\s+(?:share\s+)?sale|share\s+sale|tender\s+offer|"
    r"employee\s+tender|secondary\s+transaction|secondary\s+market\s+sale)\b", re.I)
_FUND_VALUATION_RE = re.compile(
    r"\b(valuation|valued\s+at|post-money|pre-money|worth)\b", re.I)
_FUND_VERB_RE = re.compile(
    r"\b(rais(?:e|es|ed|ing)|clos(?:e|es|ed|ing)|land(?:s|ed)?|secur(?:e|es|ed)|"
    r"bank(?:s|ed)?|net(?:s|ted)?|pull(?:s|ed)?\s+in|pick(?:s|ed)?\s+up|"
    r"scoop(?:s|ed)?\s+up|nab(?:s|bed)?|receiv(?:e|es|ed))\b", re.I)
_FUND_INVESTOR_RE = re.compile(
    r"\b(led\s+by|co-led\s+by|backed\s+by|investment\s+from|participation\s+from|"
    r"funding\s+from|investors?\s+includ)", re.I)
_FUND_WORD_RE = re.compile(r"\b(funding|financing|investment|round|capital)\b", re.I)

#: Marker appended to `detail`, and read back by the scorer and hook writer.
SECONDARY_NOTE = (
    "Secondary share sale: existing shareholders selling, so this is a liquidity "
    "and valuation event rather than new primary capital for the company."
)


def classify_funding(text: str) -> Optional[str]:
    """Is this text a capital event, and of what sort?

    Returns "secondary", "valuation" or "primary", else None. Requires a money
    amount or a named round for every path, so "we shipped three products" and
    "$13B market opportunity" do not trip it.
    """
    t = text or ""
    # "tender offer" / "secondary share sale" are unambiguous on their own, so
    # they are checked before the amount gate.
    if _FUND_SECONDARY_RE.search(t):
        return "secondary"

    has_amount = bool(_FUND_AMOUNT_RE.search(t))
    has_round = bool(_FUND_ROUND_RE.search(t))
    if not (has_amount or has_round):
        return None
    if _FUND_VALUATION_RE.search(t) and has_amount:
        # "nearly doubles its valuation to $32B" - no raise verb anywhere.
        return "valuation"
    if _FUND_VERB_RE.search(t) and (has_amount or has_round) and (
        _FUND_WORD_RE.search(t) or has_round or _FUND_INVESTOR_RE.search(t)
    ):
        return "primary"
    if _FUND_INVESTOR_RE.search(t):
        return "primary"
    if has_round and _FUND_WORD_RE.search(t):
        return "primary"
    return None


def reclassify_signals(signals: list[Signal]) -> list[Signal]:
    """Promote misfiled capital events to `funding`, recording the distinction.

    A secondary sale stays `kind="funding"` - it belongs in the same bucket for
    scoring and it is unambiguously a momentum signal - but `detail` says plainly
    that no new primary capital changed hands, so neither the scorer nor a hook
    can imply the company just banked the money.
    """
    out: list[Signal] = []
    for sig in signals or []:
        if sig is None:
            continue
        kind = classify_funding(f"{sig.headline or ''} {sig.detail or ''}")
        if kind is None or sig.kind in ("funding", "hiring"):
            # Never reclassify hiring: a job board mentioning compensation
            # bands is not a funding round.
            if kind == "secondary" and sig.kind == "funding" and SECONDARY_NOTE not in (sig.detail or ""):
                sig = sig.model_copy(deep=True)
                sig.detail = ((sig.detail or "").strip() + " " + SECONDARY_NOTE).strip()
            out.append(sig)
            continue
        fixed = sig.model_copy(deep=True)
        fixed.kind = "funding"
        note = SECONDARY_NOTE if kind == "secondary" else ""
        if note and note not in (fixed.detail or ""):
            fixed.detail = ((fixed.detail or "").strip() + " " + note).strip()
        out.append(fixed)
    return out


def is_secondary_funding(signal: Signal) -> bool:
    """Was this capital event a secondary sale rather than a primary raise?"""
    blob = f"{signal.headline or ''} {signal.detail or ''}"
    return SECONDARY_NOTE in blob or bool(_FUND_SECONDARY_RE.search(blob))


def _tech_tokens(name: str) -> list[str]:
    """'Google Analytics 4' -> ['google','analytics','4']; 'Next.js' -> ['next','js']."""
    return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]


def _merge_evidence(*values: str) -> str:
    """Union of the evidence strings, in order, trimmed so the cell stays readable."""
    out: list[str] = []
    for v in values:
        v = (v or "").strip()
        if v and v not in out:
            out.append(v)
    joined = "; ".join(out)
    return joined if len(joined) <= 300 else joined[:297] + "..."


def _absorb(winner: TechStack, loser: TechStack) -> TechStack:
    """Fold `loser` into `winner`, keeping the better confidence and both reasons."""
    merged = winner.model_copy(deep=True)
    merged.confidence = max(winner.confidence or 0.0, loser.confidence or 0.0)
    merged.evidence = _merge_evidence(winner.evidence, loser.evidence)
    return merged


def _dedupe_tech(items: Iterable[TechStack]) -> list[TechStack]:
    """One row per product.

    Keyed on NAME, not (category, name), in two passes:

      1. Same name, different category -> collapse. Cloudflare detected as both
         `hosting` and `infra` is one Cloudflare; the category is a label we
         attached, not a second fact about the company. The higher-confidence
         detection wins and names the category, and both evidence strings survive.
      2. A generic name that is a token-prefix of a more specific one -> keep the
         specific. `Google Analytics` and `Google Analytics 4` are one product
         detected at two resolutions, and GA4 is the real answer.

    We collapse across categories even for a genuinely multi-category vendor like
    HubSpot (crm + marketing) rather than special-casing it. The record answers
    "what does this company run", and a reviewer seeing the same vendor on two
    rows reads a data-quality bug, not nuance — one rule applied everywhere is
    easier to trust than a rule with exceptions. The cost is real but small: the
    secondary category is dropped, while its evidence string is kept.
    """
    # Pass 1 - collapse by name, remembering first-seen order.
    best: dict[str, TechStack] = {}
    order: list[str] = []
    for t in items or []:
        if not t or not getattr(t, "name", "").strip():
            continue
        key = t.name.strip().lower()
        if key not in best:
            best[key] = t
            order.append(key)
            continue
        incumbent = best[key]
        # Whichever detection is more confident keeps its category.
        if (t.confidence or 0) > (incumbent.confidence or 0):
            best[key] = _absorb(t, incumbent)
        else:
            best[key] = _absorb(incumbent, t)

    # Pass 2 - drop a generic name when a more specific one is also present.
    tokens = {k: _tech_tokens(best[k].name) for k in order}
    dropped: set[str] = set()
    for a in order:
        if a in dropped:
            continue
        for b in order:
            if a == b or b in dropped:
                continue
            ta, tb = tokens[a], tokens[b]
            if ta and len(ta) < len(tb) and tb[: len(ta)] == ta:
                # `a` is the generic form of `b`: fold it in and keep `b`.
                best[b] = _absorb(best[b], best[a])
                dropped.add(a)
                break

    return [best[k] for k in order if k not in dropped]


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

    rec.signals = _sort_signals(_dedupe_signals(reclassify_signals(rec.signals)))
    rec.tech_stack = _dedupe_tech(rec.tech_stack)

    if evidence_urls is not None:
        rec.sources = _unique_urls(evidence_urls)
    else:
        rec.sources = _unique_urls(rec.sources)

    rec.unknowns = _compute_unknowns(rec)
    return rec


def apply_relevance_gate(
    rec: CompanyRecord, evidence: list[tuple[str, str]], target: str
) -> CompanyRecord:
    """Strip citations that point at pages unrelated to this company.

    Applies to `sources` and to each `Signal.source_url`. A signal whose only
    source is dropped keeps its claim but loses the link, so it flows into the
    uncited path the scorer already handles rather than carrying a broken one.
    """
    out = rec.model_copy(deep=True)
    _keep, rejected = relevant_evidence(evidence, target, out.company_name, out.domain)
    if not rejected:
        return out

    bad = {r for r in rejected}
    bad_keys = {_url_compare_key(b) for b in bad}

    out.sources = [u for u in (out.sources or []) if _url_compare_key(u) not in bad_keys]
    for sig in out.signals or []:
        if sig.source_url and _url_compare_key(sig.source_url) in bad_keys:
            sig.source_url = None
    return out


def _url_compare_key(url: str) -> str:
    u = (url or "").strip().lower()
    u = re.sub(r"^https?://", "", u).split("#", 1)[0]
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


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

    # A public job board is the one hard hiring number we can get. Fold it in as
    # a proper signal before normalization so dedupe and sorting see it, and so
    # it lands in `sources` like any other cited fact. It never becomes an
    # employee_range - see the honesty note on the parser.
    jobs = parse_jobs_evidence(evidence)
    if jobs is not None and jobs.open_roles > 0:
        # The counted board is the best hiring evidence we have, so redundant
        # hiring signals go rather than sitting beside it: both the vague
        # "See open positions" scraped off a careers page, and the model's own
        # restatement of the same total ("Vanta has 93 open roles on Ashby").
        total = str(jobs.open_roles)

        def redundant(sig: Signal) -> bool:
            if sig.kind != "hiring":
                return False
            blob = f"{sig.headline or ''} {sig.detail or ''}"
            if total in re.findall(r"\d+", blob):
                return True                      # restates the same count
            return not any(ch.isdigit() for ch in sig.headline or "")

        rec.signals = [jobs_signal(jobs)] + [
            s for s in (rec.signals or []) if not redundant(s)
        ]

    # Only evidence we actually read is citable — a robots-blocked or 404'd fetch
    # still appears in the list and must not end up in `sources`.
    evidence_urls = citable_urls(evidence)

    # Provenance gate, before normalization so `unknowns` is derived from what
    # survived it. A value the evidence does not support is not a value.
    rec, _dropped = verify_record(rec, evidence)

    rec = normalize_record(rec, target, evidence_urls=evidence_urls)

    # Relevance gate, after normalization because it needs the resolved domain
    # and company name. A citation pointing at an unrelated page is worse than
    # no citation, so anything that cannot be tied to this company is dropped
    # and whatever rested on it becomes uncited.
    rec = apply_relevance_gate(rec, evidence, target)

    if base is not None:
        # base wins for anything it confidently holds.
        rec = merge_records(base, rec)
        rec.sources = _unique_urls(list(base.sources) + evidence_urls)
        rec.unknowns = _compute_unknowns(rec)

    return rec
