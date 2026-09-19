"""StubClient — a deterministic, no-network brain.

This is the reason the CLI runs with zero API keys, and it is a first-class
deliverable: it produces a *plausible research trajectory* and a *genuinely
extracted* record, not a canned answer.

`decide` reconstructs what has already happened from the rendered context (the
running trace the agent loop passes in) plus a little instance-local memory, and
applies a fixed research policy:

    nothing yet                -> web_search "<company> company overview"
    searches failing           -> retry with a different query, then go straight
                                  at the obvious official domain (never give up
                                  just because a search provider blocked us)
    have results, no homepage  -> fetch_page the most likely official domain
    homepage fetched           -> detect_tech on it
    firmographics in hand      -> web_search "<company> funding OR hiring OR news"
    news/careers URL unread    -> fetch_page the most promising one
    otherwise                  -> finish

`extract` parses the evidence blocks back out of the rendered context so every
fact it recovers keeps the URL it came from, then runs real heuristics over them:
funding / hiring / product / leadership signals with dates, employee counts, HQ,
founding year, tech stack. It stays generic over any pydantic model (it walks
`model_fields`), never raises, and leaves anything it cannot establish as None —
honest unknowns are the point.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Optional, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from lea.config import Settings
from lea.llm.anthropic_client import best_effort_instance
from lea.llm.base import LLMUsage
from lea.models import AgentAction, ToolName

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Hosts that are never the company's own site. A record whose `domain` is an
# aggregator is worse than a record with no domain at all.
AGGREGATORS = frozenset(
    {
        "linkedin.com", "crunchbase.com", "wikipedia.org", "wikimedia.org", "bloomberg.com",
        "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com", "tiktok.com",
        "glassdoor.com", "indeed.com", "zoominfo.com", "pitchbook.com", "owler.com", "dnb.com",
        "tracxn.com", "similarweb.com", "builtwith.com", "cbinsights.com", "6sense.com",
        "apollo.io", "rocketreach.co", "leadiq.com", "g2.com", "capterra.com", "trustradius.com",
        "wellfound.com", "angel.co", "reddit.com", "medium.com", "substack.com", "github.com",
        "techcrunch.com", "forbes.com", "reuters.com", "businesswire.com", "prnewswire.com",
        "bizjournals.com", "theinformation.com", "axios.com", "cnbc.com", "ycombinator.com",
        "producthunt.com", "stackshare.io", "getlatka.com", "growjo.com", "lusha.com",
    }
)

# Applicant-tracking hosts: their presence is a hiring signal in itself.
_ATS_HOSTS = ("greenhouse.io", "lever.co", "ashbyhq.com", "workday", "smartrecruiters.com", "teamtailor.com")


def _ats_host_hints() -> tuple[str, ...]:
    """Agent A owns the canonical list; import it lazily so we cannot drift."""
    try:
        from lea.tools.jobs import ATS_HOST_HINTS  # type: ignore

        return tuple(ATS_HOST_HINTS)
    except Exception as exc:  # noqa: BLE001 - a missing tool must not break the brain
        logger.debug("Falling back to the local ATS host list: %s", exc)
        return ("ashbyhq.com", "greenhouse.io", "lever.co", "workable.com")


def _ats_mentioned(context: str) -> bool:
    """Agent A's detect_tech names the board and the tool in its summary."""
    lowered = (context or "").lower()
    if "fetch_jobs" in lowered or "ats_boards" in lowered:
        return True
    return any(hint in lowered for hint in _ats_host_hints()) or any(h in lowered for h in _ATS_HOSTS)


def _domain_of(fetched: set[str], tokens: list[str]) -> Optional[str]:
    """The company's own host, from a page we already opened."""
    for url in sorted(fetched):
        if _score_official(url, tokens) > 0:
            try:
                host = _registrable(urlparse(url).netloc)
            except ValueError:  # pragma: no cover - defensive
                continue
            if host:
                return host
    return None


def _is_ats_url(url: str) -> bool:
    """True for a job-board URL, which `fetch_jobs` reads far better than `fetch_page`."""
    lowered = (url or "").lower()
    return any(hint in lowered for hint in _ats_host_hints())

# URL path/keyword hints for "the interesting second fetch".
_PROMISING = (
    ("careers", 6), ("jobs", 6), ("/about", 5), ("about-us", 5), ("news", 4), ("press", 4),
    ("blog", 3), ("funding", 5), ("newsroom", 4), ("company", 3), ("team", 2), ("investors", 4),
)

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")
_TARGET_RE = re.compile(r"^\s*TARGET(?: COMPANY)?:\s*(.+?)\s*$", re.MULTILINE)
_STEPS_LEFT_RE = re.compile(r"steps\s*=\s*(\d+)", re.IGNORECASE)
_SEARCHES_LEFT_RE = re.compile(r"searches\s*=\s*(\d+)", re.IGNORECASE)
_FETCHES_LEFT_RE = re.compile(r"fetches\s*=\s*(\d+)", re.IGNORECASE)
_JOBS_LEFT_RE = re.compile(r"job_boards\s*=\s*(\d+)", re.IGNORECASE)
_FAILED_STEP_RE = re.compile(r"^\s*\d+\.\s*([a-z_]+)\s*\([^\n]*?\)\s*->\s*FAILED", re.MULTILINE)

_NAME_NOISE = frozenset({"inc", "llc", "ltd", "corp", "the", "com", "www", "gmbh", "sas", "plc"})


def _tool_call_re(tool: str) -> re.Pattern[str]:
    return re.compile(rf"\b{tool}\s*\(", re.IGNORECASE)


# ---------------------------------------------------------------------------
# context inspection (shared by decide and extract)
# ---------------------------------------------------------------------------


def _target_from_context(context: str) -> str:
    m = _TARGET_RE.search(context or "")
    if m:
        return m.group(1).strip()
    first = (context or "").strip().splitlines()
    return first[0].strip() if first else "the company"


def _company_tokens(target: str) -> list[str]:
    """Lowercase alphanumeric tokens of the company name, longest first."""
    base = (target or "").strip()
    if "://" in base:
        base = urlparse(base).netloc or base
    if " " not in base and "." in base:
        base = _registrable(base.split("/")[0]).split(".")[0]
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", base.lower()) if len(t) > 2 and t not in _NAME_NOISE]
    return sorted(set(tokens), key=len, reverse=True)


def _slug(target: str) -> str:
    """Company name -> the domain label you would guess. "Acme Robotics" -> "acmerobotics"."""
    base = (target or "").strip()
    if "://" in base:
        base = urlparse(base).netloc or base
    if " " not in base and "." in base:
        return _registrable(base.split("/")[0]).split(".")[0]
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", base.lower()) if p and p not in _NAME_NOISE]
    return "".join(parts)


def _registrable(netloc: str) -> str:
    host = (netloc or "").lower().split(":")[0].strip("/")
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in host.split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _is_aggregator(host: str) -> bool:
    reg = _registrable(host)
    return reg in AGGREGATORS or any(reg.endswith(a) or a in reg for a in _ATS_HOSTS)


def _urls_in(context: str) -> list[str]:
    seen: list[str] = []
    for raw in _URL_RE.findall(context or ""):
        url = raw.rstrip(".,;:'\")")
        if url not in seen:
            seen.append(url)
    return seen


def _fetched_urls(context: str) -> set[str]:
    """URLs that already appear as arguments to fetch_page / detect_tech."""
    out: set[str] = set()
    for line in (context or "").splitlines():
        if "fetch_page(" not in line and "detect_tech(" not in line:
            continue
        for url in _URL_RE.findall(line):
            out.add(url.rstrip(".,;:'\")"))
    return out


def _tools_used(context: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tool in ("web_search", "fetch_page", "detect_tech", "fetch_jobs", "finish"):
        counts[tool] = len(_tool_call_re(tool).findall(context or ""))
    return counts


def _tools_failed(context: str) -> dict[str, int]:
    """Steps the trace renders as `N. tool(args) -> FAILED: ...`."""
    counts: dict[str, int] = {}
    for tool in _FAILED_STEP_RE.findall(context or ""):
        counts[tool] = counts.get(tool, 0) + 1
    return counts


def _int_after(pattern: re.Pattern[str], context: str) -> Optional[int]:
    m = pattern.search(context or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None


def _score_official(url: str, tokens: list[str]) -> float:
    """How likely is this URL the company's own homepage?"""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        logger.debug("Unparseable URL %r: %s", url, exc)
        return -100.0
    host = _registrable(parsed.netloc)
    if not host:
        return -100.0
    score = 0.0
    if _is_aggregator(host):
        score -= 50.0
    bare = host.split(".")[0]
    for tok in tokens:
        if tok == bare:
            score += 20.0
            break
        if tok in bare:
            score += 12.0
            break
        if tok in parsed.netloc.lower():
            score += 6.0
            break
    depth = len([p for p in parsed.path.split("/") if p])
    score -= depth * 3.0  # homepages win over deep links
    if parsed.path in ("", "/"):
        score += 4.0
    return score


def _score_promising(url: str, tokens: list[str]) -> float:
    """How likely is this URL to carry a signal (news / hiring / about)?"""
    lowered = url.lower()
    score = 0.0
    for needle, weight in _PROMISING:
        if needle in lowered:
            score += weight
    try:
        host = _registrable(urlparse(url).netloc)
    except ValueError:
        return score
    if _is_aggregator(host):
        score -= 2.0  # usable for news, just not preferred
    bare = host.split(".")[0]
    if any(tok in bare for tok in tokens):
        score += 3.0
    return score


# ---------------------------------------------------------------------------
# evidence blocks — keeps every fact attached to the URL it came from
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(
    r"^--- BLOCK (\d+) \| source_url:\s*(.*?)\s*---$\n(.*?)\n^--- END BLOCK \1 ---$",
    re.MULTILINE | re.DOTALL,
)
_FAILED_BLOCK_RE = re.compile(
    r"\b(fetch failed|failed to fetch|error fetching|http (?:4\d\d|5\d\d)|403 forbidden|"
    r"404 not found|timed out|timeout|blocked by robots|access denied|captcha)\b",
    re.IGNORECASE,
)


class _Evidence:
    """One evidence block: its source URL, its text, and whether it is usable."""

    __slots__ = ("url", "text", "ok")

    def __init__(self, url: str, text: str) -> None:
        self.url = (url or "").strip()
        self.text = text or ""
        body = self.text.strip()
        self.ok = bool(body) and body != "(empty)" and not _FAILED_BLOCK_RE.search(body[:400])

    @property
    def host(self) -> str:
        if not self.url:
            return ""
        try:
            return _registrable(urlparse(self.url).netloc)
        except ValueError:  # pragma: no cover - defensive
            return ""


def _evidence_blocks(context: str) -> list[_Evidence]:
    """Parse `build_extraction_context` output back into attributable blocks.

    Falls back to a single unattributed block so the extractor still works when
    it is handed raw text (an agent trace, a fixture, anything).
    """
    blocks = [_Evidence(url, body) for _, url, body in _BLOCK_RE.findall(context or "")]
    if blocks:
        return blocks
    return [_Evidence("", context or "")]


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"

_DATE_MDY_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b", re.I)
_DATE_DMY_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\.?,?\s+((?:19|20)\d{{2}})\b", re.I)
_DATE_ISO_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
_DATE_MY_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+((?:19|20)\d{{2}})\b", re.I)
_DATE_YEAR_RE = re.compile(r"\b(?:in|since|during|of)\s+((?:19|20)\d{2})\b|\b((?:20)\d{2})\b", re.I)


def _parse_date(text: str) -> tuple[Optional[str], Optional[date], str]:
    """Return ``(raw_as_published, date_or_None, precision)``.

    ``precision`` is "day", "month", "year" or "none". The raw string is kept
    verbatim because :class:`~lea.models.Signal` wants the date as published.
    """
    if not text:
        return None, None, "none"

    m = _DATE_ISO_RE.search(text)
    if m:
        return _safe_date(m.group(0), int(m.group(1)), int(m.group(2)), int(m.group(3)), "day")

    m = _DATE_MDY_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(1)[:3].lower())
        if month:
            return _safe_date(m.group(0), int(m.group(3)), month, int(m.group(2)), "day")

    m = _DATE_DMY_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(2)[:3].lower())
        if month:
            return _safe_date(m.group(0), int(m.group(3)), month, int(m.group(1)), "day")

    m = _DATE_MY_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(1)[:3].lower())
        if month:
            return _safe_date(m.group(0), int(m.group(2)), month, 1, "month")

    m = _DATE_YEAR_RE.search(text)
    if m:
        year = m.group(1) or m.group(2)
        if year:
            return _safe_date(year, int(year), 7, 1, "year")

    return None, None, "none"


def _safe_date(raw: str, year: int, month: int, day: int, precision: str) -> tuple[str, Optional[date], str]:
    try:
        return raw, date(year, month, day), precision
    except ValueError as exc:
        logger.debug("Discarding impossible date %r (%s)", raw, exc)
        return raw, None, "none"


def _recency_note(parsed: Optional[date], precision: str, today: Optional[date] = None) -> str:
    """A plain-language note on how fresh a dated item is."""
    if parsed is None:
        return "undated"
    today = today or date.today()
    if parsed > today:
        return "dated in the future — check the source"
    if precision == "year":
        years = today.year - parsed.year
        if years <= 0:
            return "this year (year only, month unknown)"
        if years == 1:
            return "last year (year only, month unknown)"
        return f"{years} years ago (year only)"
    days = (today - parsed).days
    for limit, note in ((30, "within 30 days"), (90, "within 90 days"), (180, "within 6 months"), (365, "within the last year")):
        if days <= limit:
            return note
    return "over a year ago" if days <= 730 else "over two years ago"


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"[$€£]\s?(\d[\d,]*(?:\.\d+)?)\s*(b(?:illion|n)?|m(?:illion|m)?|k|thousand)?\b", re.I)
_ROUND_RE = re.compile(r"\b(pre-seed|seed|series\s+[A-J])\b(?:\s+(?:round|funding))?", re.I)
_VALUATION_RE = re.compile(r"(?:at|to)\s+an?\s+([$€£]\s?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|b|m)\b)\s*(?:post-|pre-)?(?:money\s*)?valuation", re.I)

_FUND_VERB_RE = re.compile(r"\b(raise[sd]?|raising|secur(?:e[sd]?|ing)|clos(?:e[sd]?|ing)|lands?|landed|nets?|netted|backs?|backed|led by|investment)\b", re.I)
_HIRING_RE = re.compile(
    r"\b(we(?:'|’)?re\s+hiring|we\s+are\s+hiring|now\s+hiring|open\s+roles?|open\s+positions?|"
    r"join\s+(?:our|the)\s+team|(\d+)\s+open\s+(?:roles?|positions?|jobs?)|view\s+open\s+roles?|"
    r"growing\s+(?:the|our)\s+team)\b",
    re.I,
)
_PRODUCT_RE = re.compile(r"\b(launch(?:e[sd]|ing)?|announc(?:e[sd]?|ing)|introduc(?:e[sd]?|ing)|unveil(?:s|ed|ing)?|now\s+available|general\s+availability|releases?|rolls?\s+out)\b", re.I)
_ACQUIRE_RE = re.compile(r"\b(acquires?|acquired|acquisition\s+of|merges?\s+with|merger\s+with)\b", re.I)
_PARTNER_RE = re.compile(r"\b(partners?\s+with|partnership\s+with|teams?\s+up\s+with|integrat(?:es|ion)\s+with)\b", re.I)
_LEADER_RE = re.compile(r"\b(appoints?|appointed|joins?\s+as|named\s+(?:its\s+)?(?:new\s+)?(?:CEO|CTO|CRO|CFO|COO|CMO|VP|President|Head)|hires?\s+(?:its\s+)?(?:new\s+)?(?:CEO|CTO|CRO|CFO|COO|CMO|VP)|steps?\s+down|promoted\s+to)\b", re.I)

#: Sentences shorter than this are page furniture, not claims.
_MIN_SENTENCE = 18
_MAX_HEADLINE = 170
_GENERIC_DETAIL = "Matched by keyword in the fetched page text; no figures attached."
_AMOUNT_IN_DETAIL_RE = re.compile(r"amount (\$[\d.,]+[BMK])", re.I)

_KIND_ORDER = {"funding": 0, "hiring": 1, "product": 2, "leadership": 3, "partnership": 4, "news": 5, "other": 6}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s\|\s|\s+…\s+|\s+\.\.\.\s+", text or "")
    return [" ".join(p.split()) for p in parts if p and len(p.strip()) >= _MIN_SENTENCE]


def _normalise_amount(raw_number: str, unit: Optional[str]) -> Optional[str]:
    """"500", "M" -> "$500M". Returns the display form, or None if nonsensical."""
    try:
        value = float(raw_number.replace(",", ""))
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None
    unit = (unit or "").lower()
    if unit.startswith("b"):
        return f"${_trim(value)}B"
    if unit.startswith("m"):
        return f"${_trim(value)}M"
    if unit.startswith("k") or unit.startswith("thousand"):
        return f"${_trim(value)}K"
    if value >= 1_000_000:
        return f"${_trim(value / 1_000_000)}M"
    return None  # a bare "$40" in prose is not a funding amount


def _trim(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"


def _classify(sentence: str) -> Optional[str]:
    """Which SignalKind, if any, does this sentence carry?"""
    has_amount = bool(_AMOUNT_RE.search(sentence))
    has_round = bool(_ROUND_RE.search(sentence))
    if (has_amount or has_round) and _FUND_VERB_RE.search(sentence):
        return "funding"
    if has_round and re.search(r"\b(announc|rais|fund|investor)", sentence, re.I):
        return "funding"
    if _HIRING_RE.search(sentence):
        return "hiring"
    if _LEADER_RE.search(sentence):
        return "leadership"
    if _ACQUIRE_RE.search(sentence):
        return "news"
    if _PARTNER_RE.search(sentence):
        return "partnership"
    if _PRODUCT_RE.search(sentence):
        return "product"
    return None


def _signal_detail(sentence: str, kind: str, raw_date: Optional[str]) -> str:
    bits: list[str] = []
    if kind == "funding":
        m = _AMOUNT_RE.search(sentence)
        if m:
            amount = _normalise_amount(m.group(1), m.group(2))
            if amount:
                bits.append(f"amount {amount}")
        m = _ROUND_RE.search(sentence)
        if m:
            bits.append(f"round {m.group(1).title()}")
        m = _VALUATION_RE.search(sentence)
        if m:
            bits.append(f"valuation {' '.join(m.group(1).split())}")
    if kind == "hiring":
        m = re.search(r"(\d+)\s+open\s+(?:roles?|positions?|jobs?)", sentence, re.I)
        if m:
            bits.append(f"{m.group(1)} open roles mentioned")
    if raw_date:
        bits.append(f"reported {raw_date}")
    if not bits:
        return _GENERIC_DETAIL
    joined = "; ".join(bits)
    return joined[0].upper() + joined[1:] + "."  # not .capitalize(): it lowercases the rest


def _headline_of(sentence: str) -> str:
    """Trim to the headline budget at a word boundary."""
    text = sentence.strip()
    if len(text) <= _MAX_HEADLINE:
        return text.rstrip(" ,;:-")
    cut = text[:_MAX_HEADLINE]
    space = cut.rfind(" ")
    if space > _MAX_HEADLINE * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def _dedupe_key(signal: dict[str, Any]) -> tuple[str, str]:
    """Two renderings of one event must collapse into the richest single signal."""
    kind = signal["kind"]
    if kind == "hiring":
        return ("hiring", "")  # one hiring signal per company is enough
    if kind == "funding":
        m = _AMOUNT_IN_DETAIL_RE.search(signal.get("detail") or "")
        if m:
            return ("funding", m.group(1).upper())
    return (kind, re.sub(r"[^a-z0-9]+", "", signal["headline"].lower())[:70])


def _richness(signal: dict[str, Any]) -> tuple[int, int, int]:
    detail = signal.get("detail") or ""
    return (0 if detail == _GENERIC_DETAIL else 1, len(detail), 1 if signal.get("date") else 0)


def extract_signals(blocks: list[_Evidence], today: Optional[date] = None, limit: int = 5) -> list[dict[str, Any]]:
    """Pull dated, attributed signals out of the evidence. Never invents one."""
    collected: dict[tuple[str, str], dict[str, Any]] = {}

    for block in blocks:
        if not block.ok:
            continue
        block_raw_date, block_date, block_prec = _parse_date(block.text)
        # Only a full publication date may stand in for a sentence's own date. A
        # bare year on the page is usually "Founded in 2019", and stamping that
        # onto a hiring sentence invents a fact.
        inheritable = block_prec in ("day", "month")
        for sentence in _sentences(block.text):
            kind = _classify(sentence)
            if kind is None:
                continue
            raw_date, parsed, precision = _parse_date(sentence)
            if parsed is None and inheritable:
                raw_date, parsed, precision = block_raw_date, block_date, block_prec
            signal = {
                "kind": kind,
                "headline": _headline_of(sentence),
                "detail": _signal_detail(sentence, kind, raw_date),
                "date": raw_date,
                "source_url": block.url or None,
                "recency_note": _recency_note(parsed, precision, today),
                "_sort": (_KIND_ORDER.get(kind, 9), 0 if parsed else 1),
            }
            key = _dedupe_key(signal)
            incumbent = collected.get(key)
            if incumbent is None or _richness(signal) > _richness(incumbent):
                collected[key] = signal

    found = list(collected.values())

    # A careers page or an ATS link is a hiring signal even with no matching prose.
    if not any(s["kind"] == "hiring" for s in found):
        for block in blocks:
            lowered = block.url.lower()
            if not block.url:
                continue
            is_careers = "/careers" in lowered or "/jobs" in lowered
            is_ats = any(a in lowered for a in _ATS_HOSTS)
            if is_careers or is_ats:
                found.append(
                    {
                        "kind": "hiring",
                        "headline": "Company is actively recruiting (public careers page found)",
                        "detail": (
                            "Hiring inferred from an applicant-tracking system link."
                            if is_ats
                            else "Hiring inferred from a reachable careers page."
                        ),
                        "date": None,
                        "source_url": block.url,
                        "recency_note": "undated",
                        "_sort": (_KIND_ORDER["hiring"], 2),
                    }
                )
                break

    found.sort(key=lambda s: s["_sort"])
    for signal in found:
        signal.pop("_sort", None)
    return found[:limit]


# ---------------------------------------------------------------------------
# firmographics
# ---------------------------------------------------------------------------

_EMPLOYEE_RANGE_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*)\s*[-–—]\s*(\d{1,3}(?:,\d{3})*)\s*\+?\s*employees\b", re.I)
_EMPLOYEE_COUNT_RE = re.compile(
    r"\b(?:over|about|approximately|more than|nearly|around|~)?\s*(\d{1,3}(?:,\d{3})*|\d{1,7})\+?\s*"
    r"(?:employees|people|staff|team members|teammates|engineers)\b",
    re.I,
)
_TEAM_OF_RE = re.compile(r"\bteam of\s+(?:over|about|nearly|around|more than|~)?\s*(\d{1,3}(?:,\d{3})*|\d{1,7})\b", re.I)

_FOUNDED_RE = re.compile(
    r"\b(?:founded|established|incorporated|started|launched|est\.?|since|founded\s+in)\s*(?:in|:)?\s*((?:1[89]|20)\d{2})\b",
    re.I,
)

# Case-insensitive on the lead-in only: the place name itself must stay Capitalised,
# which is what stops these from swallowing arbitrary prose. A word ending in "."
# must be a known abbreviation, otherwise a sentence-final period lets the place
# name bleed into the next sentence ("California. Founded").
_PLACE_WORD = r"(?:St\.|Ft\.|Mt\.|Ste\.|[A-Z][A-Za-z\-']+)"
_PLACE = rf"{_PLACE_WORD}(?:\s+{_PLACE_WORD}){{0,2}}"
_HQ_LEAD = r"\b(?i:headquarter(?:s|ed)?|head office|hq|based|located)\s*(?i:is in|are in|in|at|:)?\s*"
_HQ_RE = re.compile(_HQ_LEAD + rf"({_PLACE})\s*,\s*({_PLACE})")
_HQ_CITY_ONLY_RE = re.compile(_HQ_LEAD + rf"({_PLACE})\b")
#: Footer address lines: "228 Park Ave S, New York, NY 10003"
_ADDRESS_RE = re.compile(rf"({_PLACE}),\s*([A-Z]{{2}})\s+\d{{5}}(?:-\d{{4}})?\b")
#: Agent A surfaces fetched location candidates; prompts render them as a line.
_LOCATIONS_LINE_RE = re.compile(r"^\s*locations?\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)

_US_STATES = frozenset(
    {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
        "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
        "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
        "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
        "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
        "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
        "washington", "west virginia", "wisconsin", "wyoming",
        "ca", "ny", "tx", "wa", "ma", "il", "co", "ga", "fl", "pa", "nc", "va", "or", "ut", "dc",
        "az", "nj", "mn", "mi", "oh", "tn", "mo", "md", "in", "wi", "nv",
    }
)

# --- industry vocabulary --------------------------------------------------
#
# Label -> ((needle, weight), ...). Specific phrases ("soc 2", "spend management")
# are strong evidence of what a company SELLS; bare words ("analytics",
# "security") are weak, because they show up in anyone's nav, footer and script
# inventory. Scoring is weighted and region-aware — see `_score_industries` —
# so dict order carries no meaning at all.
_INDUSTRY_VOCAB: dict[str, tuple[tuple[str, float], ...]] = {
    "Security & Compliance": (
        ("soc 2", 3.5), ("soc2", 3.5), ("iso 27001", 3.5), ("iso27001", 3.5), ("hipaa", 3.0),
        ("pci dss", 3.0), ("fedramp", 3.0), ("hitrust", 3.0), ("nist", 2.0), ("gdpr", 2.0),
        ("compliance automation", 3.5), ("security compliance", 3.5), ("trust management", 3.5),
        ("grc", 3.0), ("governance, risk", 3.0), ("risk and compliance", 3.0),
        ("security questionnaire", 3.0), ("audit-ready", 2.5), ("audit readiness", 2.5),
        ("security posture", 2.5), ("vulnerability management", 3.0), ("penetration testing", 3.0),
        ("zero trust", 2.5), ("siem", 2.5), ("threat detection", 3.0), ("endpoint protection", 3.0),
        ("identity and access", 2.5), ("cybersecurity", 2.5), ("infosec", 2.0),
        ("compliance", 1.5), ("security", 0.8),
    ),
    "Fintech": (
        ("fintech", 3.0), ("spend management", 3.5), ("expense management", 3.5),
        ("corporate card", 3.5), ("finance operations", 3.0), ("financial operations", 3.0),
        ("accounts payable", 3.0), ("accounting automation", 3.0), ("invoicing", 2.5),
        ("payments", 2.0), ("payment processing", 3.0), ("treasury", 2.5), ("lending", 2.5),
        ("banking", 2.0), ("underwriting", 2.0), ("reimbursement", 2.0), ("bill pay", 3.0),
    ),
    "Developer Tools & Infrastructure": (
        ("developer tools", 3.5), ("developer platform", 3.0), ("devtools", 3.0),
        ("ci/cd", 3.0), ("continuous integration", 3.0), ("observability", 3.0),
        ("infrastructure as code", 3.0), ("kubernetes", 2.5), ("devops", 2.5),
        ("api platform", 3.0), ("sdk", 1.5), ("code review", 2.5), ("deployment", 1.5),
        ("serverless", 2.5), ("developers", 1.0),
    ),
    "Data & Analytics": (
        ("data warehouse", 3.5), ("data infrastructure", 3.5), ("data pipeline", 3.0),
        ("business intelligence", 3.5), ("product analytics", 3.5), ("analytics platform", 3.0),
        ("data platform", 2.5), ("etl", 2.5), ("data lake", 3.0), ("dashboards", 1.5),
        ("reporting", 1.0), ("analytics", 1.0), ("data science", 2.0),
    ),
    "HR & Payroll": (
        ("payroll", 3.5), ("applicant tracking", 3.5), ("hris", 3.5), ("people ops", 3.0),
        ("benefits administration", 3.0), ("recruiting software", 3.0), ("hr software", 3.5),
        ("employee onboarding", 2.5), ("performance reviews", 2.5), ("workforce", 1.5),
    ),
    "Marketing Technology": (
        ("marketing automation", 3.5), ("martech", 3.0), ("email marketing", 3.0),
        ("lifecycle marketing", 3.0), ("customer engagement", 2.5), ("adtech", 3.0),
        ("demand generation", 2.5), ("crm", 2.0), ("seo", 1.5),
    ),
    "Healthtech": (
        ("healthtech", 3.5), ("electronic health record", 3.5), ("ehr", 3.0), ("telehealth", 3.5),
        ("clinical", 2.0), ("patient", 1.5), ("medical records", 3.0), ("care delivery", 2.5),
    ),
    "E-commerce": (("e-commerce", 3.0), ("ecommerce", 3.0), ("online store", 2.5), ("checkout", 2.0), ("merchants", 2.0)),
    "Logistics": (("logistics", 3.0), ("supply chain", 3.0), ("freight", 3.0), ("warehouse automation", 3.5), ("fulfilment", 2.5), ("fulfillment", 2.5)),
    "Insurtech": (("insurtech", 3.5), ("insurance", 2.0), ("claims processing", 3.0), ("policyholder", 3.0)),
    "Proptech": (("proptech", 3.5), ("real estate", 2.5), ("property management", 3.0)),
    "Edtech": (("edtech", 3.5), ("learning management", 3.0), ("online courses", 2.5), ("students", 1.5)),
    "Legaltech": (("legaltech", 3.5), ("contract lifecycle", 3.0), ("legal operations", 3.0), ("e-discovery", 3.0)),
    "Robotics": (("robotics", 3.0), ("autonomous robots", 3.5), ("robots", 2.0)),
    "Biotech": (("biotech", 3.0), ("drug discovery", 3.5), ("genomics", 3.0), ("therapeutics", 3.0)),
    "Marketplace": (("marketplace", 2.5), ("two-sided market", 3.0), ("buyers and sellers", 3.0)),
    "Artificial Intelligence": (("artificial intelligence", 2.0), ("machine learning", 2.0), ("llm", 2.0), ("foundation model", 3.0)),
}

#: Winning needle -> a more precise sub_industry, when one applies.
_SUB_INDUSTRY = {
    "soc 2": "Compliance Automation", "soc2": "Compliance Automation",
    "iso 27001": "Compliance Automation", "compliance automation": "Compliance Automation",
    "security compliance": "Compliance Automation", "trust management": "Trust Management",
    "grc": "GRC", "security questionnaire": "Compliance Automation",
    "vulnerability management": "Vulnerability Management", "threat detection": "Threat Detection",
    "spend management": "Spend Management", "expense management": "Spend Management",
    "corporate card": "Spend Management", "payments": "Payments",
    "payment processing": "Payments", "accounts payable": "Accounts Payable",
    "payroll": "Payroll", "applicant tracking": "Applicant Tracking",
    "observability": "Observability", "ci/cd": "CI/CD",
    "data warehouse": "Data Infrastructure", "data infrastructure": "Data Infrastructure",
    "product analytics": "Product Analytics", "business intelligence": "Business Intelligence",
    "marketing automation": "Marketing Automation", "warehouse automation": "Warehouse Automation",
    "telehealth": "Telehealth", "electronic health record": "EHR",
}

#: A winner must clear this, and beat the runner-up by `_INDUSTRY_MARGIN`.
#: Below either bar the honest answer is None.
_INDUSTRY_FLOOR = 4.0
_INDUSTRY_MARGIN = 1.35
#: One needle can only vote so many times, so a repeated nav item cannot decide.
_NEEDLE_OCCURRENCE_CAP = 3

# Region weights: where a term appears matters more than how often.
_W_LABELLED = 4.0   # an explicit title:/description:/<title> line
_W_LEAD = 2.5       # the first few lines of a block: title, h1, lead paragraph
_W_BODY = 1.0       # ordinary prose

#: Lines that are inventory, chrome or boilerplate — they do not get to vote on
#: what a company sells. "Detected technologies: Google Analytics 4, ..." naming
#: an analytics vendor is exactly how Vanta got labelled "Analytics".
_NON_PROSE_LINE_RE = re.compile(
    r"^\s*(?:detected\s+technolog\w*|tech(?:nolog\w*)?|stack|scripts?|vendors?|cookies?|"
    r"privacy|terms|copyright|all\s+rights\s+reserved)\s*[:=]|"
    r"^\s*(?:©|&copy;)|"
    r"\b(?:googletagmanager|cdn\.|\.js\b|\.css\b|<script|<style|data-w-id|window\.datalayer)\b|"
    r"\ball\s+rights\s+reserved\b|\bcookie\s+(?:policy|settings|consent)\b",
    re.IGNORECASE,
)
_DESCRIPTION_LINE_RE = re.compile(r"^\s*(?:description|meta[\s_-]?description|summary)\s*[:=]\s*(.+)$", re.IGNORECASE)

#: Needles must match on word boundaries — plain `in` counts "hris" inside
#: "Chris" and "etl" inside "Bartlett".
_NEEDLE_RE: dict[str, re.Pattern[str]] = {
    needle: re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)", re.IGNORECASE)
    for needles in _INDUSTRY_VOCAB.values()
    for needle, _ in needles
}

# Tech names Agent A's detector surfaces, mapped to a category.
_TECH_CATALOG = {
    "greenhouse": ("ats", "Greenhouse"), "lever": ("ats", "Lever"), "ashby": ("ats", "Ashby"),
    "workday": ("hr", "Workday"), "segment": ("analytics", "Segment"),
    "google analytics": ("analytics", "Google Analytics"), "amplitude": ("analytics", "Amplitude"),
    "mixpanel": ("analytics", "Mixpanel"), "hubspot": ("crm", "HubSpot"),
    "salesforce": ("crm", "Salesforce"), "marketo": ("martech", "Marketo"),
    "intercom": ("support", "Intercom"), "zendesk": ("support", "Zendesk"),
    "drift": ("martech", "Drift"), "next.js": ("framework", "Next.js"),
    "react": ("framework", "React"), "webflow": ("cms", "Webflow"),
    "wordpress": ("cms", "WordPress"), "contentful": ("cms", "Contentful"),
    "cloudflare": ("hosting", "Cloudflare"), "vercel": ("hosting", "Vercel"),
    "aws": ("hosting", "AWS"), "stripe": ("payments", "Stripe"),
    "snowflake": ("data", "Snowflake"), "datadog": ("observability", "Datadog"),
}

# Page-title furniture that must never become a company name.
_TITLE_FURNITURE = re.compile(
    r"\b(about(?:\s+us)?|home(?:page)?|careers?|jobs?|pricing|blog|contact(?:\s+us)?|login|"
    r"sign\s?in|sign\s?up|products?|solutions?|resources?|news(?:room)?|press|support|docs?|"
    r"documentation|privacy|terms|customers?|platform|overview|welcome|different by design)\b",
    re.I,
)
_TITLE_LINE_RE = re.compile(r"^\s*(?:<title>\s*(.+?)\s*</title>|title\s*[:=]\s*(.+?))\s*$", re.IGNORECASE | re.MULTILINE)


def _clean_company_name(target: str, blocks: list[_Evidence]) -> Optional[str]:
    """The caller's target beats any page title. Titles are a last resort, cleaned."""
    t = (target or "").strip()
    if t and t.lower() not in ("the company", "unknown", ""):
        if "://" in t or (" " not in t and "." in t):
            label = _slug(t)
            return label.capitalize() if label else None
        return t

    for block in blocks:
        if not block.ok:
            continue
        m = _TITLE_LINE_RE.search(block.text)
        if not m:
            continue
        title = (m.group(1) or m.group(2) or "").strip()
        for fragment in re.split(r"\s*[|•·–—]\s*|\s+-\s+", title):
            fragment = fragment.strip(" -|")
            if not fragment or _TITLE_FURNITURE.search(fragment):
                continue
            if 1 < len(fragment) <= 60:
                return fragment
    return None


def _pick_domain(blocks: list[_Evidence], tokens: list[str]) -> Optional[str]:
    """Most frequent non-aggregator host, preferring one that resembles the target.

    A host that only ever appeared in a failed fetch can never win — that is how
    "tracxn.com" ended up in a record it had no business being in.
    """
    tally: dict[str, int] = {}
    homepage: dict[str, bool] = {}
    for block in blocks:
        host = block.host
        if not host or not block.ok or _is_aggregator(host):
            continue
        tally[host] = tally.get(host, 0) + 1
        try:
            path = urlparse(block.url).path
        except ValueError:  # pragma: no cover - defensive
            path = ""
        homepage[host] = homepage.get(host, False) or path in ("", "/")
    if not tally:
        return None

    def resemblance(host: str) -> float:
        bare = host.split(".")[0]
        for tok in tokens:
            if tok == bare:
                return 20.0
            if tok in bare or bare in tok:
                return 12.0
        return 0.0

    def score(host: str) -> tuple[float, int, str]:
        return (resemblance(host) + (2.0 if homepage.get(host) else 0.0), tally[host], host)

    best = max(tally, key=score)
    if resemblance(best) > 0 or tally[best] >= 2:
        return best
    logger.debug("No host in the evidence resembles %r; leaving domain unknown", tokens)
    return None


def _first_int(pattern: re.Pattern[str], text: str, lo: int, hi: int) -> Optional[int]:
    for m in pattern.finditer(text):
        for group in m.groups():
            if not group:
                continue
            try:
                value = int(str(group).replace(",", ""))
            except ValueError:
                continue
            if lo <= value <= hi:
                return value
    return None


def _band(n: Optional[int]) -> Optional[str]:
    if not n:
        return None
    for low, high in ((1, 10), (11, 50), (51, 200), (201, 500), (501, 1000), (1001, 5000), (5001, 10000)):
        if n <= high:
            return f"{low}-{high}"
    return "10000+"


def _extract_hq(text: str) -> tuple[Optional[str], Optional[str]]:
    m = _HQ_RE.search(text)
    if m:
        return m.group(1).strip(), _country_of(m.group(2).strip())
    m = _ADDRESS_RE.search(text)
    if m:
        return m.group(1).strip(), _country_of(m.group(2).strip())
    m = _LOCATIONS_LINE_RE.search(text)
    if m:
        first = m.group(1).split(";")[0].strip()
        parts = [p.strip() for p in first.split(",")]
        if parts and parts[0]:
            return parts[0], _country_of(parts[1]) if len(parts) > 1 else None
    m = _HQ_CITY_ONLY_RE.search(text)
    if m:
        return m.group(1).strip(), None
    return None, None


def _country_of(region: Optional[str]) -> Optional[str]:
    """"California" / "NY" name a US state, not a country. Say so honestly."""
    if not region:
        return None
    return "United States" if region.lower() in _US_STATES else region


def _first_sentence_about(blocks: list[_Evidence], tokens: list[str]) -> Optional[str]:
    if not tokens:
        return None
    for block in blocks:
        if not block.ok:
            continue
        for sentence in _sentences(block.text):
            if len(sentence) < 40 or sentence.startswith(("-", "#", "TARGET", "---")):
                continue
            low = sentence.lower()
            if any(tok in low for tok in tokens) and not _TITLE_FURNITURE.match(sentence):
                return sentence[:400]
    return None


def _strip_non_prose(line: str, vendor_names: tuple[str, ...]) -> str:
    """Remove URLs and known vendor names so they cannot vote on the industry."""
    cleaned = _URL_RE.sub(" ", line)
    cleaned = re.sub(r"\b[\w-]+\.(?:com|io|co|net|org|ai|dev)\b", " ", cleaned)
    for vendor in vendor_names:
        cleaned = re.sub(re.escape(vendor), " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _industry_corpus(blocks: list[_Evidence]) -> list[tuple[str, float]]:
    """Split the evidence into ``(text, region_weight)`` segments.

    Title/description lines outrank the lead of a block, which outranks ordinary
    prose. Script inventories, cookie banners and footers are dropped entirely.
    """
    vendors = tuple(name for _, (_, name) in _TECH_CATALOG.items())
    segments: list[tuple[str, float]] = []

    for block in blocks:
        if not block.ok:
            continue
        prose: list[str] = []
        for line in block.text.splitlines():
            if not line.strip():
                continue
            if _NON_PROSE_LINE_RE.search(line):
                continue
            m = _TITLE_LINE_RE.match(line) or _DESCRIPTION_LINE_RE.match(line)
            if m:
                value = next((g for g in m.groups() if g), "")
                segments.append((_strip_non_prose(value, vendors), _W_LABELLED))
                continue
            prose.append(_strip_non_prose(line, vendors))

        # The first few prose lines of a page are its title, h1 and lead
        # paragraph — the part that actually says what the company does.
        lead, body = prose[:3], prose[3:]
        if lead:
            segments.append((" \n".join(lead), _W_LEAD))
        if body:
            segments.append((" \n".join(body), _W_BODY))
    return segments


def _score_industries(blocks: list[_Evidence]) -> dict[str, tuple[float, str]]:
    """Label -> (score, strongest matching needle). Weighted, not first-match."""
    segments = _industry_corpus(blocks)
    if not segments:
        return {}

    scored: dict[str, tuple[float, str]] = {}
    for label, needles in _INDUSTRY_VOCAB.items():
        total = 0.0
        best_needle = ""
        best_contrib = 0.0
        for needle, weight in needles:
            contrib = 0.0
            pattern = _NEEDLE_RE[needle]
            for text, region_weight in segments:
                hits = len(pattern.findall(text))
                if hits:
                    contrib += min(hits, _NEEDLE_OCCURRENCE_CAP) * weight * region_weight
            if contrib > 0:
                total += contrib
                if contrib > best_contrib or (contrib == best_contrib and weight > 2.0 and not best_needle):
                    best_contrib, best_needle = contrib, needle
        if total > 0:
            scored[label] = (total, best_needle)
    return scored


def infer_industry(blocks: list[_Evidence]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(industry, sub_industry)``, or ``(None, None)`` when unsure.

    A single incidental mention never decides an industry: the winner must clear
    `_INDUSTRY_FLOOR` and beat the runner-up by `_INDUSTRY_MARGIN`. Ties resolve
    to unknown, which this project treats as a feature rather than a failure.
    """
    scored = _score_industries(blocks)
    if not scored:
        return None, None

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1][0], kv[0]))
    label, (score, needle) = ranked[0]
    if score < _INDUSTRY_FLOOR:
        logger.debug("Industry %r scored %.1f, below the floor; leaving unknown", label, score)
        return None, None
    if len(ranked) > 1:
        runner_up = ranked[1][1][0]
        if runner_up > 0 and score < runner_up * _INDUSTRY_MARGIN:
            logger.debug(
                "Industry is ambiguous (%s %.1f vs %s %.1f); leaving unknown",
                label, score, ranked[1][0], runner_up,
            )
            return None, None
    return label, _SUB_INDUSTRY.get(needle)


def _extract_tech(blocks: list[_Evidence]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in blocks:
        if not block.ok:
            continue
        lowered = block.text.lower()
        for needle, (category, name) in _TECH_CATALOG.items():
            if needle in lowered and name not in seen:
                seen.add(name)
                out.append(
                    {
                        "category": category,
                        "name": name,
                        "evidence": f"Name appears in page text/markup at {block.url or 'the fetched page'}.",
                        "confidence": 0.4,
                    }
                )
    return out[:12]


# ---------------------------------------------------------------------------


def _heuristic_pool(context: str, target: str, today: Optional[date] = None) -> dict[str, Any]:
    """Everything simple regex can honestly recover from the evidence."""
    blocks = _evidence_blocks(context)
    usable = [b for b in blocks if b.ok]
    tokens = _company_tokens(target)
    text = "\n".join(b.text for b in usable)

    pool: dict[str, Any] = {}
    pool["input"] = target
    pool["company_name"] = _clean_company_name(target, blocks)
    pool["domain"] = _pick_domain(blocks, tokens)

    urls = [b.url for b in usable if b.url]
    if not urls:
        urls = _urls_in(context)
    if urls:
        pool["sources"] = list(dict.fromkeys(urls))[:12]
        pool["evidence_urls"] = pool["sources"][:3]
        pool["source_url"] = pool["sources"][0]

    # --- firmographics ---------------------------------------------------
    year = _first_int(_FOUNDED_RE, text, 1800, (today or date.today()).year)
    if year:
        pool["founded_year"] = year

    m = _EMPLOYEE_RANGE_RE.search(text)
    if m:
        pool["employee_range"] = f"{m.group(1)}-{m.group(2)}"
    headcount = _first_int(_EMPLOYEE_COUNT_RE, text, 1, 3_000_000) or _first_int(_TEAM_OF_RE, text, 1, 3_000_000)
    if headcount:
        pool["employee_estimate"] = headcount
        pool.setdefault("employee_range", _band(headcount))

    city, country = _extract_hq(text)
    if city:
        pool["hq_city"] = city
    if country:
        pool["hq_country"] = country

    industry, sub_industry = infer_industry(blocks)
    if industry:
        pool["industry"] = industry
    if sub_industry:
        pool["sub_industry"] = sub_industry

    description = _first_sentence_about(usable, tokens)
    if description:
        pool["description"] = description

    # --- signals and tech ------------------------------------------------
    signals = extract_signals(blocks, today=today)
    if signals:
        pool["signals"] = signals
        top = signals[0]
        # Single-Signal extractions get the strongest one's fields directly.
        for key in ("kind", "headline", "detail", "date", "recency_note"):
            pool[key] = top.get(key)
        if top.get("source_url"):
            pool["source_url"] = top["source_url"]
        pool["grounded_in"] = top.get("headline")
        pool["angle"] = f"{top.get('kind', 'signal').title()} signal"

    tech = _extract_tech(blocks)
    if tech:
        pool["tech_stack"] = tech

    # --- narrative fields shared by several models -----------------------
    summary_bits = [f"Heuristic (no-LLM) extraction over {len(usable)} usable evidence block(s)."]
    if signals:
        summary_bits.append(f"{len(signals)} signal(s) found: " + ", ".join(s["kind"] for s in signals) + ".")
    else:
        summary_bits.append("No funding, hiring, product or leadership signal matched.")
    pool["summary"] = " ".join(summary_bits)
    pool["reasoning"] = pool["summary"]
    pool["confidence"] = 0.25
    return pool


# ---------------------------------------------------------------------------

#: Fields a later pipeline stage fills in — not honest "unknown facts" for the
#: extraction pass to confess to.
_NOT_REALLY_UNKNOWN = frozenset({"input", "unknowns", "sources", "icp_fit", "hooks", "evidence_urls"})


class StubClient:
    """Deterministic keyless LLMClient. Zero tokens, zero network."""

    name = "stub"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings
        self.model = "stub"
        # Lightweight memory across calls, unioned with what the context reveals.
        self._fetched: set[str] = set()
        self._searched: list[str] = []
        self._steps = 0

    # -- LLMClient -------------------------------------------------------

    def decide(
        self,
        system: str,
        context: str,
        allowed_tools: list[dict[str, Any]],
    ) -> tuple[AgentAction, LLMUsage]:
        usage = LLMUsage(tokens_in=0, tokens_out=0, model=self.model)
        ctx = context or ""
        target = _target_from_context(ctx)
        tokens = _company_tokens(target)

        allowed = self._allowed_names(allowed_tools)
        used = _tools_used(ctx)
        failed = _tools_failed(ctx)
        fetched = self._fetched | _fetched_urls(ctx)
        urls = [u for u in _urls_in(ctx) if u not in fetched]
        self._steps += 1

        steps_left = _int_after(_STEPS_LEFT_RE, ctx)
        searches_left = _int_after(_SEARCHES_LEFT_RE, ctx)
        fetches_left = _int_after(_FETCHES_LEFT_RE, ctx)
        jobs_left = _int_after(_JOBS_LEFT_RE, ctx)

        def act(tool: ToolName, thought: str, args: dict[str, Any]) -> tuple[AgentAction, LLMUsage]:
            return AgentAction(thought=thought, tool=tool, args=args), usage

        def finish(thought: str) -> tuple[AgentAction, LLMUsage]:
            return act(ToolName.FINISH, thought, {})

        # Budget guards come first — the stub must never overrun the loop's limits.
        if steps_left is not None and steps_left <= 1:
            return finish("Step budget is nearly spent; stopping with what has been gathered.")

        attempted_searches = used.get("web_search", 0)
        failed_searches = failed.get("web_search", 0)
        ok_searches = max(0, attempted_searches - failed_searches)

        can_search = ToolName.WEB_SEARCH.value in allowed and (searches_left is None or searches_left > 0)
        can_fetch = ToolName.FETCH_PAGE.value in allowed and (fetches_left is None or fetches_left > 0)
        can_tech = ToolName.DETECT_TECH.value in allowed and (fetches_left is None or fetches_left > 0)
        # One job-board lookup is plenty: it returns exact role counts and a
        # department split in a single call, and Agent C budgets them separately.
        can_jobs = (
            ToolName.FETCH_JOBS.value in allowed
            and used.get("fetch_jobs", 0) == 0
            and (jobs_left is None or jobs_left > 0)
        )

        search_arg = self._arg(allowed_tools, ToolName.WEB_SEARCH, ("query", "q", "search"), "query")
        fetch_arg = self._arg(allowed_tools, ToolName.FETCH_PAGE, ("url", "link", "page"), "url")

        # 1. No successful search yet -> establish who this company is. A failed
        #    search is a provider problem, not a reason to give up: rephrase.
        if ok_searches == 0 and can_search and attempted_searches < 3:
            query = f"{target} company overview" if attempted_searches == 0 else f'"{target}" official site'
            self._searched.append(query)
            reason = (
                f"Nothing is known yet about {target}. A broad overview search should surface the "
                "official site and a one-line description to anchor everything else."
                if attempted_searches == 0
                else f"The previous search failed ({failed_searches} failed so far). Retrying with a "
                "narrower phrasing before falling back to guessing the domain."
            )
            return act(ToolName.WEB_SEARCH, reason, {search_arg: query})

        # 2. Search is not working at all -> go straight at the obvious domain.
        guess = self._guessed_homepage(target)
        if ok_searches == 0 and not urls and guess and guess not in fetched and can_fetch:
            self._fetched.add(guess)
            return act(
                ToolName.FETCH_PAGE,
                f"Search has failed {failed_searches or attempted_searches} time(s) and yielded no URLs. "
                f"{guess} is the obvious candidate for {target}'s own site — worth one direct fetch "
                "before giving up on firmographics.",
                {fetch_arg: guess},
            )

        # 3. Have results, homepage not fetched -> fetch the most likely official domain.
        homepage_fetched = any(self._looks_official(u, tokens) for u in fetched)
        if urls and not homepage_fetched and can_fetch:
            best = max(urls, key=lambda u: _score_official(u, tokens))
            if _score_official(best, tokens) > -20:
                self._fetched.add(best)
                return act(
                    ToolName.FETCH_PAGE,
                    f"The search results point at {best} as {target}'s own site. Firmographics "
                    "(what they sell, size, HQ, founding year) belong on the company's own pages.",
                    {fetch_arg: best},
                )

        # 4. Homepage in hand -> read its stack.
        homepage = next((u for u in sorted(fetched) if self._looks_official(u, tokens)), None)
        if homepage and used.get("detect_tech", 0) == 0 and can_tech:
            return act(
                ToolName.DETECT_TECH,
                f"{target}'s homepage is fetched but its stack is unknown. Detected technology is a "
                "cheap, concrete qualification signal and a hook to lean on later.",
                {self._arg(allowed_tools, ToolName.DETECT_TECH, ("url", "domain", "link"), "url"): homepage},
            )

        # 4b. A job board is the single best public evidence of hiring and org
        #     shape: exact open-role counts and a department split, where a
        #     careers page gives prose and the board's HTML shell gives nothing.
        if can_jobs:
            ats_url = next((u for u in urls if _is_ats_url(u)), None)
            ats_seen = ats_url or _ats_mentioned(ctx)
            hiring_wanted = "hiring" in ctx.lower()
            if ats_seen or hiring_wanted:
                jobs_arg = self._arg(
                    allowed_tools,
                    ToolName.FETCH_JOBS,
                    ("company_or_url", "url", "company", "domain", "query"),
                    "company_or_url",
                )
                # The board URL is the best handle; the domain and the bare name
                # both work as fallbacks, and the tool resolves them itself.
                handle = ats_url or _domain_of(fetched, tokens) or target
                if ats_url:
                    self._fetched.add(ats_url)
                return act(
                    ToolName.FETCH_JOBS,
                    f"{handle} is an applicant-tracking board. Reading it gives the exact number "
                    "of open roles and which departments are growing — the strongest hiring signal "
                    "available, and a far better size proxy than anything on the marketing site.",
                    {jobs_arg: handle},
                )

        # 5. Firmographics covered -> go looking for recent signals.
        if ok_searches < 2 and can_search and attempted_searches < 4:
            query = f"{target} funding OR hiring OR news 2025"
            self._searched.append(query)
            return act(
                ToolName.WEB_SEARCH,
                "Firmographics are covered but there is no dated signal yet. Funding, hiring and "
                "news are what an outreach hook can actually lean on.",
                {search_arg: query},
            )

        # 6. Read the most promising unread news/about/careers page. Capped: the
        #    homepage plus two signal pages is enough for a useful record, and
        #    trawling every remaining link is how budget gets wasted.
        if urls and can_fetch and used.get("fetch_page", 0) < 3:
            # An ATS board's HTML shell carries almost nothing; fetch_jobs reads
            # its JSON API instead, so never spend a page fetch on one.
            page_urls = [u for u in urls if not (_is_ats_url(u) and ToolName.FETCH_JOBS.value in allowed)]
            if not page_urls:
                page_urls = urls
            best = max(page_urls, key=lambda u: _score_promising(u, tokens))
            if _score_promising(best, tokens) > 0:
                self._fetched.add(best)
                return act(
                    ToolName.FETCH_PAGE,
                    f"{best} looks like the highest-value unread page (news / about / careers) for "
                    "turning the search snippets into a citable, dated signal.",
                    {fetch_arg: best},
                )

        # 7. Last resort before finishing: the careers page of a known domain.
        careers = self._guessed_careers(target, fetched, tokens)
        if careers and careers not in fetched and can_fetch and used.get("fetch_page", 0) < 3:
            self._fetched.add(careers)
            return act(
                ToolName.FETCH_PAGE,
                f"No unread search result looks worthwhile, but {careers} would give a direct hiring "
                "signal, which is the single most useful thing still missing.",
                {fetch_arg: careers},
            )

        return finish(
            "Firmographics, technology and signal searches are all done and no unread page looks "
            "worth the remaining budget. Stopping here rather than padding the trace."
        )

    def extract(
        self,
        system: str,
        context: str,
        schema_model: type[T],
        hint: Optional[str] = None,
    ) -> tuple[T, LLMUsage]:
        usage = LLMUsage(tokens_in=0, tokens_out=0, model=self.model)
        ctx = context or ""
        target = _target_from_context(ctx)
        pool = _heuristic_pool(ctx, target)

        data: dict[str, Any] = {}
        unfilled: list[str] = []

        for name in schema_model.model_fields:
            if name == "unknowns":
                continue
            value = pool.get(name)
            if value is None or value == "" or value == []:
                # Heuristics could not establish it. Required fields are left for
                # best_effort_instance / the model's own default to handle.
                if name not in _NOT_REALLY_UNKNOWN:
                    unfilled.append(name)
                continue
            data[name] = value

        if "unknowns" in schema_model.model_fields:
            data["unknowns"] = sorted(unfilled)

        try:
            return schema_model.model_validate(data), usage
        except ValidationError as exc:
            logger.debug(
                "StubClient.extract: direct validation of %s failed (%s); repairing",
                schema_model.__name__,
                exc.errors()[:3],
            )
            return best_effort_instance(schema_model, data), usage

    # -- internals -------------------------------------------------------

    @staticmethod
    def _allowed_names(allowed_tools: Optional[list[dict[str, Any]]]) -> set[str]:
        names: set[str] = set()
        for spec in allowed_tools or []:
            if isinstance(spec, dict):
                raw = spec.get("name") or spec.get("tool")
            else:
                raw = getattr(spec, "name", None)
            if raw is not None:
                names.add(str(getattr(raw, "value", raw)))
        return names or {t.value for t in ToolName}

    @staticmethod
    def _arg(
        allowed_tools: Optional[list[dict[str, Any]]],
        tool: ToolName,
        candidates: tuple[str, ...],
        default: str,
    ) -> str:
        """Pick the argument name this deployment's tool actually declares."""
        for spec in allowed_tools or []:
            if isinstance(spec, dict):
                raw = spec.get("name") or spec.get("tool")
                schema = spec.get("args_schema") or spec.get("input_schema") or spec.get("schema")
            else:
                raw = getattr(spec, "name", None)
                schema = getattr(spec, "args_schema", None)
            if str(getattr(raw, "value", raw)) != tool.value:
                continue
            props = schema.get("properties", schema) if isinstance(schema, dict) else {}
            if isinstance(props, dict):
                for cand in candidates:
                    if cand in props:
                        return cand
        return default

    @staticmethod
    def _looks_official(url: str, tokens: list[str]) -> bool:
        return _score_official(url, tokens) > 0

    @staticmethod
    def _guessed_homepage(target: str) -> Optional[str]:
        slug = _slug(target)
        return f"https://{slug}.com" if slug else None

    @staticmethod
    def _guessed_careers(target: str, fetched: set[str], tokens: list[str]) -> Optional[str]:
        """`https://<their domain>/careers`, derived from a page we already read."""
        for url in sorted(fetched):
            if not StubClient._looks_official(url, tokens):
                continue
            try:
                host = urlparse(url).netloc
            except ValueError:  # pragma: no cover - defensive
                continue
            if host:
                return f"https://{host}/careers"
        slug = _slug(target)
        return f"https://{slug}.com/careers" if slug else None
