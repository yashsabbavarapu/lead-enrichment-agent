"""Open roles from a company's public ATS job board.

Why this exists: headcount is the single hardest field to source honestly.
LinkedIn and Crunchbase are closed, and "we are 250 people" almost never appears
in prose — so `employee_range` comes back unknown and the ICP's company_size
criterion scores a shrug every time.

Applicant tracking systems are the way through. Ashby, Greenhouse, Lever and
Workable all publish their customers' boards as unauthenticated JSON, and
`tech_detect` already spots the board URL on a careers page. From one request we
get an exact open-role count (a real headcount proxy), the departments doing the
hiring (is there a GTM org at all?), and dated postings (a hard buying signal).

Everything here is counts, titles and places. Job *descriptions* are megabytes of
HTML we never read: they are dropped before parsing and before recording.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from lea.config import Settings
from lea.models import Observation, ToolName

from . import http as _http
from .fetch import fetch_page
from .tech_detect import board_url_for, extract_ats_boards, recall_ats_boards, remember_ats_boards

# --------------------------------------------------------------------------
# limits
# --------------------------------------------------------------------------

# Snowflake's Ashby board is 4.3MB. Past this we are not reading a job board.
MAX_RESPONSE_CHARS = 8_000_000
# Aggregate at most this many roles; `open_roles` still reports the true total.
MAX_ROLES_AGGREGATED = 2_000
MAX_TITLES = 15
MAX_LOCATIONS = 12
MAX_DEPARTMENTS = 25
# Only tried when no ATS was detected — two 404s, not a four-platform sweep.
MAX_SLUG_GUESSES = 2
GUESS_ORDER: tuple[str, ...] = ("ashby", "greenhouse")
# Careers URLs probed when we have a domain but no remembered board.
CAREERS_PATHS: tuple[str, ...] = ("/careers", "/jobs")

RECENT_WINDOW_DAYS = 30

# Job-description fields, per platform. Dropped from the fixture so a 2.5MB
# board records as ~100KB of *valid* JSON — blind truncation would not replay.
DESCRIPTION_KEYS = frozenset(
    {
        "description", "descriptionHtml", "descriptionPlain", "descriptionSocial",
        "descriptionBody", "descriptionBodyPlain", "content", "additional",
        "additionalPlain", "lists", "opening", "openingPlain",
        "salaryDescription", "salaryDescriptionPlain", "data_compliance",
    }
)


# --------------------------------------------------------------------------
# the common shape every platform is flattened into
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    title: str
    department: str = ""
    location: str = ""
    remote: bool = False
    posted: str = ""  # ISO date; "" when the board publishes none


def _s(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value not in (None, "") else ""


def _iso_date(value: Any) -> str:
    """Boards date things three different ways. Normalize to YYYY-MM-DD or ''."""
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):  # Lever: epoch milliseconds
        try:
            seconds = float(value) / (1000.0 if float(value) > 1e11 else 1.0)
            return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
        except Exception:
            return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        # Ashby/Greenhouse hand back full ISO timestamps, Workable a bare date.
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", text) else ""


def _looks_remote(*values: Any) -> bool:
    return any("remote" in str(v).lower() for v in values if v not in (None, ""))


# --------------------------------------------------------------------------
# per-platform parsers (shapes verified against live responses)
# --------------------------------------------------------------------------


def parse_ashby(payload: Any) -> list[Role]:
    """`{"jobs": [{title, department, team, location, secondaryLocations, ...}]}`"""
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    roles: list[Role] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        if job.get("isListed") is False:
            continue  # unlisted postings are not public openings
        roles.append(
            Role(
                title=_s(job.get("title")),
                department=_s(job.get("department")) or _s(job.get("team")),
                location=_s(job.get("location")),
                remote=bool(job.get("isRemote")) or _looks_remote(job.get("location")),
                posted=_iso_date(job.get("publishedAt")),
            )
        )
    return roles


def parse_greenhouse(payload: Any) -> list[Role]:
    """Two shapes from one API: `/departments` (nested, has department names)
    and `/jobs` (flat, does not). A job can appear under a parent and a child
    department, so ids are deduplicated — that is what makes the count match
    the board's own `meta.total`."""
    if not isinstance(payload, dict):
        return []

    roles: list[Role] = []
    if isinstance(payload.get("departments"), list):
        seen: set[Any] = set()
        for dept in payload["departments"]:
            if not isinstance(dept, dict):
                continue
            name = _s(dept.get("name"))
            for job in dept.get("jobs") or []:
                if not isinstance(job, dict):
                    continue
                key = job.get("id", id(job))
                if key in seen:
                    continue
                seen.add(key)
                roles.append(_greenhouse_role(job, name))
        return roles

    for job in payload.get("jobs") or []:
        if isinstance(job, dict):
            roles.append(_greenhouse_role(job, ""))
    return roles


def _greenhouse_role(job: dict[str, Any], department: str) -> Role:
    location = job.get("location")
    where = _s(location.get("name")) if isinstance(location, dict) else _s(location)
    return Role(
        title=_s(job.get("title")),
        department=department,
        location=where,
        remote=_looks_remote(where),
        posted=_iso_date(job.get("first_published") or job.get("updated_at")),
    )


def parse_lever(payload: Any) -> list[Role]:
    """A bare list of postings; everything useful lives under `categories`."""
    roles: list[Role] = []
    for job in payload or []:
        if not isinstance(job, dict):
            continue
        cats = job.get("categories") if isinstance(job.get("categories"), dict) else {}
        where = _s(cats.get("location"))
        roles.append(
            Role(
                title=_s(job.get("text")),
                # Lever customers disagree on which field is the org unit: some
                # put "Engineering" in department, some put the brand there and
                # the real team in `team`.
                department=_s(cats.get("department")) or _s(cats.get("team")),
                location=where,
                remote=_looks_remote(where, job.get("workplaceType")),
                posted=_iso_date(job.get("createdAt")),
            )
        )
    return roles


def parse_workable(payload: Any) -> list[Role]:
    """`{"name": ..., "jobs": [{title, department, city, country, ...}]}`"""
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    roles: list[Role] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        where = ", ".join(p for p in (_s(job.get("city")), _s(job.get("country"))) if p)
        roles.append(
            Role(
                title=_s(job.get("title")),
                department=_s(job.get("department")) or _s(job.get("function")),
                location=where,
                remote=bool(job.get("telecommuting")) or _looks_remote(where),
                posted=_iso_date(job.get("published_on") or job.get("created_at")),
            )
        )
    return roles


# --------------------------------------------------------------------------
# platform table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Platform:
    key: str
    label: str
    # Tried in order. Greenhouse needs two because only `/departments` carries
    # department names, and a board with none still has to answer "how many?".
    api_urls: tuple[str, ...]
    parse: Callable[[Any], list[Role]]


PLATFORMS: dict[str, Platform] = {
    "ashby": Platform(
        "ashby", "Ashby",
        ("https://api.ashbyhq.com/posting-api/job-board/{slug}",),
        parse_ashby,
    ),
    "greenhouse": Platform(
        "greenhouse", "Greenhouse",
        (
            "https://boards-api.greenhouse.io/v1/boards/{slug}/departments",
            "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
        ),
        parse_greenhouse,
    ),
    "lever": Platform(
        "lever", "Lever",
        ("https://api.lever.co/v0/postings/{slug}?mode=json",),
        parse_lever,
    ),
    "workable": Platform(
        "workable", "Workable",
        ("https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true",),
        parse_workable,
    ),
}


def prune_descriptions(raw: str) -> str:
    """Strip job-description fields so the fixture stays reviewable JSON."""
    try:
        payload = json.loads(raw)
    except Exception:
        return raw

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in DESCRIPTION_KEYS}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    try:
        return json.dumps(walk(payload), separators=(",", ":"))
    except Exception:
        return raw


# --------------------------------------------------------------------------
# resolving company -> (platform, slug)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    platform: str
    slug: str
    how: str  # provenance, surfaced in data["resolved_by"]


_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$", re.I)


def slugify(name: str) -> str:
    """'Match Group, Inc.' -> 'matchgroup'. How ATS slugs are usually minted."""
    cleaned = re.sub(r"\b(inc|llc|ltd|corp|co|gmbh|sa|bv|plc)\b\.?", " ", (name or "").lower())
    return re.sub(r"[^a-z0-9]", "", cleaned)


def _domain_of(value: str) -> str:
    """The registrable-ish host in the input, or '' if it is a bare name."""
    raw = (value or "").strip().lower().rstrip("/")
    if "://" in raw:
        host = (urlparse(raw).hostname or "").strip(".")
    elif _HOST_RE.match(raw.split("/")[0]):
        host = raw.split("/")[0]
    else:
        return ""
    return host[4:] if host.startswith("www.") else host


def _candidates_from_boards(boards: Iterable[dict[str, str]], how: str) -> list[Candidate]:
    out: list[Candidate] = []
    for b in boards:
        platform, slug = (b.get("platform") or "").lower(), b.get("slug") or ""
        if platform in PLATFORMS and slug:
            out.append(Candidate(platform, slug, how))
    return out


def resolve_candidates(company_or_url: str, settings: Settings) -> tuple[list[Candidate], list[str]]:
    """Ordered boards worth trying, cheapest provenance first, plus trace notes.

    1. the input is already an ATS board URL      — free
    2. a board `detect_tech` saw earlier this run — free
    3. the company's own careers page             — at most 2 page fetches
    4. the domain label guessed as the slug       — at most MAX_SLUG_GUESSES calls
    """
    value = (company_or_url or "").strip()
    notes: list[str] = []

    direct = _candidates_from_boards(extract_ats_boards("", value), "board_url")
    if direct:
        return direct[:1], notes

    domain = _domain_of(value)
    remembered = _candidates_from_boards(recall_ats_boards(domain or value), "detected_ats")
    if remembered:
        return remembered[:1], notes

    if domain:
        for path in CAREERS_PATHS:
            page = fetch_page(f"https://{domain}{path}", settings)
            if not page.ok:
                notes.append(f"{domain}{path}: {page.error or 'unreadable'}")
                continue
            boards = extract_ats_boards(page.text or "", page.url or "")
            if boards:
                remember_ats_boards(domain, boards)
                found = _candidates_from_boards(boards, "careers_page")
                if found:
                    return found[:1], notes
            notes.append(f"no ATS board linked from {domain}{path}")

    # Nothing detected: the domain label (or the company name) is very often the
    # slug. Bounded on purpose — a miss here costs one 404 per platform.
    label = slugify(domain.split(".")[0] if domain else value)
    if not label:
        return [], notes
    notes.append(f"no ATS detected; trying '{label}' as a board slug")
    return [Candidate(p, label, "slug_guess") for p in GUESS_ORDER[:MAX_SLUG_GUESSES]], notes


# --------------------------------------------------------------------------
# fetching one board
# --------------------------------------------------------------------------


@dataclass
class BoardResult:
    ok: bool
    roles: list[Role]
    api_url: str
    error: str = ""
    missing: bool = False  # 404/410: this board does not exist, try the next
    from_fixture: bool = False
    truncated: bool = False


def fetch_board(candidate: Candidate, settings: Settings) -> BoardResult:
    """One platform, one slug. Never raises; a 404 is `missing`, not a crash."""
    platform = PLATFORMS[candidate.platform]
    last = BoardResult(False, [], "", error="no endpoint tried")

    for template in platform.api_urls:
        api_url = template.format(slug=candidate.slug)
        res = _http.request(
            settings,
            "GET",
            api_url,
            headers={"Accept": "application/json"},
            check_robots=True,
            record_transform=prune_descriptions,
        )
        if not res.ok:
            missing = res.status in (404, 410)
            last = BoardResult(
                False, [], api_url,
                error=(
                    f"no public {platform.label} board at '{candidate.slug}'"
                    if missing
                    else (res.error or f"HTTP {res.status}")
                ),
                missing=missing,
                from_fixture=res.from_fixture,
            )
            if missing:
                return last  # the slug is wrong; another endpoint will not help
            continue

        text = res.text or ""
        if len(text) > MAX_RESPONSE_CHARS:
            return BoardResult(
                False, [], api_url,
                error=(
                    f"{platform.label} board '{candidate.slug}' returned "
                    f"{len(text):,} chars, over the {MAX_RESPONSE_CHARS:,} limit"
                ),
                from_fixture=res.from_fixture,
            )
        try:
            payload = json.loads(text)
        except Exception as exc:
            last = BoardResult(
                False, [], api_url,
                error=f"{platform.label} board '{candidate.slug}' returned unparseable JSON: {exc}",
                from_fixture=res.from_fixture,
            )
            continue

        roles = [r for r in platform.parse(payload) if r.title]
        truncated = len(roles) > MAX_ROLES_AGGREGATED
        result = BoardResult(
            True, roles, api_url, from_fixture=res.from_fixture, truncated=truncated
        )
        if roles:
            return result
        # An empty first endpoint may just be the wrong *view* of a live board
        # (Greenhouse `/departments` on a board that files nothing under one),
        # so try the next one — but an empty last endpoint is a real answer.
        last = result

    return last


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def aggregate(roles: list[Role]) -> dict[str, Any]:
    sample = roles[:MAX_ROLES_AGGREGATED]

    departments: dict[str, int] = {}
    for r in sample:
        departments[r.department or "Unspecified"] = departments.get(r.department or "Unspecified", 0) + 1
    ordered_departments = dict(
        sorted(departments.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_DEPARTMENTS]
    )

    location_counts: dict[str, int] = {}
    spelling: dict[str, str] = {}
    for r in sample:
        if not r.location:
            continue
        key = r.location.lower()
        spelling.setdefault(key, r.location)
        location_counts[key] = location_counts.get(key, 0) + 1
    locations = [
        spelling[k]
        for k, _ in sorted(location_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_LOCATIONS]
    ]

    dates = sorted({r.posted for r in sample if r.posted}, reverse=True)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)).date().isoformat()

    return {
        "open_roles": len(roles),
        "departments": ordered_departments,
        "locations": locations,
        "titles": _title_sample(sample, ordered_departments),
        "remote_roles": sum(1 for r in sample if r.remote),
        "latest_posted": dates[0] if dates else "",
        "posted_last_30d": sum(1 for r in sample if r.posted and r.posted >= cutoff),
    }


def _title_sample(roles: list[Role], departments: dict[str, int]) -> list[str]:
    """Up to MAX_TITLES titles, spread across departments.

    Taking the first 15 in payload order would show fifteen engineering roles on
    an engineering-heavy board and hide the GTM hiring that actually qualifies
    the lead.
    """
    buckets: dict[str, list[str]] = {}
    for r in roles:
        buckets.setdefault(r.department or "Unspecified", []).append(r.title)

    out: list[str] = []
    seen: set[str] = set()
    order = list(departments) or list(buckets)
    while len(out) < MAX_TITLES:
        drained = True
        for dept in order:
            queue = buckets.get(dept)
            if not queue:
                continue
            drained = False
            title = queue.pop(0)
            if title.lower() not in seen:
                seen.add(title.lower())
                out.append(title)
            if len(out) >= MAX_TITLES:
                break
        if drained:
            break
    return out


def render_summary(company: str, ats: str, stats: dict[str, Any], board_url: str) -> str:
    """One compact line. This is what the LLM actually reads."""
    count = stats["open_roles"]
    if not count:
        return (
            f"{company} — 0 open roles on its public {ats} board ({board_url}). "
            f"The board is live but empty: not hiring publicly right now."
        )

    parts = [f"{company} — {count} open role{'s' if count != 1 else ''} on {ats}"]
    depts = list(stats["departments"].items())[:3]
    if depts:
        parts.append("across " + ", ".join(f"{name} ({n})" for name, n in depts))

    line = " ".join(parts)
    if stats["locations"]:
        line += "; locations incl. " + ", ".join(stats["locations"][:3])
    if stats["remote_roles"]:
        line += f"; {stats['remote_roles']} remote"
    if stats["posted_last_30d"]:
        line += (
            f"; {stats['posted_last_30d']} posted in the last {RECENT_WINDOW_DAYS} days"
            + (f" (latest {stats['latest_posted']})" if stats["latest_posted"] else "")
        )
    elif stats["latest_posted"]:
        line += f"; most recent posting {stats['latest_posted']}"
    return line + f". Board: {board_url}"


# Hosts whose first label is the ATS, never the company ('jobs.ashbyhq.com').
ATS_HOST_HINTS = ("ashbyhq.com", "greenhouse.io", "lever.co", "workable.com")


def _display_name(company_or_url: str, slug: str) -> str:
    """Prefer what the caller typed, as long as it was a name and not a URL."""
    value = (company_or_url or "").strip()
    slug_name = slug.replace("-", " ").replace("_", " ").title()
    domain = _domain_of(value)
    if not domain:
        return value if value and "/" not in value and "://" not in value else (slug_name or "This company")
    if any(hint in domain for hint in ATS_HOST_HINTS):
        return slug_name or "This company"
    return domain.split(".")[0].replace("-", " ").title() or slug_name or "This company"


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


class FetchJobsTool:
    name = ToolName.FETCH_JOBS
    description = (
        "Read a company's public applicant-tracking job board (Ashby, Greenhouse, "
        "Lever or Workable) and return the EXACT number of open roles, a count per "
        "department, hiring locations and a sample of job titles. No auth, one "
        "request, hard numbers rather than prose.\n"
        "Use it when: (a) detect_tech reported an ATS — pass the board URL it "
        "printed; (b) headcount or company size is still unknown, because open-role "
        "count and department spread are the best public proxy for org size and "
        "shape when LinkedIn and Crunchbase are unavailable; (c) you need evidence "
        "the company is growing, or which teams are growing — 'hiring 9 AEs' is a "
        "dated buying signal and an outreach hook. A GTM-heavy board (Sales, "
        "Marketing, Customer Success) says far more about ICP fit than a homepage "
        "does.\n"
        "Do NOT web_search for 'how many employees' first — try this. It accepts an "
        "ATS board URL, a company domain, or a company name, and finds the board "
        "itself. A live board with 0 roles is a real, useful answer: that company "
        "is not hiring publicly."
    )
    args_schema: dict[str, Any] = {
        "company_or_url": {
            "type": "string",
            "required": True,
            "description": (
                "An ATS board URL (best — e.g. https://jobs.ashbyhq.com/vanta or "
                "https://boards.greenhouse.io/duolingo), otherwise the company "
                "domain (vanta.com) or its name (Vanta)."
            ),
        },
    }

    def run(self, args: dict[str, Any], settings: Settings) -> Observation:
        started = time.perf_counter()
        raw = ""
        try:
            raw = str((args or {}).get("company_or_url") or "").strip()
            if not raw:
                return self._fail("missing required arg 'company_or_url'", raw, started)

            candidates, notes = resolve_candidates(raw, settings)
            if not candidates:
                return self._fail(
                    f"could not work out a job board for {raw!r}"
                    + (f" ({'; '.join(notes[:2])})" if notes else ""),
                    raw,
                    started,
                )

            attempts: list[str] = []
            board: Optional[BoardResult] = None
            chosen: Optional[Candidate] = None
            for candidate in candidates:
                result = fetch_board(candidate, settings)
                if result.ok:
                    board, chosen = result, candidate
                    break
                attempts.append(result.error)

            if board is None or chosen is None:
                return self._fail(
                    "; ".join(notes + attempts) or f"no public job board found for {raw!r}",
                    raw,
                    started,
                )

            platform = PLATFORMS[chosen.platform]
            board_url = board_url_for(chosen.platform, chosen.slug) or board.api_url
            company = _display_name(raw, chosen.slug)
            stats = aggregate(board.roles)

            data: dict[str, Any] = {
                "company": company,
                "ats": platform.label,
                "platform": platform.key,
                "slug": chosen.slug,
                "board_url": board_url,
                "api_url": board.api_url,
                "resolved_by": chosen.how,
                "truncated": board.truncated,
                "from_fixture": board.from_fixture,
                **stats,
            }
            return Observation(
                tool=self.name,
                ok=True,
                summary=render_summary(company, platform.label, stats, board_url),
                data=data,
                sources=[board_url],
                latency_ms=_ms(started),
            )
        except Exception as exc:
            return self._fail(f"{type(exc).__name__}: {exc}", raw, started)

    def _fail(self, error: str, raw: str, started: float) -> Observation:
        return Observation(
            tool=self.name,
            ok=False,
            summary=f"job board lookup failed: {error}",
            data={
                "company_or_url": raw,
                "open_roles": None,
                "departments": {},
                "locations": [],
                "titles": [],
            },
            error=error,
            sources=[],
            latency_ms=_ms(started),
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
