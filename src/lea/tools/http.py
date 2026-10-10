"""Shared HTTP plumbing: one client factory, a JSON fixture cache, and a robots.txt gate.

Every network call in the tool layer goes through `request()` so that `--offline`
is a hard guarantee (no socket is ever opened) and `--record-fixtures` captures a
replayable corpus for demos and tests.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

from lea.config import Settings

# DuckDuckGo (and a few CDNs) 403 anything that looks like a script. A plain
# desktop Chrome UA is the difference between results and an empty page.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# A real spread of desktop browsers. DuckDuckGo's bot heuristic keys partly on the
# UA, so rotating between these clears a lot of what merely looks like a rate limit.
BROWSER_UAS: tuple[str, ...] = (
    BROWSER_UA,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)


# --------------------------------------------------------------------------
# SSRF guard
# --------------------------------------------------------------------------

# fetch_page follows URLs an LLM picked out of untrusted search results, so the
# agent is a confused deputy by construction: without this, a hostile page can
# redirect it at cloud metadata or anything else on the host's network.
#
# Escape hatch lives here rather than on Settings, which is a frozen contract.
ALLOW_PRIVATE_HOSTS = False

MAX_REDIRECT_HOPS = 5

# Names refused regardless of what they resolve to: cloud metadata endpoints, and
# loopback names that must not depend on DNS behaving as expected.
BLOCKED_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata",
        "instance-data",
        "instance-data.ec2.internal",
        "localhost",
        "localhost.localdomain",
    }
)
BLOCKED_HOST_SUFFIXES = (".localhost", ".internal", ".local")
# Cloud metadata addresses, called out explicitly even though the link-local and
# unique-local checks below already cover them.
METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254"})

ALLOWED_SCHEMES = frozenset({"http", "https"})


def _resolve_host(host: str) -> list[str]:
    """Every IP a hostname resolves to. Indirection so tests can stub DNS."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return list({info[4][0] for info in infos})


def _ip_rejection(ip: str) -> Optional[str]:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None  # not an IP literal we can judge; DNS check handles the rest
    if ip in METADATA_IPS or (addr.is_link_local and str(addr).startswith("169.254.169.254")):
        return f"{ip} is a cloud metadata endpoint"
    # Ordered most specific first: 0.0.0.0 and 127.0.0.1 are both "private" too,
    # and the narrower label makes a better error message.
    for attr, label in (
        ("is_unspecified", "unspecified"),
        ("is_loopback", "loopback"),
        ("is_link_local", "link-local"),
        ("is_multicast", "multicast"),
        ("is_reserved", "reserved"),
        ("is_private", "private"),
    ):
        if getattr(addr, attr, False):
            return f"{ip} is a {label} address"
    return None


def url_rejection(url: str, *, allow_private: Optional[bool] = None) -> Optional[str]:
    """Why this URL must not be fetched, or None if it is fine.

    Resolves the hostname and judges the resolved IPs, because `evil.example.com`
    is free to point at 127.0.0.1.

    Known limitation: resolving here and connecting later leaves a DNS-rebinding
    window. Closing it fully means pinning the resolved IP through the transport,
    which is more machinery than this project warrants — noted, not hidden.
    """
    if allow_private is None:
        allow_private = ALLOW_PRIVATE_HOSTS
    if allow_private:
        return None

    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return f"could not parse URL: {url!r}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return f"refused {url} — only http and https are allowed, got {scheme or 'no scheme'!r}"

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return f"refused {url} — no hostname"
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_HOST_SUFFIXES):
        label = "a loopback" if "localhost" in host else "an internal or cloud-metadata"
        return f"refused {url} — {host} is {label} hostname"

    literal = _ip_rejection(host)
    if literal:
        return f"refused {url} — {literal}"

    try:
        resolved = _resolve_host(host)
    except Exception:
        # Unresolvable means unreachable; let the transport fail naturally rather
        # than blaming the guard for a DNS outage.
        return None

    for ip in resolved:
        reason = _ip_rejection(ip)
        if reason:
            return f"refused {url} — {host} resolves to {reason}"
    return None


class _BlockedRedirect(Exception):
    """A redirect hop pointed somewhere the guard refuses to follow."""


def _follow_redirects(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    data: Optional[dict[str, Any]],
    json_body: Optional[dict[str, Any]],
    headers: Optional[dict[str, str]],
    params: Optional[dict[str, Any]],
    allow_private: Optional[bool],
) -> httpx.Response:
    """Walk the redirect chain ourselves, validating each hop before following it."""
    current_method = method
    current_url = url
    current_params = params
    current_data = data
    current_json = json_body

    for _ in range(MAX_REDIRECT_HOPS + 1):
        resp = client.request(
            current_method,
            current_url,
            data=current_data,
            json=current_json,
            headers=headers,
            params=current_params,
        )
        location = resp.headers.get("location")
        if not (300 <= resp.status_code < 400 and location):
            return resp

        next_url = urljoin(str(resp.url), location)
        blocked = url_rejection(next_url, allow_private=allow_private)
        if blocked:
            raise _BlockedRedirect(f"{blocked} (redirected from {current_url})")

        # 303, and in practice 301/302, turn the follow-up into a bodyless GET.
        if resp.status_code in (301, 302, 303) and current_method != "HEAD":
            current_method = "GET"
            current_data = None
            current_json = None
        # The Location already carries its own query string.
        current_params = None
        current_url = next_url

    raise _BlockedRedirect(f"too many redirects (>{MAX_REDIRECT_HOPS}) starting at {url}")


# --------------------------------------------------------------------------
# Response envelope
# --------------------------------------------------------------------------


@dataclass
class HttpResult:
    """Uniform shape for live, replayed, and failed calls alike."""

    ok: bool
    url: str
    status: int = 0
    headers: dict[str, str] = None  # type: ignore[assignment]
    text: str = ""
    error: Optional[str] = None
    from_fixture: bool = False
    latency_ms: int = 0
    # Set when the request timed out, so callers can tell "unresponsive" apart
    # from "refused" — they deserve different retry policies.
    timed_out: bool = False

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = {}

    def header(self, name: str) -> str:
        return self.headers.get(name.lower(), "")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Client factory
# --------------------------------------------------------------------------


def build_client(
    settings: Settings,
    *,
    browser_ua: bool = False,
    user_agent: Optional[str] = None,
    follow_redirects: bool = True,
    timeout: Optional[float] = None,
) -> httpx.Client:
    """One sync client honoring Settings. Callers own closing it (or use `request`).

    `request()` passes follow_redirects=False so it can validate every hop itself;
    httpx following them internally would skip the SSRF guard on the destination.
    """
    ua = user_agent or (BROWSER_UA if browser_ua else settings.user_agent)
    return httpx.Client(
        timeout=timeout if timeout is not None else settings.http_timeout_s,
        follow_redirects=follow_redirects,
        headers={
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


# --------------------------------------------------------------------------
# Fixture cache
# --------------------------------------------------------------------------


def canonical_url(url: str) -> str:
    """Collapse the cosmetic differences that should never split a cache entry.

    `https://vanta.com`, `http://www.vanta.com/` and `https://VANTA.com/` are one
    page as far as a demo is concerned, and a fixture that misses over a `www.`
    prefix just looks broken.
    """
    raw = (url or "").strip()
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.lower()
    host = (parsed.netloc or "").lower()
    if host.endswith(":80"):
        host = host[:-3]
    elif host.endswith(":443"):
        host = host[:-4]
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").rstrip("/")
    if not host:
        return raw.lower()
    # Scheme is deliberately dropped: http vs https is not a different document.
    canonical = host + path
    if parsed.query:
        canonical += f"?{parsed.query}"
    return canonical


def fixture_key(url: str, body: Any = None) -> str:
    """sha256 over the canonical URL (+ body for POST); one request, one file."""
    h = hashlib.sha256()
    h.update(canonical_url(url).encode("utf-8", "replace"))
    if body is not None:
        if isinstance(body, (dict, list)):
            # sorted keys => stable key regardless of dict ordering
            payload = json.dumps(body, sort_keys=True, default=str)
        else:
            payload = str(body)
        h.update(b"|")
        h.update(payload.encode("utf-8", "replace"))
    return h.hexdigest()


def fixture_path(settings: Settings, key: str) -> Path:
    return Path(settings.fixture_dir) / f"{key}.json"


def read_fixture(settings: Settings, url: str, body: Any = None) -> Optional[HttpResult]:
    key = fixture_key(url, body)
    path = fixture_path(settings, key)
    if not path.exists():
        # Hand-written fixtures carry a "url" field; allow a slower exact-URL scan
        # so humans can name files readably instead of by sha256.
        alt = _scan_fixtures_for_url(settings, url, body)
        if alt is None:
            return None
        path = alt
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # corrupt fixture is a miss, not a crash
        return HttpResult(ok=False, url=url, error=f"bad fixture {path.name}: {exc}", from_fixture=True)
    headers = {str(k).lower(): str(v) for k, v in (raw.get("headers") or {}).items()}
    status = int(raw.get("status", 200))
    return HttpResult(
        ok=200 <= status < 400,
        url=raw.get("url", url),
        status=status,
        headers=headers,
        text=raw.get("text", ""),
        from_fixture=True,
        error=None if 200 <= status < 400 else f"HTTP {status} (fixture)",
    )


def _scan_fixtures_for_url(settings: Settings, url: str, body: Any) -> Optional[Path]:
    d = Path(settings.fixture_dir)
    if not d.is_dir():
        return None
    want_q = _body_query(body)
    try:
        candidates = sorted(d.glob("*.json"))
    except Exception:
        return None
    for p in candidates:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _same_url(raw.get("url", ""), url):
            if want_q and raw.get("query") and raw["query"].strip().lower() != want_q:
                continue
            return p
    return None


def _body_query(body: Any) -> str:
    if isinstance(body, dict):
        for k in ("q", "query"):
            if body.get(k):
                return str(body[k]).strip().lower()
    return ""


def _url_query(url: str) -> str:
    """The search term carried in the URL's own query string, if any.

    POST-based search providers put the term in the body; GET-based ones (Scrappa,
    Brave) put it in the URL. Both need the `query` annotation on the fixture, or
    the offline fuzzy match has nothing to compare a reworded query against.
    """
    try:
        from urllib.parse import parse_qs

        params = parse_qs(urlparse(url or "").query)
    except Exception:
        return ""
    for key in ("query", "q"):
        values = params.get(key) or []
        if values and str(values[0]).strip():
            return str(values[0]).strip().lower()
    return ""


def _same_url(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return canonical_url(a) == canonical_url(b)


# Fixtures are demo artifacts that live in the repo, not a web archive. Modern
# marketing pages inline megabytes of framework JSON; capping keeps the corpus
# reviewable. Signals buried past the cap are lost — an accepted trade.
MAX_FIXTURE_TEXT_CHARS = 300_000


def strip_inline_assets(html: str) -> str:
    """Drop inline <script>/<style>/<svg>/<noscript> bodies, keep everything else.

    An opt-in `record_transform` for recording heavyweight third-party pages: a
    news article is typically 240KB of which 140KB is inlined framework JSON and
    vector icons. What the tool layer actually reads survives untouched —
    `extract_text` already discards these tags, and `<script src=...>` elements
    are kept so tech detection still fingerprints the page.

    Not used automatically: an ordinary page fixture must keep its inline
    scripts, because half the tech-detect signatures match on them.
    """
    if not html:
        return html
    try:
        from selectolax.lexbor import LexborHTMLParser as HTMLParser

        tree = HTMLParser(html)
        for tag in ("script", "style", "noscript", "svg"):
            for node in tree.css(tag):
                if tag == "script" and node.attributes.get("src"):
                    continue  # the src is the fingerprint; the body is not
                node.decompose()
        return tree.html or html
    except Exception:
        return html


def write_fixture(settings: Settings, url: str, body: Any, result: HttpResult) -> None:
    try:
        d = Path(settings.fixture_dir)
        d.mkdir(parents=True, exist_ok=True)
        text = result.text or ""
        truncated = len(text) > MAX_FIXTURE_TEXT_CHARS
        payload = {
            "url": url,
            "status": result.status,
            "headers": result.headers,
            "text": text[:MAX_FIXTURE_TEXT_CHARS],
            "recorded_at": _now_iso(),
        }
        if truncated:
            payload["truncated_from"] = len(text)
        q = _body_query(body) or _url_query(url)
        if q:
            payload["query"] = q
        fixture_path(settings, fixture_key(url, body)).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except Exception:
        # Recording is a convenience; never let it break a live run.
        pass


# --------------------------------------------------------------------------
# robots.txt
# --------------------------------------------------------------------------

_ROBOTS_CACHE: dict[str, Optional[RobotFileParser]] = {}


def clear_robots_cache() -> None:
    _ROBOTS_CACHE.clear()


# --------------------------------------------------------------------------
# In-process GET cache
# --------------------------------------------------------------------------

# The agent routinely fetches a page and then runs detect_tech on the same URL.
# Caching successful GETs for the life of the process turns that into one request
# instead of two — politer to the site and noticeably faster.
RESPONSE_CACHE_TTL_S = 300.0
_RESPONSE_CACHE: dict[str, tuple[float, HttpResult]] = {}


def clear_response_cache() -> None:
    _RESPONSE_CACHE.clear()


def _cache_get(key: str) -> Optional[HttpResult]:
    entry = _RESPONSE_CACHE.get(key)
    if entry is None:
        return None
    stored_at, result = entry
    if time.monotonic() - stored_at > RESPONSE_CACHE_TTL_S:
        _RESPONSE_CACHE.pop(key, None)
        return None
    return result


def _cache_put(key: str, result: HttpResult) -> None:
    _RESPONSE_CACHE[key] = (time.monotonic(), result)


def robots_allows(url: str, settings: Settings) -> bool:
    """Fail open: any error fetching or parsing robots.txt means 'allowed'."""
    if not settings.respect_robots:
        return True
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return True
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        if host_key in _ROBOTS_CACHE:
            rp = _ROBOTS_CACHE[host_key]
        elif settings.offline:
            # No network in offline mode and robots is not worth a fixture.
            _ROBOTS_CACHE[host_key] = None
            rp = None
        else:
            rp = _load_robots(host_key, settings)
            _ROBOTS_CACHE[host_key] = rp
        if rp is None:
            return True
        return bool(rp.can_fetch(settings.user_agent, url))
    except Exception:
        return True


def _load_robots(host_key: str, settings: Settings) -> Optional[RobotFileParser]:
    try:
        with build_client(settings) as client:
            resp = client.get(f"{host_key}/robots.txt", timeout=min(settings.http_timeout_s, 8.0))
        if resp.status_code >= 400:
            return None
        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp
    except Exception:
        return None


# --------------------------------------------------------------------------
# The one entry point
# --------------------------------------------------------------------------


def request(
    settings: Settings,
    method: str,
    url: str,
    *,
    data: Optional[dict[str, Any]] = None,
    json_body: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    browser_ua: bool = False,
    user_agent: Optional[str] = None,
    check_robots: bool = False,
    use_cache: bool = True,
    record: bool = True,
    allow_private_hosts: Optional[bool] = None,
    timeout: Optional[float] = None,
    client: Optional[httpx.Client] = None,
    record_transform: Optional[Callable[[str], str]] = None,
) -> HttpResult:
    """Perform (or replay) one HTTP call. Never raises.

    `record_transform` shrinks the body *on its way to disk only*; the caller
    still gets the full response. An ATS board is 2.5MB of job descriptions we
    never read, and blind truncation would leave invalid JSON in the fixture —
    the caller knows which fields are dead weight, so it does the pruning.
    """
    started = time.perf_counter()
    method = method.upper()
    full_url = _with_params(url, params)
    body = json_body if json_body is not None else data
    cache_key = fixture_key(full_url, body) if (use_cache and method == "GET") else None

    # Offline reads the fixture every time, deliberately ahead of the in-process
    # cache: --offline must reflect exactly what is on disk, so a demo behaves the
    # same on its first run and its tenth.
    if settings.offline:
        hit = read_fixture(settings, full_url, body)
        if hit is None:
            return HttpResult(
                ok=False,
                url=full_url,
                error=f"offline: no fixture recorded for {full_url}",
                from_fixture=True,
                latency_ms=_ms(started),
            )
        hit.latency_ms = _ms(started)
        return hit

    if cache_key is not None:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    # Guard the URL the caller asked for, before a socket is opened.
    blocked = url_rejection(full_url, allow_private=allow_private_hosts)
    if blocked:
        return HttpResult(ok=False, url=full_url, error=blocked, latency_ms=_ms(started))

    if check_robots and not robots_allows(full_url, settings):
        return HttpResult(
            ok=False,
            url=full_url,
            error=f"robots.txt disallows fetching {full_url}",
            latency_ms=_ms(started),
        )

    owns_client = client is None
    try:
        c = client or build_client(
            settings,
            browser_ua=browser_ua,
            user_agent=user_agent,
            follow_redirects=False,
            timeout=timeout,
        )
        try:
            resp = _follow_redirects(
                c,
                method,
                url,
                data=data if json_body is None else None,
                json_body=json_body,
                headers=headers,
                params=params,
                allow_private=allow_private_hosts,
            )
        finally:
            if owns_client:
                c.close()
    except _BlockedRedirect as exc:
        return HttpResult(ok=False, url=full_url, error=str(exc), latency_ms=_ms(started))
    except Exception as exc:
        return HttpResult(
            ok=False,
            url=full_url,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=_ms(started),
            timed_out=isinstance(exc, httpx.TimeoutException),
        )

    result = HttpResult(
        ok=resp.status_code < 400,
        url=str(resp.url),
        status=resp.status_code,
        headers={k.lower(): v for k, v in resp.headers.items()},
        text=resp.text,
        error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
        latency_ms=_ms(started),
    )
    if cache_key is not None and result.ok:
        _cache_put(cache_key, result)
    # `record=False` lets a caller defer recording until it knows the body is
    # actually useful — a search provider must not cache a bot-challenge page.
    if settings.record_fixtures and record:
        write_fixture(settings, full_url, body, _for_fixture(result, record_transform))
    return result


def _for_fixture(result: HttpResult, transform: Optional[Callable[[str], str]]) -> HttpResult:
    if transform is None or not result.text:
        return result
    try:
        pruned = transform(result.text)
    except Exception:
        return result  # a bad pruner must not cost us the recording
    if not pruned or pruned == result.text:
        return result
    return replace(result, text=pruned)


def _with_params(url: str, params: Optional[dict[str, Any]]) -> str:
    if not params:
        return url
    try:
        from urllib.parse import urlencode

        parsed = urlparse(url)
        query = urlencode({k: v for k, v in sorted(params.items())})
        merged = f"{parsed.query}&{query}" if parsed.query else query
        return urlunparse(parsed._replace(query=merged))
    except Exception:
        return url


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
