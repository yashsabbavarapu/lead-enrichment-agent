"""Shared HTTP plumbing: one client factory, a JSON fixture cache, and a robots.txt gate.

Every network call in the tool layer goes through `request()` so that `--offline`
is a hard guarantee (no socket is ever opened) and `--record-fixtures` captures a
replayable corpus for demos and tests.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

from lea.config import Settings

# DuckDuckGo (and a few CDNs) 403 anything that looks like a script. A plain
# desktop Chrome UA is the difference between results and an empty page.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


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


def build_client(settings: Settings, *, browser_ua: bool = False) -> httpx.Client:
    """One sync client honoring Settings. Callers own closing it (or use `request`)."""
    ua = BROWSER_UA if browser_ua else settings.user_agent
    return httpx.Client(
        timeout=settings.http_timeout_s,
        follow_redirects=True,
        headers={
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


# --------------------------------------------------------------------------
# Fixture cache
# --------------------------------------------------------------------------


def fixture_key(url: str, body: Any = None) -> str:
    """sha256 over URL (+ body for POST) so the same request always maps to one file."""
    h = hashlib.sha256()
    h.update(url.encode("utf-8", "replace"))
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


def _same_url(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a.rstrip("/").lower() == b.rstrip("/").lower()


def write_fixture(settings: Settings, url: str, body: Any, result: HttpResult) -> None:
    try:
        d = Path(settings.fixture_dir)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "url": url,
            "status": result.status,
            "headers": result.headers,
            "text": result.text,
            "recorded_at": _now_iso(),
        }
        q = _body_query(body)
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
    check_robots: bool = False,
    client: Optional[httpx.Client] = None,
) -> HttpResult:
    """Perform (or replay) one HTTP call. Never raises."""
    started = time.perf_counter()
    method = method.upper()
    full_url = _with_params(url, params)
    body = json_body if json_body is not None else data

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

    if check_robots and not robots_allows(full_url, settings):
        return HttpResult(
            ok=False,
            url=full_url,
            error=f"robots.txt disallows fetching {full_url}",
            latency_ms=_ms(started),
        )

    owns_client = client is None
    try:
        c = client or build_client(settings, browser_ua=browser_ua)
        try:
            resp = c.request(
                method,
                url,
                data=data if json_body is None else None,
                json=json_body,
                headers=headers,
                params=params,
            )
        finally:
            if owns_client:
                c.close()
    except Exception as exc:
        return HttpResult(
            ok=False,
            url=full_url,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=_ms(started),
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
    if settings.record_fixtures:
        write_fixture(settings, full_url, body, result)
    return result


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
