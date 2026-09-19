"""Web search. DuckDuckGo's keyless HTML endpoint is the default path and the one
that has to work without any API key at all; the paid providers are upgrades.
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from selectolax.parser import HTMLParser

from lea.config import Settings
from lea.models import Observation, SearchResult, ToolName

from . import http as _http

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
TAVILY_URL = "https://api.tavily.com/search"
SERPER_URL = "https://google.serper.dev/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# --- DuckDuckGo anti-bot handling ----------------------------------------
# DDG trips a bot heuristic after a handful of quick queries. The trip is usually
# cleared by a different endpoint + UA rather than by waiting alone, so retries
# rotate both. Module-level so tests can shrink them to zero.
# Budget tradeoff, tuned against a genuinely blocked DDG: a bot-check does not
# clear by waiting, and a blocked endpoint tends to HANG rather than refuse
# quickly, so patience costs a full timeout per attempt and buys nothing. The
# degraded fixture fallback is good, so failing over FAST beats retrying.
# Transient faults (resets, 5xx) are a different case and keep a short retry.
DDG_MAX_ATTEMPTS = 3  # transient path only
DDG_RETRY_BACKOFF_S: tuple[float, ...] = (0.6, 1.5)
# A hard bot-check: try the current endpoint and the alternate once each, then
# stop. Both endpoints challenged means blocked, not unlucky.
DDG_BLOCK_MAX_ATTEMPTS = 2
# Deliberately below settings.http_timeout_s: a hanging search should surrender
# to the cache quickly rather than stall a live demo.
DDG_REQUEST_TIMEOUT_S = 6.0
DDG_MIN_REQUEST_GAP_S = 1.0
DDG_JITTER = 0.25  # +/- fraction applied to every sleep

# Strings DDG serves on its challenge/anomaly page instead of results.
DDG_BLOCK_MARKERS: tuple[str, ...] = (
    "anomaly",
    "unusual traffic",
    "are you a robot",
    "blocked",
    "captcha",
    "challenge-platform",
    "please try again later",
    "detected unusual activity",
)
DDG_SHORT_BODY_BYTES = 800  # a real results page is tens of kilobytes

BLOCKED_ADVICE = "retry in a minute, set TAVILY_API_KEY, or run with --offline"

_last_ddg_request_at: float = 0.0


def _sleep(seconds: float) -> None:
    """Indirection so tests can neutralize backoff without patching time.sleep globally."""
    if seconds > 0:
        time.sleep(seconds)


def _jittered(seconds: float) -> float:
    return seconds * random.uniform(1.0 - DDG_JITTER, 1.0 + DDG_JITTER)


def reset_pacing() -> None:
    """Forget the last-request timestamp (tests, and any long-idle process)."""
    global _last_ddg_request_at
    _last_ddg_request_at = 0.0


def _pace(settings: Settings) -> None:
    """Self-throttle outbound DDG calls.

    A batch run firing queries back to back is exactly what trips the bot check,
    so spacing them slightly is cheaper than getting blocked and retrying.
    """
    global _last_ddg_request_at
    if settings.offline or DDG_MIN_REQUEST_GAP_S <= 0:
        return
    now = time.monotonic()
    if _last_ddg_request_at:
        wait = _jittered(DDG_MIN_REQUEST_GAP_S) - (now - _last_ddg_request_at)
        if wait > 0:
            _sleep(wait)
    _last_ddg_request_at = time.monotonic()


def ddg_block_reason(res: "_http.HttpResult") -> Optional[str]:
    """Name the block if this response is one, else None.

    Distinguishing a block from a genuine zero-result query matters: one is worth
    retrying on another endpoint, the other is not.
    """
    # A blocked DDG often stalls the connection instead of refusing it. Hammering
    # an unresponsive endpoint costs a full timeout per try and fixes nothing, so
    # this counts as a block: try the other backend once, then use the cache.
    if getattr(res, "timed_out", False):
        return "endpoint did not respond (timed out)"
    if res.status in (202, 403, 429):
        label = {202: "HTTP 202 accepted-but-empty", 403: "HTTP 403 forbidden", 429: "HTTP 429 rate limited"}
        return label[res.status]
    if res.status >= 500:
        return f"HTTP {res.status} upstream error"
    body = res.text or ""
    head = body[:4000].lower()
    for marker in DDG_BLOCK_MARKERS:
        if marker in head:
            return f"challenge page (matched '{marker}')"
    if res.ok and len(body.strip()) < DDG_SHORT_BODY_BYTES:
        return f"suspiciously short response ({len(body.strip())} bytes)"
    return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def unwrap_ddg_href(href: str) -> str:
    """DDG wraps every result in a redirector: //duckduckgo.com/l/?uddg=<urlencoded>.

    Handing that wrapper to the fetch tool would fetch DuckDuckGo, not the company,
    so the real target has to be pulled back out of the `uddg` query parameter.
    """
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
    except Exception:
        return href
    if "duckduckgo.com" in (parsed.netloc or "") and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        target = (qs.get("uddg") or [""])[0]
        if target:
            return unquote(target)
    # Some layouts emit the wrapper without a host, e.g. "/l/?uddg=..."
    if parsed.path.startswith("/l/") and "uddg=" in (parsed.query or ""):
        target = (parse_qs(parsed.query).get("uddg") or [""])[0]
        if target:
            return unquote(target)
    return href


def _clean(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def _is_usable(url: str) -> bool:
    """Drop anything still pointing at DuckDuckGo.

    Sponsored slots are `duckduckgo.com/y.js?ad_domain=...` redirects with no uddg
    parameter to unwrap. They are ads, not research, and their URLs are enormous.
    """
    if not url.startswith(("http://", "https://")):
        return False
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    return not host.endswith("duckduckgo.com")


def _dedupe(results: list[SearchResult], limit: int) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if not r.url or not _is_usable(r.url):
            continue
        key = r.url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------


def _result_containers(tree: HTMLParser) -> list:
    out: list = []
    for selector in ("div.result", "div.web-result", "div.results_links"):
        out.extend(tree.css(selector))
    return out


def parse_ddg_html(html: str) -> list[SearchResult]:
    """Parse the html.duckduckgo.com/html/ layout: a.result__a + a.result__snippet."""
    out: list[SearchResult] = []
    if not html:
        return out
    tree = HTMLParser(html)
    seen_nodes: set[int] = set()
    for node in _result_containers(tree):
        # One container carries several of these classes at once; visit it once.
        if node.mem_id in seen_nodes:
            continue
        seen_nodes.add(node.mem_id)
        link = node.css_first("a.result__a")
        if link is None:
            continue
        url = unwrap_ddg_href(link.attributes.get("href", "") or "")
        snippet_node = node.css_first("a.result__snippet") or node.css_first(".result__snippet")
        out.append(
            SearchResult(
                title=_clean(link.text()),
                url=url,
                snippet=_clean(snippet_node.text() if snippet_node else ""),
            )
        )
    if out:
        return out
    # Fallback for markup variants where results are not wrapped in div.result.
    for link in tree.css("a.result__a"):
        url = unwrap_ddg_href(link.attributes.get("href", "") or "")
        out.append(SearchResult(title=_clean(link.text()), url=url, snippet=""))
    return out


def parse_ddg_lite(html: str) -> list[SearchResult]:
    """Parse lite.duckduckgo.com/lite/: a flat table of a.result-link rows followed by
    a td.result-snippet row."""
    out: list[SearchResult] = []
    if not html:
        return out
    tree = HTMLParser(html)
    links = tree.css("a.result-link")
    snippets = [_clean(td.text()) for td in tree.css("td.result-snippet")]
    for i, link in enumerate(links):
        url = unwrap_ddg_href(link.attributes.get("href", "") or "")
        out.append(
            SearchResult(
                title=_clean(link.text()),
                url=url,
                snippet=snippets[i] if i < len(snippets) else "",
            )
        )
    if out:
        return out
    # Older lite markup uses bare links inside the results table.
    for link in tree.css("a[href]"):
        href = link.attributes.get("href", "") or ""
        if "uddg=" not in href:
            continue
        out.append(SearchResult(title=_clean(link.text()), url=unwrap_ddg_href(href), snippet=""))
    return out


def parse_tavily(payload: dict[str, Any]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for item in (payload or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        out.append(
            SearchResult(
                title=_clean(item.get("title")),
                url=(item.get("url") or "").strip(),
                snippet=_clean(item.get("content")),
            )
        )
    return out


def parse_serper(payload: dict[str, Any]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for item in (payload or {}).get("organic") or []:
        if not isinstance(item, dict):
            continue
        out.append(
            SearchResult(
                title=_clean(item.get("title")),
                url=(item.get("link") or "").strip(),
                snippet=_clean(item.get("snippet")),
            )
        )
    return out


def parse_brave(payload: dict[str, Any]) -> list[SearchResult]:
    out: list[SearchResult] = []
    web = (payload or {}).get("web") or {}
    for item in web.get("results") or []:
        if not isinstance(item, dict):
            continue
        out.append(
            SearchResult(
                title=_clean(item.get("title")),
                url=(item.get("url") or "").strip(),
                snippet=_clean(item.get("description")),
            )
        )
    return out


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


class WebSearchTool:
    name = ToolName.WEB_SEARCH
    description = (
        "Search the web for a company, its funding, hiring, news or people. "
        "Returns a numbered list of titles, URLs and snippets. Use targeted queries "
        "(e.g. 'Ramp Series D funding 2024') rather than bare company names."
    )
    args_schema: dict[str, Any] = {
        "query": {"type": "string", "required": True, "description": "Search query."},
        "num_results": {
            "type": "integer",
            "required": False,
            "default": 5,
            "description": "How many results to return (1-10).",
        },
    }

    def run(self, args: dict[str, Any], settings: Settings) -> Observation:
        started = time.perf_counter()
        provider = "unknown"
        try:
            query = str((args or {}).get("query") or "").strip()
            if not query:
                return self._fail("missing required arg 'query'", started, {"provider": provider})
            num = _coerce_int((args or {}).get("num_results"), 5)
            num = max(1, min(num, 10))
            provider = settings.resolved_search()

            if provider == "duckduckgo":
                results, err = self._duckduckgo(query, num, settings)
            elif provider == "tavily":
                results, err = self._tavily(query, num, settings)
            elif provider == "serper":
                results, err = self._serper(query, num, settings)
            elif provider == "brave":
                results, err = self._brave(query, num, settings)
            elif provider == "fixture":
                results, err = self._fixture(query, num, settings)
            else:
                return self._fail(f"unknown search provider '{provider}'", started, {"provider": provider})

            results = _dedupe(results, num) if results else []

            # Last resort before failing: if this query was ever recorded, stale
            # results beat no results. A blocked demo with a populated cache should
            # still produce a record.
            if (err or not results) and provider != "fixture":
                cached, _ = self._fixture(query, num, settings)
                cached = _dedupe(cached, num)
                if cached:
                    reason = err or f"{provider} returned no results"
                    note = (
                        f"[cached] {provider} unavailable ({reason}); "
                        f"served {len(cached)} result(s) from a recorded fixture."
                    )
                    return Observation(
                        tool=self.name,
                        ok=True,
                        summary=note + "\n" + _render(cached, settings.max_snippet_chars),
                        data={
                            "provider": provider,
                            "query": query,
                            "results": [r.model_dump() for r in cached],
                            "degraded": True,
                            "degraded_reason": reason,
                            "served_from": "fixture",
                        },
                        sources=[r.url for r in cached],
                        latency_ms=_ms(started),
                    )

            if err:
                return self._fail(err, started, {"provider": provider, "query": query})
            if not results:
                return self._fail(
                    f"no results for '{query}' via {provider} — try a shorter or differently worded query",
                    started,
                    {"provider": provider, "query": query, "results": []},
                )

            urls = [r.url for r in results]
            return Observation(
                tool=self.name,
                ok=True,
                summary=_render(results, settings.max_snippet_chars),
                data={
                    "provider": provider,
                    "query": query,
                    "results": [r.model_dump() for r in results],
                    "degraded": False,
                },
                sources=urls,
                latency_ms=_ms(started),
            )
        except Exception as exc:  # last-resort net: the loop must never see a traceback
            return self._fail(f"{type(exc).__name__}: {exc}", started, {"provider": provider})

    # -- providers -------------------------------------------------------

    def _duckduckgo(
        self, query: str, num: int, settings: Settings
    ) -> tuple[list[SearchResult], Optional[str]]:
        """Try up to DDG_MAX_ATTEMPTS times, alternating endpoint and User-Agent.

        The html and lite endpoints are separate backends with separate layouts and
        separate bot heuristics, so a trip on one frequently clears on the other.
        """
        blocked_reasons: list[str] = []
        last_error: Optional[str] = None
        hard_blocks = 0

        for attempt in range(DDG_MAX_ATTEMPTS):
            use_lite = attempt % 2 == 1
            url = DDG_LITE_URL if use_lite else DDG_HTML_URL
            # Same index into both lists would pair endpoint and UA forever; the
            # offset keeps the combinations varying across attempts.
            ua = _http.BROWSER_UAS[attempt % len(_http.BROWSER_UAS)]
            form = {"q": query} if use_lite else {"q": query, "kl": "us-en"}
            referer = "https://lite.duckduckgo.com/" if use_lite else "https://duckduckgo.com/"

            _pace(settings)
            res = _http.request(
                settings,
                "POST",
                url,
                data=form,
                user_agent=ua,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Referer": referer,
                },
                record=False,
                timeout=DDG_REQUEST_TIMEOUT_S,
            )

            # Parse before judging: a page that yields results is a success no matter
            # how short it is or what words happen to appear in it.
            results = parse_ddg_lite(res.text) if use_lite else parse_ddg_html(res.text)
            if results:
                # Only now is this body worth keeping. Recording unconditionally would
                # persist challenge pages and poison every later --offline run.
                if settings.record_fixtures:
                    _http.write_fixture(settings, url, form, res)
                return results, None

            reason = ddg_block_reason(res)
            if reason is not None:
                blocked_reasons.append(f"{_host(url)}: {reason}")
                last_error = reason
            elif not res.ok:
                last_error = res.error
            else:
                last_error = "no parseable results"

            # Offline runs have no second chance to win — the fixture either matched
            # or it did not, and sleeping would just stall the demo.
            if settings.offline:
                break

            if reason is not None:
                hard_blocks += 1
                if hard_blocks >= DDG_BLOCK_MAX_ATTEMPTS:
                    break  # both endpoints challenged: hand over to the cache now
                continue  # the alternate endpoint is a different backend — go straight there

            if attempt < DDG_MAX_ATTEMPTS - 1:
                _sleep(_jittered(DDG_RETRY_BACKOFF_S[min(attempt, len(DDG_RETRY_BACKOFF_S) - 1)]))

        if blocked_reasons:
            detail = "; ".join(dict.fromkeys(blocked_reasons))
            return [], (
                f"duckduckgo temporarily blocked this client (bot check — {detail}) — {BLOCKED_ADVICE}"
            )
        if last_error and last_error != "no parseable results":
            return [], f"duckduckgo request failed: {last_error} — {BLOCKED_ADVICE}"
        return [], (
            f"duckduckgo returned no parseable results for '{query}' after "
            f"{DDG_MAX_ATTEMPTS} attempts across both endpoints — {BLOCKED_ADVICE}"
        )

    def _tavily(self, query: str, num: int, settings: Settings) -> tuple[list[SearchResult], Optional[str]]:
        if not settings.tavily_api_key and not settings.offline:
            return [], "tavily selected but TAVILY_API_KEY is not set"
        res = _http.request(
            settings,
            "POST",
            TAVILY_URL,
            json_body={
                "api_key": settings.tavily_api_key or "",
                "query": query,
                "max_results": num,
                "search_depth": "basic",
            },
        )
        if not res.ok:
            return [], f"tavily: {res.error}"
        payload = _json(res.text)
        if payload is None:
            return [], "tavily: response was not JSON"
        return parse_tavily(payload), None

    def _serper(self, query: str, num: int, settings: Settings) -> tuple[list[SearchResult], Optional[str]]:
        if not settings.serper_api_key and not settings.offline:
            return [], "serper selected but SERPER_API_KEY is not set"
        res = _http.request(
            settings,
            "POST",
            SERPER_URL,
            json_body={"q": query, "num": num},
            headers={
                "X-API-KEY": settings.serper_api_key or "",
                "Content-Type": "application/json",
            },
        )
        if not res.ok:
            return [], f"serper: {res.error}"
        payload = _json(res.text)
        if payload is None:
            return [], "serper: response was not JSON"
        return parse_serper(payload), None

    def _brave(self, query: str, num: int, settings: Settings) -> tuple[list[SearchResult], Optional[str]]:
        if not settings.brave_api_key and not settings.offline:
            return [], "brave selected but BRAVE_API_KEY is not set"
        res = _http.request(
            settings,
            "GET",
            BRAVE_URL,
            params={"q": query, "count": num},
            headers={
                "X-Subscription-Token": settings.brave_api_key or "",
                "Accept": "application/json",
            },
        )
        if not res.ok:
            return [], f"brave: {res.error}"
        payload = _json(res.text)
        if payload is None:
            return [], "brave: response was not JSON"
        return parse_brave(payload), None

    def _fixture(self, query: str, num: int, settings: Settings) -> tuple[list[SearchResult], Optional[str]]:
        """Fixture-only mode: try every recorded search endpoint for this query."""
        for url, body, parser in (
            (DDG_HTML_URL, {"q": query, "kl": "us-en"}, "ddg_html"),
            (DDG_HTML_URL, {"q": query}, "ddg_html"),
            (DDG_LITE_URL, {"q": query}, "ddg_lite"),
            (TAVILY_URL, {"query": query}, "tavily"),
            (SERPER_URL, {"q": query}, "serper"),
        ):
            hit = _http.read_fixture(settings, url, body)
            if hit is None or not hit.text:
                continue
            if parser == "ddg_html":
                results = parse_ddg_html(hit.text)
            elif parser == "ddg_lite":
                results = parse_ddg_lite(hit.text)
            elif parser == "tavily":
                results = parse_tavily(_json(hit.text) or {})
            else:
                results = parse_serper(_json(hit.text) or {})
            if results:
                return results, None

        # Exact-key lookup missed. An LLM rarely reproduces a recorded query word
        # for word, so fall back to the closest recorded search for this endpoint —
        # the difference between a useful offline demo and an empty one.
        near = _best_recorded_search(settings, query)
        if near is not None:
            text, kind = near
            results = parse_ddg_lite(text) if kind == "ddg_lite" else parse_ddg_html(text)
            if results:
                return results, None
        return [], f"fixture: no recorded search results for '{query}'"

    # -- plumbing --------------------------------------------------------

    def _fail(self, error: str, started: float, data: dict[str, Any]) -> Observation:
        return Observation(
            tool=self.name,
            ok=False,
            summary=f"search failed: {error}",
            data={**data, "results": data.get("results", []), "degraded": False},
            error=error,
            latency_ms=_ms(started),
        )


# Pure grammar: no signal either way.
_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "for", "and", "or", "in", "on", "at", "to", "is", "s", "inc", "co", "ltd"}
)

# Words that describe the KIND of research rather than its subject. Two queries
# sharing only these — "Ramp careers" vs "vanta careers hiring" — are about
# different companies and must never match each other.
_GENERIC_TOKENS = frozenset(
    {
        "company", "companies", "overview", "about", "official", "site", "website", "homepage",
        "profile", "info", "information", "research",
        "funding", "funded", "raise", "raised", "round", "series", "valuation", "investors",
        "investment", "capital", "revenue", "arr",
        "careers", "career", "hiring", "hire", "jobs", "job", "roles", "openings", "recruiting",
        "employees", "headcount", "team", "leadership", "ceo", "founder", "founders",
        "news", "announcement", "press", "blog", "latest", "recent", "update", "updates",
        "headquarters", "hq", "location", "based", "industry", "customers", "pricing", "product",
        "platform", "software", "startup", "tech", "stack",
        "2023", "2024", "2025", "2026",
    }
)

# The distinctive-token gate below does the real work; this only trims noise.
_FIXTURE_MATCH_THRESHOLD = 0.2


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in "".join(c if c.isalnum() else " " for c in text.lower()).split()
        if t and t not in _STOPWORDS
    }


def _match_score(wanted: set[str], recorded: set[str]) -> float:
    """How well a recorded query answers `wanted`. 0.0 means "do not use it".

    Gated on sharing a *distinctive* token — in practice the company name.
    Scored by overlap coefficient rather than Jaccard, so a short query ("Vanta
    careers") still matches a longer recorded one ("vanta careers hiring")
    instead of being punished for the words it did not say.
    """
    shared = wanted & recorded
    if not shared:
        return 0.0
    if not (shared - _GENERIC_TOKENS):
        return 0.0  # only research-verb words in common: different subject
    score = len(shared) / max(1, min(len(wanted), len(recorded)))
    return score if score >= _FIXTURE_MATCH_THRESHOLD else 0.0


def _best_recorded_search(settings: Settings, query: str) -> Optional[tuple[str, str]]:
    """Closest recorded DDG search for `query`, as (html_text, parser_kind)."""
    import json as _json_mod
    from pathlib import Path as _Path

    wanted = _tokens(query)
    if not wanted:
        return None
    best: Optional[tuple[float, str, str]] = None
    try:
        candidates = sorted(_Path(settings.fixture_dir).glob("*.json"))
    except Exception:
        return None

    for path in candidates:
        try:
            raw = _json_mod.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = raw.get("url", "")
        if url == DDG_LITE_URL:
            kind = "ddg_lite"
        elif url == DDG_HTML_URL:
            kind = "ddg_html"
        else:
            continue
        recorded = _tokens(raw.get("query", ""))
        if not recorded:
            continue
        score = _match_score(wanted, recorded)
        if score > 0 and (best is None or score > best[0]):
            best = (score, raw.get("text", ""), kind)

    return (best[1], best[2]) if best else None


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def _render(results: list[SearchResult], max_snippet: int) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title or '(no title)'} — {r.url}")
        if r.snippet:
            snip = r.snippet
            if len(snip) > max_snippet:
                snip = snip[:max_snippet].rstrip() + "…"
            lines.append(f"   {snip}")
    return "\n".join(lines)


def _json(text: str) -> Optional[dict[str, Any]]:
    import json as _json_mod

    try:
        payload = _json_mod.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
