"""Web search. DuckDuckGo's keyless HTML endpoint is the default path and the one
that has to work without any API key at all; the paid providers are upgrades.
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from lea.config import Settings
from lea.models import Observation, SearchResult, ToolName

from . import http as _http

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
TAVILY_URL = "https://api.tavily.com/search"
SERPER_URL = "https://google.serper.dev/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
SCRAPPA_URL = "https://scrappa.co/api/search"

# --- Scrappa --------------------------------------------------------------
# Scrappa proxies a live Google SERP, so it is slow by nature: 5-7s warm, and a
# cold start has been measured past 25s. settings.http_timeout_s is 15s because
# it is sized for page fetches, and using it here would guillotine exactly the
# calls we most want to wait for. Raising the global default instead would make
# every page fetch sluggish, so this provider carries its own budget.
SCRAPPA_TIMEOUT_S = 45.0

# Credits are finite — 500/month, one per SUCCESSFUL request. Measured against
# the account endpoint: nine calls returning four 200s and five 503s cost exactly
# four credits, matching Scrappa's documented "failed responses including 503 are
# never charged". So a failure costs TIME, not money — which is what the retry
# and fallback policy below is actually optimising against.
#
# A 200 carrying zero results IS an answer; asking again buys the same answer for
# another credit, so that case is never retried. A 5xx or a timeout returned no
# answer at all, and that alone is worth exactly one more try.
SCRAPPA_SERVER_RETRIES = 1
SCRAPPA_RETRY_BACKOFF_S = 2.0

# ...but only when the fault arrived quickly. Latency here is sharply bimodal:
# every success measured landed in 5.8-6.9s, every 503 in 41-46s. Retrying a
# fault that slow turned a single search step into 91s and consumed an entire
# 180s run budget in two steps. Past ~12s the call is almost certainly a 503
# already; waiting for a second one beats nothing.
SCRAPPA_FAST_FAULT_S = 12.0

# Account balance. Free to call, never charged, and exact.
SCRAPPA_USAGE_URL = "https://scrappa.co/api/account/usage"

# When the primary provider fails transiently, try these rather than lose the
# step. Keyless only, deliberately: silently failing over to another *paid*
# provider would spend a budget nobody authorised for this query.
#
# Worth it because Scrappa's 503s are slow and, in a bad window, common — five of
# nine in one measured run, uncorrelated with query shape or request rate.
# Without this the agent burns a step and 43 seconds and learns nothing.
FALLBACK_PROVIDERS: dict[str, tuple[str, ...]] = {
    "scrappa": ("duckduckgo",),
}

_PROVIDERS = frozenset({"scrappa", "duckduckgo", "tavily", "serper", "brave", "fixture"})

# Failures worth trying another provider for: the upstream had a bad moment.
# A 401/422/missing-key is a configuration fault — the same query will fail the
# same way forever, and quietly succeeding elsewhere only hides the real problem.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "no response within", "upstream error", "rate limited", "out of monthly credits",
    "http 5", "timed out", "temporarily", "returned no results",
)


def is_transient_search_error(error: Optional[str]) -> bool:
    """Is this failure the provider having a bad moment, or a broken setup?"""
    low = (error or "").lower()
    if not low:
        return False
    if "api_key" in low or "422" in low or "key rejected" in low:
        return False
    return any(marker in low for marker in _TRANSIENT_MARKERS)


def scrappa_balance(settings: Settings) -> Optional[dict[str, Any]]:
    """Remaining Scrappa credits, or None if they cannot be read. Never raises.

    Free to call and never charged, so it is safe to show before a batch run.
    Reports `used_this_month` alongside the balance, because "479 left" means
    more once you know only 21 went this month.
    """
    if settings.offline or not settings.scrappa_api_key:
        return None
    res = _http.request(
        settings,
        "GET",
        SCRAPPA_USAGE_URL,
        headers={"x-api-key": settings.scrappa_api_key, "Accept": "application/json"},
        use_cache=False,
        record=False,
        timeout=15.0,
    )
    payload = _json(res.text) if res.ok else None
    if not payload:
        return None
    credits = payload.get("credits") if isinstance(payload.get("credits"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    balance = payload.get("balance")
    if not isinstance(balance, (int, float)):
        balance = credits.get("usable")
    if not isinstance(balance, (int, float)) or isinstance(balance, bool):
        return None
    return {
        "balance": int(balance),
        "used_this_month": int(usage.get("current_month") or credits.get("free_used_this_month") or 0),
        "used_last_24h": int(usage.get("last_24_hours") or 0),
    }

# Google's panel keys, in the order they are worth reading, mapped to labels for
# the observation text. Everything else the panel carries (*_links companions,
# image blobs, profile lists) is navigation furniture, not firmographics.
SCRAPPA_KG_FIELDS: tuple[tuple[str, str], ...] = (
    ("type", "Type"),
    ("entity_type", "Type"),
    ("description", "Description"),
    ("founded", "Founded"),
    ("headquarters", "Headquarters"),
    ("number_of_employees", "Employees"),
    ("revenue", "Revenue"),
    ("industry", "Industry"),
    ("ceo", "CEO"),
    ("cto", "CTO"),
    ("founders", "Founders"),
    ("parent_organization", "Parent organization"),
    ("subsidiaries", "Subsidiaries"),
    ("traded_as", "Traded as"),
    ("stock_price", "Stock price"),
    ("website", "Website"),
)

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


def parse_scrappa(payload: dict[str, Any]) -> list[SearchResult]:
    """Map Scrappa's `organic_results` onto the common shape.

    `link` is the destination; `redirect_link` is a google.com/goto tracking
    wrapper. Handing the wrapper downstream would have the fetch tool retrieving
    Google instead of the company — the same trap as DuckDuckGo's `uddg`
    redirector, arriving through a different door.
    """
    out: list[SearchResult] = []
    for item in (payload or {}).get("organic_results") or []:
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


def _kg_value(value: Any) -> str:
    """Flatten one knowledge-panel value to text, or '' if it is not worth text."""
    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, dict):
        return _clean(str(value.get("name") or value.get("text") or ""))
    if isinstance(value, list):
        parts = [_kg_value(v) for v in value[:6]]
        return ", ".join(p for p in parts if p)
    return ""


def parse_scrappa_knowledge_graph(payload: dict[str, Any]) -> dict[str, str]:
    """The firmographic fields out of Google's knowledge panel, when there is one.

    Genuinely absent much of the time — Scrappa returns `null`, not an empty dict —
    and nothing here invents a field that Google did not show.
    """
    kg = (payload or {}).get("knowledge_graph")
    if not isinstance(kg, dict):
        return {}
    out: dict[str, str] = {}
    title = _kg_value(kg.get("title"))
    if title:
        out["title"] = title
    for key, label in SCRAPPA_KG_FIELDS:
        if label in out:
            continue  # first spelling wins (type before entity_type)
        text = _kg_value(kg.get(key))
        if text:
            out[label] = text
    return out


def render_knowledge_graph(kg: dict[str, str]) -> str:
    """One compact block, placed above the results so extraction reads it first."""
    fields = [(k, v) for k, v in kg.items() if k != "title"]
    if not fields:
        return ""
    who = kg.get("title") or "this company"
    return f"Google knowledge panel for {who} — " + "; ".join(f"{k}: {v}" for k, v in fields)


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
        "Returns a numbered list of titles, URLs and snippets, and — when the "
        "provider is a Google SERP and the company is well known — a knowledge "
        "panel stating founded date, headquarters, headcount and CEO outright.\n"
        "Keep queries SHORT and anchored on the company name plus at most two "
        "words for the angle: 'Vanta funding round', 'Ramp headquarters', "
        "'Datadog'. Measured behaviour, not style advice — a long keyword-stuffed "
        "query ('Vanta SOC 2 compliance automation software company') loses the "
        "brand anchor and comes back with generic topic pages about SOC teams, "
        "while 'Vanta funding round' returns the company's own announcement, the "
        "press release and the CNBC story. If a company name is also a common "
        "word (Ramp, Linear, Front), add ONE disambiguator — 'Ramp fintech', "
        "'Linear issue tracker' — or search its domain instead."
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

            # Side channel for anything a provider returns that is richer than a
            # list of links. Only Scrappa fills it (Google's knowledge panel), and
            # keeping it out of the (results, error) tuple leaves every other
            # provider's signature alone.
            extras: dict[str, Any] = {}

            if provider not in _PROVIDERS:
                return self._fail(f"unknown search provider '{provider}'", started, {"provider": provider})

            results, err = self._dispatch(provider, query, num, settings, extras)
            chain = [provider]

            # A slow-failing primary with a keyless alternative available is the
            # one place a second attempt clearly pays: the agent only gets so many
            # steps, and losing one to somebody else's bad minute is pure waste.
            if (err or not results) and is_transient_search_error(err) and not settings.offline:
                for alternate in FALLBACK_PROVIDERS.get(provider, ()):
                    alt_results, alt_err = self._dispatch(alternate, query, num, settings, extras)
                    chain.append(alternate)
                    if alt_results:
                        results, err, provider = alt_results, None, alternate
                        break
                    err = err or alt_err

            results = _dedupe(results, num) if results else []

            # Last resort before failing: if this query was ever recorded, stale
            # results beat no results. A blocked demo with a populated cache should
            # still produce a record.
            if (err or not results) and provider != "fixture":
                cached_extras: dict[str, Any] = {}
                cached, _ = self._fixture(query, num, settings, cached_extras)
                cached = _dedupe(cached, num)
                if cached:
                    reason = err or f"{provider} returned no results"
                    note = (
                        f"[cached] {provider} unavailable ({reason}); "
                        f"served {len(cached)} result(s) from a recorded fixture."
                    )
                    body, kg, kg_source = _with_knowledge_graph(
                        _render(cached, settings.max_snippet_chars), cached_extras
                    )
                    return Observation(
                        tool=self.name,
                        ok=True,
                        summary=note + "\n" + body,
                        data={
                            "provider": provider,
                            "query": query,
                            "results": [r.model_dump() for r in cached],
                            "provider_chain": chain,
                            "degraded": True,
                            "degraded_reason": reason,
                            "served_from": "fixture",
                            "knowledge_graph": kg,
                        },
                        sources=[r.url for r in cached] + kg_source,
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
            body, kg, kg_source = _with_knowledge_graph(
                _render(results, settings.max_snippet_chars), extras
            )
            return Observation(
                tool=self.name,
                ok=True,
                summary=body,
                data={
                    "provider": provider,
                    "provider_chain": chain,
                    "query": query,
                    "results": [r.model_dump() for r in results],
                    "degraded": False,
                    "knowledge_graph": kg,
                },
                sources=urls + kg_source,
                latency_ms=_ms(started),
            )
        except Exception as exc:  # last-resort net: the loop must never see a traceback
            return self._fail(f"{type(exc).__name__}: {exc}", started, {"provider": provider})

    # -- providers -------------------------------------------------------

    def _dispatch(
        self, provider: str, query: str, num: int, settings: Settings, extras: dict[str, Any]
    ) -> tuple[list[SearchResult], Optional[str]]:
        """Route to one provider. Separate so the fallback chain can reuse it."""
        if provider == "scrappa":
            return self._scrappa(query, num, settings, extras)
        if provider == "duckduckgo":
            return self._duckduckgo(query, num, settings)
        if provider == "tavily":
            return self._tavily(query, num, settings)
        if provider == "serper":
            return self._serper(query, num, settings)
        if provider == "brave":
            return self._brave(query, num, settings)
        if provider == "fixture":
            return self._fixture(query, num, settings, extras)
        return [], f"unknown search provider '{provider}'"

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

    def _scrappa(
        self, query: str, num: int, settings: Settings, extras: dict[str, Any]
    ) -> tuple[list[SearchResult], Optional[str]]:
        """Google SERP via Scrappa. One credit per call — see SCRAPPA_SERVER_RETRIES."""
        if not settings.scrappa_api_key and not settings.offline:
            return [], "scrappa selected but SCRAPPA_API_KEY is not set"

        res = None
        for attempt in range(SCRAPPA_SERVER_RETRIES + 1):
            attempt_started = time.monotonic()
            res = _http.request(
                settings,
                "GET",
                SCRAPPA_URL,
                # The API rejects `q` with a 422; the field is named `query`.
                params={"query": query},
                headers={
                    "x-api-key": settings.scrappa_api_key or "",
                    "Accept": "application/json",
                },
                timeout=SCRAPPA_TIMEOUT_S,
            )
            server_fault = res.timed_out or res.status >= 500
            if not server_fault or settings.offline or attempt == SCRAPPA_SERVER_RETRIES:
                break
            if time.monotonic() - attempt_started > SCRAPPA_FAST_FAULT_S:
                break  # a slow fault: give the run its time back, not another wait
            _sleep(_jittered(SCRAPPA_RETRY_BACKOFF_S))

        if res is None:  # unreachable; keeps the type checker and the reader honest
            return [], "scrappa: no response"
        if not res.ok:
            return [], _scrappa_error(res)

        payload = _json(res.text)
        if payload is None:
            return [], "scrappa: response was not JSON"

        extras["knowledge_graph"] = parse_scrappa_knowledge_graph(payload)
        extras["search_url"] = _clean(payload.get("search_url"))
        return parse_scrappa(payload), None

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

    def _fixture(
        self,
        query: str,
        num: int,
        settings: Settings,
        extras: Optional[dict[str, Any]] = None,
    ) -> tuple[list[SearchResult], Optional[str]]:
        """Fixture-only mode: try every recorded search endpoint for this query."""
        extras = extras if extras is not None else {}
        for url, body, parser in (
            (scrappa_request_url(query), None, "scrappa"),
            (DDG_HTML_URL, {"q": query, "kl": "us-en"}, "ddg_html"),
            (DDG_HTML_URL, {"q": query}, "ddg_html"),
            (DDG_LITE_URL, {"q": query}, "ddg_lite"),
            (TAVILY_URL, {"query": query}, "tavily"),
            (SERPER_URL, {"q": query}, "serper"),
        ):
            hit = _http.read_fixture(settings, url, body)
            if hit is None or not hit.text:
                continue
            results = _parse_recorded(hit.text, parser, extras)
            if results:
                return results, None

        # Exact-key lookup missed. An LLM rarely reproduces a recorded query word
        # for word, so fall back to the closest recorded search for this endpoint —
        # the difference between a useful offline demo and an empty one.
        near = _best_recorded_search(settings, query)
        if near is not None:
            text, kind = near
            results = _parse_recorded(text, kind, extras)
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


def scrappa_request_url(query: str) -> str:
    """The exact URL `_http.request` will build, so fixture keys line up."""
    from urllib.parse import urlencode

    return f"{SCRAPPA_URL}?{urlencode({'query': query})}"


def _parse_recorded(text: str, kind: str, extras: dict[str, Any]) -> list[SearchResult]:
    """Replay one recorded search body with the parser that matches its provider."""
    if kind == "scrappa":
        payload = _json(text) or {}
        extras["knowledge_graph"] = parse_scrappa_knowledge_graph(payload)
        extras["search_url"] = _clean(payload.get("search_url"))
        return parse_scrappa(payload)
    if kind == "ddg_lite":
        return parse_ddg_lite(text)
    if kind == "tavily":
        return parse_tavily(_json(text) or {})
    if kind == "serper":
        return parse_serper(_json(text) or {})
    return parse_ddg_html(text)


def _recorded_kind(url: str) -> Optional[str]:
    if url.startswith(SCRAPPA_URL):
        return "scrappa"
    if url == DDG_LITE_URL:
        return "ddg_lite"
    if url == DDG_HTML_URL:
        return "ddg_html"
    return None


def _best_recorded_search(settings: Settings, query: str) -> Optional[tuple[str, str]]:
    """Closest recorded search for `query`, as (body_text, parser_kind)."""
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
        kind = _recorded_kind(raw.get("url", ""))
        if kind is None:
            continue
        recorded = _tokens(raw.get("query", ""))
        if not recorded:
            continue
        score = _match_score(wanted, recorded)
        if score > 0 and (best is None or score > best[0]):
            best = (score, raw.get("text", ""), kind)

    return (best[1], best[2]) if best else None


def _scrappa_error(res: "_http.HttpResult") -> str:
    """Phrase a Scrappa failure so the next step is obvious from the message."""
    detail = _clean((res.text or "")[:200])
    if res.timed_out:
        return f"scrappa: no response within {SCRAPPA_TIMEOUT_S:.0f}s"
    if res.status in (401, 403):
        return f"scrappa: key rejected (HTTP {res.status}) — check SCRAPPA_API_KEY"
    if res.status == 422:
        return (
            f"scrappa: HTTP 422 — the API rejected the request parameters "
            f"(the search term must be sent as 'query', not 'q'){f': {detail}' if detail else ''}"
        )
    if res.status == 429:
        return "scrappa: HTTP 429 — rate limited or out of monthly credits"
    if res.status >= 500:
        return f"scrappa: HTTP {res.status} upstream error{f' — {detail}' if detail else ''}"
    return f"scrappa: {res.error or 'request failed'}"


def _with_knowledge_graph(
    body: str, extras: dict[str, Any]
) -> tuple[str, dict[str, str], list[str]]:
    """Fold Google's panel above the result list, and make it citable.

    Returns (summary, knowledge_graph, extra_sources). The panel goes FIRST: it is
    the only part of a SERP that states founded date, HQ, headcount or CEO
    outright, and burying it under nine snippets is how extraction misses it.
    """
    kg = extras.get("knowledge_graph") or {}
    if not isinstance(kg, dict) or not kg:
        return body, {}, []
    rendered = render_knowledge_graph(kg)
    if not rendered:
        return body, {}, []
    search_url = extras.get("search_url") or ""
    sources = [search_url] if search_url.startswith(("http://", "https://")) else []
    return f"{rendered}\n\n{body}" if body else rendered, kg, sources


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
