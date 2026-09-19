"""Web search. DuckDuckGo's keyless HTML endpoint is the default path and the one
that has to work without any API key at all; the paid providers are upgrades.
"""

from __future__ import annotations

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


def _dedupe(results: list[SearchResult], limit: int) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if not r.url or not r.url.startswith(("http://", "https://")):
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


def parse_ddg_html(html: str) -> list[SearchResult]:
    """Parse the html.duckduckgo.com/html/ layout: a.result__a + a.result__snippet."""
    out: list[SearchResult] = []
    if not html:
        return out
    tree = HTMLParser(html)
    for node in tree.css("div.result, div.web-result, div.results_links"):
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

            if err:
                return self._fail(err, started, {"provider": provider, "query": query})
            results = _dedupe(results, num)
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
        form = {"q": query, "kl": "us-en"}
        res = _http.request(
            settings,
            "POST",
            DDG_HTML_URL,
            data=form,
            browser_ua=True,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://duckduckgo.com/",
            },
        )
        results: list[SearchResult] = []
        first_error: Optional[str] = None
        if res.ok:
            results = parse_ddg_html(res.text)
        else:
            first_error = res.error

        if results:
            return results, None

        # The html endpoint intermittently serves an anomaly/empty page; lite is a
        # different backend with a different layout and usually still answers.
        lite = _http.request(
            settings,
            "POST",
            DDG_LITE_URL,
            data={"q": query},
            browser_ua=True,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://lite.duckduckgo.com/",
            },
        )
        if lite.ok:
            results = parse_ddg_lite(lite.text)
            if results:
                return results, None
            return [], first_error or "duckduckgo returned no parseable results (rate limited?)"
        return [], first_error or lite.error or "duckduckgo request failed"

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
        return [], f"fixture: no recorded search results for '{query}'"

    # -- plumbing --------------------------------------------------------

    def _fail(self, error: str, started: float, data: dict[str, Any]) -> Observation:
        return Observation(
            tool=self.name,
            ok=False,
            summary=f"search failed: {error}",
            data={**data, "results": data.get("results", [])},
            error=error,
            latency_ms=_ms(started),
        )


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
