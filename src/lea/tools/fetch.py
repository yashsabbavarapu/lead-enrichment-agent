"""Fetch a page and turn it into readable text the LLM can reason over."""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from selectolax.parser import HTMLParser

from lea.config import Settings
from lea.models import Observation, ToolName

from . import http as _http

# Chrome-shaped headers keep a fair number of marketing sites from serving a 403.
MAX_HTML_CHARS = 60_000
SUMMARY_TEXT_CHARS = 800

# Chrome/boilerplate containers that never carry the substance of a page.
DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "nav",
    "footer",
    "header",
    "aside",
    "iframe",
    "template",
)

# Tags that end the current line of text. <a>/<span>/<em> are deliberately absent:
# their text belongs to the sentence around them.
BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "main", "article", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "ul", "ol", "dl", "dt", "dd", "table", "tr", "td", "th", "blockquote",
        "figure", "figcaption", "pre", "address", "details", "summary", "label",
        "button", "option", "body",
    }
)

ADDRESS_HINT = re.compile(
    r"(?:\b\d{1,5}\s+[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,4}\s+"
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Suite|Ste\.?|Floor|Fl\.?)"
    r"[^\n]{0,60})",
    re.IGNORECASE,
)
CITY_HINT = re.compile(
    r"\b(?:headquartered in|HQ(?:\s+is)?\s+in|based in|offices? in|located in)\s+([A-Z][^.\n,;]{2,60})"
)

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


# --------------------------------------------------------------------------
# extraction (importable without a network round trip)
# --------------------------------------------------------------------------


def extract_meta(tree: HTMLParser) -> dict[str, str]:
    meta: dict[str, str] = {}
    title = tree.css_first("title")
    if title is not None:
        meta["title"] = _one_line(title.text())
    for node in tree.css("meta"):
        attrs = node.attributes
        key = (attrs.get("property") or attrs.get("name") or "").strip().lower()
        content = (attrs.get("content") or "").strip()
        if not key or not content:
            continue
        if key == "description":
            meta.setdefault("description", _one_line(content))
        elif key.startswith("og:") or key.startswith("twitter:"):
            meta.setdefault(key, _one_line(content))
    if "title" not in meta and meta.get("og:title"):
        meta["title"] = meta["og:title"]
    if "description" not in meta and meta.get("og:description"):
        meta["description"] = meta["og:description"]
    return meta


def extract_text(html: str) -> str:
    """Readable main text: boilerplate stripped, <main>/<article> preferred."""
    if not html:
        return ""
    tree = HTMLParser(html)
    body = tree.body or tree.root
    if body is None:
        return ""
    for tag in DROP_TAGS:
        for node in body.css(tag):
            node.decompose()

    root = None
    for selector in ("main", "article", '[role="main"]', "#content", "#main"):
        candidate = body.css_first(selector)
        if candidate is not None and len(candidate.text(separator=" ", strip=True) or "") > 200:
            root = candidate
            break
    root = root or body

    # Walk once in document order. Inline elements (including <a>, so link text
    # survives) merge into the current line; block elements start a new one.
    lines: list[str] = []
    buf: list[str] = []
    pending_prefix = ""

    def flush() -> None:
        nonlocal pending_prefix
        line = _one_line("".join(buf))
        buf.clear()
        prefix, pending_prefix = pending_prefix, ""
        if line:
            lines.append(prefix + line)

    for node in root.traverse(include_text=True):
        tag = node.tag
        if tag == "-text":
            buf.append(node.text(deep=False) or "")
            continue
        if tag in BLOCK_TAGS:
            flush()
            if tag in ("h1", "h2", "h3", "h4"):
                lines.append("")  # blank line before a heading keeps sections legible
            elif tag in ("li", "dd"):
                pending_prefix = "- "
        elif tag in ("br", "hr"):
            flush()
    flush()

    text = "\n".join(lines)
    if not text.strip():
        text = root.text(separator=" ", strip=True) or ""
    return _collapse(text)


def extract_locations(text: str, meta: dict[str, str]) -> list[str]:
    hits: list[str] = []
    haystack = "\n".join([text, meta.get("description", ""), meta.get("og:description", "")])
    for m in ADDRESS_HINT.finditer(haystack):
        hits.append(_one_line(m.group(0)))
    for m in CITY_HINT.finditer(haystack):
        hits.append(_one_line(m.group(0)))
    out: list[str] = []
    seen: set[str] = set()
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out[:8]


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…[truncated]"


def _one_line(s: Optional[str]) -> str:
    return _WS.sub(" ", (s or "").replace("\r", " ").replace("\n", " ")).strip()


def _collapse(s: str) -> str:
    s = _WS.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return _BLANKS.sub("\n\n", s).strip()


def is_html(content_type: str, text: str) -> bool:
    ct = (content_type or "").lower()
    if "html" in ct or "xml" in ct:
        return True
    if ct and not ct.startswith("text/"):
        return False
    head = (text or "")[:2000].lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head


# --------------------------------------------------------------------------
# shared fetch used by both FetchPageTool and DetectTechTool
# --------------------------------------------------------------------------


def fetch_page(url: str, settings: Settings) -> _http.HttpResult:
    return _http.request(settings, "GET", url, browser_ua=True, check_robots=True)


def normalize_url(url: str) -> str:
    url = (url or "").strip().strip("<>").strip('"').strip("'")
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return "https://" + url
    return url


def http_error_message(res: _http.HttpResult, url: str) -> str:
    """Phrase failures so the agent can actually pick a different next step."""
    status = res.status
    if status == 404:
        return f"404 — page does not exist, try a different URL ({url})"
    if status == 403:
        return f"403 — site blocked the request (bot protection); try another page or source ({url})"
    if status == 401:
        return f"401 — page requires authentication, pick a public page instead ({url})"
    if status == 429:
        return f"429 — rate limited by {url}; wait or use a different source"
    if 500 <= status < 600:
        return f"{status} — server error at {url}; try again later or use a different source"
    if status:
        return f"HTTP {status} fetching {url}"
    return res.error or f"could not fetch {url}"


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


class FetchPageTool:
    name = ToolName.FETCH_PAGE
    description = (
        "Fetch one web page and return its readable main text plus title and meta "
        "description. Use it on a company homepage, /about, /careers, a funding "
        "announcement, or any URL returned by web_search."
    )
    args_schema: dict[str, Any] = {
        "url": {"type": "string", "required": True, "description": "Absolute URL to fetch."},
        "reason": {
            "type": "string",
            "required": False,
            "description": "One short phrase: what you expect this page to tell you.",
        },
    }

    def run(self, args: dict[str, Any], settings: Settings) -> Observation:
        started = time.perf_counter()
        url = ""
        try:
            url = normalize_url(str((args or {}).get("url") or ""))
            if not url:
                return self._fail("missing required arg 'url'", url, started)
            if not url.startswith(("http://", "https://")):
                return self._fail(f"unsupported URL scheme: {url}", url, started)

            res = fetch_page(url, settings)
            if not res.ok:
                return self._fail(http_error_message(res, url), url, started, status=res.status)

            content_type = res.header("content-type")
            if not is_html(content_type, res.text):
                return self._fail(
                    f"content-type '{content_type or 'unknown'}' is not HTML — "
                    f"this tool only reads web pages ({url})",
                    url,
                    started,
                    status=res.status,
                )

            html = res.text or ""
            tree = HTMLParser(html)
            meta = extract_meta(tree)
            text = truncate(extract_text(html), settings.max_page_chars)
            locations = extract_locations(text, meta)
            title = meta.get("title", "")

            summary_parts = [p for p in (title, meta.get("description", "")) if p]
            body_preview = text[:SUMMARY_TEXT_CHARS]
            if len(text) > SUMMARY_TEXT_CHARS:
                body_preview += "…"
            summary_parts.append(body_preview)
            summary = "\n".join(summary_parts).strip()

            return Observation(
                tool=self.name,
                ok=True,
                summary=summary or f"fetched {url} (no extractable text)",
                data={
                    "url": res.url or url,
                    "requested_url": url,
                    "status": res.status,
                    "title": title,
                    "description": meta.get("description", ""),
                    "meta": meta,
                    "locations": locations,
                    "text": text,
                    "html": truncate(html, MAX_HTML_CHARS),
                    "headers": res.headers,
                    "from_fixture": res.from_fixture,
                    "char_count": len(text),
                },
                sources=[res.url or url],
                latency_ms=_ms(started),
            )
        except Exception as exc:
            return self._fail(f"{type(exc).__name__}: {exc}", url, started)

    def _fail(self, error: str, url: str, started: float, status: int = 0) -> Observation:
        return Observation(
            tool=self.name,
            ok=False,
            summary=f"fetch failed: {error}",
            data={"url": url, "status": status, "text": "", "html": ""},
            error=error,
            sources=[],
            latency_ms=_ms(started),
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
