"""Infer a company's tech stack from what its homepage ships to the browser.

Everything here is inference, never assertion: each hit carries the literal string
that matched and where it matched, so the report can cite its own reasoning.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from lea.config import Settings
from lea.models import Observation, TechStack, ToolName

from .fetch import fetch_page, http_error_message, normalize_url


@dataclass(frozen=True)
class Signature:
    """One product and the fingerprints it leaves behind."""

    category: str
    name: str
    # Substrings matched against script src / link href hostnames+paths.
    scripts: tuple[str, ...] = ()
    # Substrings matched against the raw HTML (attributes, inline JS, comments).
    html: tuple[str, ...] = ()
    # (header-name, substring) — substring "" means "header present at all".
    headers: tuple[tuple[str, str], ...] = ()
    # Cookie names appearing in set-cookie.
    cookies: tuple[str, ...] = ()
    confidence: float = 0.7


SIGNATURES: tuple[Signature, ...] = (
    # ---- analytics ------------------------------------------------------
    Signature("analytics", "Google Analytics 4",
              scripts=("googletagmanager.com/gtag/js", "google-analytics.com/analytics.js"),
              html=("gtag('config'", 'gtag("config"', "G-"), confidence=0.8),
    Signature("analytics", "Google Tag Manager",
              scripts=("googletagmanager.com/gtm.js",), html=("GTM-",), confidence=0.8),
    Signature("analytics", "Segment",
              scripts=("cdn.segment.com/analytics.js", "cdn.segment.io"),
              html=("analytics.load(",), confidence=0.85),
    Signature("analytics", "Amplitude",
              scripts=("cdn.amplitude.com", "amplitude.com/libs"),
              html=("amplitude.getInstance",), confidence=0.85),
    Signature("analytics", "Mixpanel",
              scripts=("cdn.mxpnl.com", "cdn4.mxpnl.com"), html=("mixpanel.init",), confidence=0.85),
    Signature("analytics", "Heap",
              scripts=("cdn.heapanalytics.com",), html=("heap.load(",), confidence=0.85),
    Signature("analytics", "Plausible",
              scripts=("plausible.io/js",), confidence=0.9),
    Signature("analytics", "PostHog",
              scripts=("app.posthog.com", "posthog.com/static/array.js", "i.posthog.com"),
              html=("posthog.init",), confidence=0.85),
    Signature("analytics", "Hotjar", scripts=("static.hotjar.com",), html=("hjid:",), confidence=0.85),

    # ---- marketing / CRM ------------------------------------------------
    Signature("marketing", "HubSpot",
              scripts=("js.hs-scripts.com", "js.hsforms.net", "js.hs-analytics.net"),
              cookies=("hubspotutk",), confidence=0.85),
    Signature("marketing", "Marketo",
              scripts=("munchkin.marketo.net", ".mktoresp.com"), html=("Munchkin.init",), confidence=0.85),
    Signature("marketing", "Pardot",
              scripts=("pi.pardot.com", "pd.salesforceiq.com"), html=("piAId",), confidence=0.8),
    Signature("crm", "Salesforce",
              scripts=("force.com", "salesforce.com/embeddedservice", "my.salesforce.com"),
              html=("embedded_svc",), confidence=0.75),
    Signature("marketing", "Intercom",
              scripts=("widget.intercom.io", "js.intercomcdn.com"),
              html=("window.intercomSettings",), confidence=0.9),
    Signature("marketing", "Drift", scripts=("js.driftt.com", "js.drift.com"),
              html=("drift.load",), confidence=0.85),
    Signature("marketing", "Qualified", scripts=("js.qualified.com",),
              html=("qualified.init",), confidence=0.85),
    Signature("marketing", "Mailchimp", scripts=("chimpstatic.com", "list-manage.com"), confidence=0.75),
    Signature("marketing", "6sense", scripts=("j.6sc.co", "6sense.com"), confidence=0.8),

    # ---- support --------------------------------------------------------
    Signature("support", "Zendesk",
              scripts=("static.zdassets.com", "zendesk.com/embeddable"),
              html=("zEsettings", "zdassets"), confidence=0.85),
    Signature("support", "Front", scripts=("chat-assets.frontapp.com",), confidence=0.85),
    Signature("support", "Freshdesk", scripts=("freshwidget.com", "widget.freshworks.com"), confidence=0.85),
    Signature("support", "Crisp", scripts=("client.crisp.chat",), confidence=0.9),

    # ---- infra / hosting ------------------------------------------------
    Signature("hosting", "Vercel",
              headers=(("server", "vercel"), ("x-vercel-id", "")),
              scripts=("_vercel/insights",), confidence=0.9),
    Signature("hosting", "Netlify", headers=(("server", "netlify"), ("x-nf-request-id", "")), confidence=0.9),
    Signature("infra", "Cloudflare",
              headers=(("server", "cloudflare"), ("cf-ray", "")),
              scripts=("static.cloudflareinsights.com",), confidence=0.9),
    Signature("infra", "AWS CloudFront",
              headers=(("via", "cloudfront"), ("x-amz-cf-id", ""), ("server", "amazons3")),
              confidence=0.85),
    Signature("infra", "Fastly", headers=(("x-served-by", "cache-"), ("x-fastly-request-id", "")),
              confidence=0.8),
    Signature("hosting", "Heroku", headers=(("via", "vegur"), ("server", "cowboy")), confidence=0.7),
    Signature("hosting", "GitHub Pages", headers=(("server", "github.com"),), confidence=0.85),

    # ---- frameworks / CMS ----------------------------------------------
    Signature("framework", "Next.js",
              headers=(("x-powered-by", "next.js"),),
              scripts=("/_next/static",), html=("__NEXT_DATA__",), confidence=0.9),
    Signature("framework", "React", html=("data-reactroot", "react-dom", "__REACT_DEVTOOLS"),
              scripts=("react.production.min.js",), confidence=0.6),
    Signature("framework", "Vue.js", html=("data-v-app", "__VUE__", "data-v-"),
              scripts=("vue.min.js", "vue.runtime"), confidence=0.6),
    Signature("framework", "Nuxt", html=("__NUXT__",), scripts=("/_nuxt/",), confidence=0.85),
    Signature("framework", "Svelte", html=("svelte-", "__sveltekit"), scripts=("/_app/immutable",),
              confidence=0.7),
    Signature("framework", "Angular", html=("ng-version", "ng-app"), scripts=("angular.min.js",),
              confidence=0.8),
    Signature("framework", "Gatsby", html=("___gatsby", "gatsby-announcer"), scripts=("/page-data/",),
              confidence=0.85),
    Signature("cms", "Webflow", html=("data-wf-page", "data-wf-site"),
              scripts=("assets.website-files.com", "cdn.prod.website-files.com", "webflow.js"),
              confidence=0.9),
    Signature("cms", "WordPress", html=("wp-content", "wp-includes", 'name="generator" content="WordPress'),
              headers=(("x-powered-by", "wordpress"),), confidence=0.85),
    Signature("ecommerce", "Shopify",
              scripts=("cdn.shopify.com", "shopifycloud"), headers=(("x-shopid", ""),),
              html=("Shopify.theme",), confidence=0.9),
    Signature("cms", "HubSpot CMS", scripts=("hs-sites.com", "cdn2.hubspot.net"),
              html=("hs_cos_wrapper",), confidence=0.85),
    Signature("cms", "Contentful", scripts=("ctfassets.net",), confidence=0.75),
    Signature("cms", "Sanity", scripts=("cdn.sanity.io",), confidence=0.75),
    Signature("framework", "Ruby on Rails", headers=(("x-powered-by", "phusion passenger"),),
              html=("csrf-param",), cookies=("_rails_session",), confidence=0.6),
    Signature("framework", "Django", cookies=("csrftoken", "sessionid"), html=("csrfmiddlewaretoken",),
              confidence=0.55),

    # ---- auth -----------------------------------------------------------
    Signature("auth", "Auth0", scripts=("cdn.auth0.com", "auth0.com/js"), html=(".auth0.com/authorize",),
              confidence=0.85),
    Signature("auth", "Okta", scripts=("okta-cdn.com", "global.oktacdn.com"), html=(".okta.com/oauth2",),
              confidence=0.85),
    Signature("auth", "Clerk", scripts=("clerk.accounts.dev", "clerk.com/npm", "js.clerk.com"),
              html=("__clerk",), confidence=0.85),
    Signature("auth", "WorkOS", scripts=("workos.com",), html=("workos.com/sso",), confidence=0.8),

    # ---- payments -------------------------------------------------------
    Signature("payments", "Stripe", scripts=("js.stripe.com",), html=("stripe.com/checkout",),
              confidence=0.9),
    Signature("payments", "Chargebee", scripts=("js.chargebee.com",), confidence=0.9),
    Signature("payments", "Recurly", scripts=("js.recurly.com",), confidence=0.9),
    Signature("payments", "Paddle", scripts=("cdn.paddle.com",), confidence=0.9),

    # ---- docs -----------------------------------------------------------
    Signature("docs", "Docusaurus", html=('content="Docusaurus', "docusaurus-"), confidence=0.85),
    Signature("docs", "ReadMe", scripts=("readme.io", "readmessl.com"), html=("readme.io",), confidence=0.8),
    Signature("docs", "GitBook", scripts=("gitbook.com", "gitbook.io"), html=("gitbook",), confidence=0.75),
    Signature("docs", "Mintlify", scripts=("mintlify.com", "mintlify.b-cdn.net"), html=("mintlify",),
              confidence=0.8),

    # ---- jobs (hiring signals — these are the ones that matter for GTM) --
    Signature("jobs", "Greenhouse",
              scripts=("boards.greenhouse.io", "greenhouse.io/embed"),
              html=("boards.greenhouse.io", "grnhse"), confidence=0.9),
    Signature("jobs", "Lever", scripts=("jobs.lever.co",), html=("jobs.lever.co",), confidence=0.9),
    Signature("jobs", "Ashby", scripts=("jobs.ashbyhq.com",), html=("ashbyhq.com",), confidence=0.9),
    Signature("jobs", "Workable", scripts=("apply.workable.com",), html=("workable.com",), confidence=0.85),
    Signature("jobs", "Rippling ATS", html=("ats.rippling.com",), confidence=0.85),
)


# --------------------------------------------------------------------------
# ATS job boards: the platform *and* the org slug
# --------------------------------------------------------------------------

# Knowing a company runs Greenhouse is a weak signal; knowing its board is
# `boards.greenhouse.io/duolingo` is an exact, free, public headcount proxy —
# that slug is the key to the job-board JSON API (see tools/jobs.py). We are
# already parsing the careers page here, so we capture it in the same pass
# instead of scraping the page a second time.

# Path segments that are part of the URL grammar, never an org slug.
ATS_RESERVED_SLUGS = frozenset(
    {
        "embed", "api", "v0", "v1", "v2", "v3", "jobs", "job", "job_board",
        "job-board", "posting-api", "postings", "boards", "widget", "accounts",
        "static", "assets", "js", "css", "www", "search", "apply", "j",
    }
)

_SLUG = r"([A-Za-z0-9][A-Za-z0-9_-]*)"

# (display name, platform key, pattern). Every capturing group in a pattern is
# an alternative spelling of the same slug, so callers take the first non-empty.
ATS_BOARD_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "Ashby",
        "ashby",
        re.compile(rf"(?:jobs\.ashbyhq\.com|api\.ashbyhq\.com/posting-api/job-board)/{_SLUG}", re.I),
    ),
    (
        "Greenhouse",
        "greenhouse",
        re.compile(
            rf"(?:boards|job-boards)\.greenhouse\.io/(?:embed/job_board(?:/js)?/?\?for=)?{_SLUG}"
            rf"|boards-api\.greenhouse\.io/v1/boards/{_SLUG}",
            re.I,
        ),
    ),
    (
        "Lever",
        "lever",
        re.compile(rf"(?:jobs\.lever\.co|api\.lever\.co/v0/postings)/{_SLUG}", re.I),
    ),
    (
        "Workable",
        "workable",
        re.compile(
            rf"apply\.workable\.com/(?:api/v\d+/widget/accounts/)?{_SLUG}"
            rf"|\b{_SLUG}\.workable\.com",
            re.I,
        ),
    ),
)

# Where a human would go to read the board. Kept next to the patterns so the two
# never drift; tools/jobs.py owns the *API* URLs.
ATS_BOARD_URLS: dict[str, str] = {
    "ashby": "https://jobs.ashbyhq.com/{slug}",
    "greenhouse": "https://job-boards.greenhouse.io/{slug}",
    "lever": "https://jobs.lever.co/{slug}",
    "workable": "https://apply.workable.com/{slug}",
}

# JSON embedded in a <script> escapes its slashes, and hydration payloads are
# exactly where modern careers pages hide the board URL.
_ESCAPED_SLASH = re.compile(r"\\+/|\\u002[fF]", re.I)


def board_url_for(platform: str, slug: str) -> str:
    template = ATS_BOARD_URLS.get((platform or "").lower())
    return template.format(slug=slug) if template and slug else ""


def extract_ats_boards(html: str, url: str = "") -> list[dict[str, str]]:
    """Every ATS board referenced by this page: platform, org slug, board URL.

    Pure string work, no network. Ordered by first appearance, deduplicated on
    (platform, slug) — a careers page links its own board many times over.
    """
    haystack = _ESCAPED_SLASH.sub("/", f"{url or ''}\n{html or ''}")
    if not haystack.strip():
        return []

    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for name, platform, pattern in ATS_BOARD_PATTERNS:
        for match in pattern.finditer(haystack):
            slug = next((g for g in match.groups() if g), "")
            slug = slug.strip().strip("-_").lower()
            if not slug or slug in ATS_RESERVED_SLUGS or len(slug) < 2:
                continue
            key = (platform, slug)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "ats": name,
                    "platform": platform,
                    "slug": slug,
                    "board_url": board_url_for(platform, slug),
                    "evidence": f"found '{match.group(0)[:80]}' on the page",
                }
            )
    return out


# A careers page is usually visited once, by detect_tech, several steps before
# anything asks about headcount. Remembering what it saw — for this process only —
# is what lets fetch_jobs hit the right board without fetching the page again.
_ATS_BOARD_MEMO: dict[str, list[dict[str, str]]] = {}


def domain_key(url_or_domain: str) -> str:
    """`https://www.Vanta.com/careers` -> `vanta.com`. The memo's cache key."""
    raw = (url_or_domain or "").strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        host = (urlparse(raw).hostname or "").strip(".")
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def remember_ats_boards(url: str, boards: Iterable[dict[str, str]]) -> None:
    key = domain_key(url)
    entries = [b for b in boards if b.get("slug")]
    if key and entries:
        _ATS_BOARD_MEMO[key] = entries


def recall_ats_boards(url_or_domain: str) -> list[dict[str, str]]:
    return list(_ATS_BOARD_MEMO.get(domain_key(url_or_domain), ()))


def clear_ats_memo() -> None:
    _ATS_BOARD_MEMO.clear()


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass
class _Hit:
    evidence: str
    confidence: float
    rank: int  # lower = stronger evidence source


def _script_sources(tree: HTMLParser) -> list[str]:
    srcs: list[str] = []
    for node in tree.css("script[src]"):
        src = (node.attributes.get("src") or "").strip()
        if src:
            srcs.append(src)
    for node in tree.css("link[href]"):
        href = (node.attributes.get("href") or "").strip()
        if href:
            srcs.append(href)
    for node in tree.css("iframe[src]"):
        src = (node.attributes.get("src") or "").strip()
        if src:
            srcs.append(src)
    return srcs


def detect_from_html(html: str, headers: Optional[dict[str, str]] = None) -> list[TechStack]:
    """Pure signature matching — no network. Other modules can call this directly."""
    html = html or ""
    norm_headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    lower_html = html.lower()

    try:
        tree = HTMLParser(html) if html else None
        srcs = _script_sources(tree) if tree is not None else []
    except Exception:
        srcs = []
    lower_srcs = [s.lower() for s in srcs]
    cookie_blob = " ".join(
        v for k, v in norm_headers.items() if k in ("set-cookie", "cookie")
    ).lower()

    best: dict[tuple[str, str], _Hit] = {}

    for sig in SIGNATURES:
        hit: Optional[_Hit] = None

        for header_name, needle in sig.headers:
            value = norm_headers.get(header_name.lower())
            if value is None:
                continue
            if needle and needle.lower() not in value.lower():
                continue
            shown = f"{header_name}: {value[:60]}" if value else header_name
            # A response header is the strongest evidence available, but nothing
            # here is ever certain — cap below 1.0.
            hit = _Hit(f"header '{shown}'", min(0.95, sig.confidence + 0.1), 0)
            break

        if hit is None:
            for needle in sig.scripts:
                match = next((s for s, low in zip(srcs, lower_srcs) if needle.lower() in low), None)
                if match is not None:
                    hit = _Hit(f"matched '{needle}' in script src", sig.confidence, 1)
                    break

        if hit is None and cookie_blob:
            for needle in sig.cookies:
                if needle.lower() in cookie_blob:
                    hit = _Hit(f"matched cookie '{needle}' in set-cookie", sig.confidence, 2)
                    break

        if hit is None:
            for needle in sig.html:
                # Bare short tokens like "G-" are too noisy on their own.
                if len(needle) < 3:
                    continue
                if needle.lower() in lower_html:
                    hit = _Hit(f"matched '{needle}' in page HTML", max(0.4, sig.confidence - 0.1), 3)
                    break

        if hit is not None:
            key = (sig.category, sig.name)
            existing = best.get(key)
            if existing is None or hit.rank < existing.rank:
                best[key] = hit

    out = [
        TechStack(category=cat, name=name, evidence=h.evidence, confidence=round(h.confidence, 2))
        for (cat, name), h in best.items()
    ]
    out.sort(key=lambda t: (t.category, -t.confidence, t.name))
    return out


def detect_from_result(html: str, headers: dict[str, str], url: str = "") -> list[TechStack]:
    """detect_from_html plus a couple of host-level inferences from the final URL."""
    tech = detect_from_html(html, headers)
    known = {(t.category, t.name) for t in tech}
    host = (urlparse(url).netloc or "").lower() if url else ""
    host_map = {
        "webflow.io": ("cms", "Webflow"),
        "vercel.app": ("hosting", "Vercel"),
        "netlify.app": ("hosting", "Netlify"),
        "github.io": ("hosting", "GitHub Pages"),
        "myshopify.com": ("ecommerce", "Shopify"),
    }
    for suffix, (cat, name) in host_map.items():
        if host.endswith(suffix) and (cat, name) not in known:
            tech.append(
                TechStack(category=cat, name=name, evidence=f"site is hosted on {suffix}", confidence=0.9)
            )
    return tech


# --------------------------------------------------------------------------
# the tool
# --------------------------------------------------------------------------


class DetectTechTool:
    name = ToolName.DETECT_TECH
    description = (
        "Infer the technologies a site uses (analytics, CRM, hosting, framework, "
        "auth, payments, docs, applicant tracking) from its headers and markup. "
        "Each finding carries the evidence that produced it; applicant-tracking "
        "hits are hiring signals. This tool loads the page itself, so it does NOT "
        "need fetch_page to run first — and calling it on a URL already fetched "
        "this run is cheap, because the response is reused from cache."
    )
    args_schema: dict[str, Any] = {
        "url": {
            "type": "string",
            "required": True,
            "description": "Absolute URL, usually the company homepage.",
        },
    }

    def run(self, args: dict[str, Any], settings: Settings) -> Observation:
        started = time.perf_counter()
        url = ""
        try:
            url = normalize_url(str((args or {}).get("url") or ""))
            if not url:
                return self._fail("missing required arg 'url'", url, started)

            res = fetch_page(url, settings)
            if not res.ok:
                return self._fail(http_error_message(res, url), url, started)

            final_url = res.url or url
            tech = detect_from_result(res.text or "", res.headers, final_url)
            boards = extract_ats_boards(res.text or "", final_url)
            # Remembered against the page's own host so fetch_jobs can find it by
            # company domain later in the run.
            remember_ats_boards(final_url, boards)

            if not tech:
                return Observation(
                    tool=self.name,
                    ok=True,
                    summary=f"No known technologies matched on {final_url}.",
                    data={"url": final_url, "tech": [], "ats_boards": boards},
                    sources=[final_url],
                    latency_ms=_ms(started),
                )

            return Observation(
                tool=self.name,
                ok=True,
                summary=_render(final_url, tech, boards),
                data={"url": final_url, "tech": [t.model_dump() for t in tech], "ats_boards": boards},
                sources=[final_url],
                latency_ms=_ms(started),
            )
        except Exception as exc:
            return self._fail(f"{type(exc).__name__}: {exc}", url, started)

    def _fail(self, error: str, url: str, started: float) -> Observation:
        return Observation(
            tool=self.name,
            ok=False,
            summary=f"tech detection failed: {error}",
            data={"url": url, "tech": [], "ats_boards": []},
            error=error,
            latency_ms=_ms(started),
        )


def _render(url: str, tech: Iterable[TechStack], boards: Iterable[dict[str, str]] = ()) -> str:
    by_cat: dict[str, list[TechStack]] = {}
    for t in tech:
        by_cat.setdefault(t.category, []).append(t)
    lines = [f"Tech detected on {url}:"]
    for cat in sorted(by_cat):
        for t in by_cat[cat]:
            lines.append(f"- [{cat}] {t.name} (confidence {t.confidence:.2f}) — {t.evidence}")
    for b in boards:
        # Spelled out for the planner: this is the exact argument fetch_jobs wants.
        lines.append(
            f"- [jobs] public {b['ats']} job board '{b['slug']}' at {b['board_url']} — "
            f"call fetch_jobs with this URL for an exact open-role count."
        )
    return "\n".join(lines)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
