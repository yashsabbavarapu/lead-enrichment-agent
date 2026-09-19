"""Infer a company's tech stack from what its homepage ships to the browser.

Everything here is inference, never assertion: each hit carries the literal string
that matched and where it matched, so the report can cite its own reasoning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from selectolax.parser import HTMLParser

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
            if not tech:
                return Observation(
                    tool=self.name,
                    ok=True,
                    summary=f"No known technologies matched on {final_url}.",
                    data={"url": final_url, "tech": []},
                    sources=[final_url],
                    latency_ms=_ms(started),
                )

            return Observation(
                tool=self.name,
                ok=True,
                summary=_render(final_url, tech),
                data={"url": final_url, "tech": [t.model_dump() for t in tech]},
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
            data={"url": url, "tech": []},
            error=error,
            latency_ms=_ms(started),
        )


def _render(url: str, tech: Iterable[TechStack]) -> str:
    by_cat: dict[str, list[TechStack]] = {}
    for t in tech:
        by_cat.setdefault(t.category, []).append(t)
    lines = [f"Tech detected on {url}:"]
    for cat in sorted(by_cat):
        for t in by_cat[cat]:
            lines.append(f"- [{cat}] {t.name} (confidence {t.confidence:.2f}) — {t.evidence}")
    return "\n".join(lines)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
