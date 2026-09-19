"""Tool-layer tests. Nothing here touches the network.

Two isolation strategies are used: `respx` mocks for the HTTP paths, and
`Settings(offline=True)` pointed at a temp fixture dir for the replay paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
import respx

from lea.config import Settings
from lea.models import Observation, ToolName
from lea.tools import http as lea_http
from lea.tools import search as lea_search
from lea.tools.fetch import FetchPageTool, extract_meta, extract_text, truncate
from lea.tools.registry import build_registry, tool_specs
from lea.tools.search import (
    WebSearchTool,
    ddg_block_reason,
    parse_ddg_html,
    parse_ddg_lite,
    parse_brave,
    parse_serper,
    parse_tavily,
    unwrap_ddg_href,
)
from lea.tools.jobs import (
    FetchJobsTool,
    Role,
    aggregate,
    parse_ashby,
    parse_greenhouse,
    parse_lever,
    parse_workable,
    prune_descriptions,
    resolve_candidates,
    slugify,
)
from lea.tools.tech_detect import (
    SIGNATURES,
    DetectTechTool,
    clear_ats_memo,
    detect_from_html,
    extract_ats_boards,
    recall_ats_boards,
)

from selectolax.parser import HTMLParser

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Module-level caches and pacing state must not leak between tests."""
    lea_http.clear_robots_cache()
    lea_http.clear_response_cache()
    lea_search.reset_pacing()
    clear_ats_memo()
    yield
    lea_http.clear_robots_cache()
    lea_http.clear_response_cache()
    lea_search.reset_pacing()
    clear_ats_memo()


PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def stub_dns(monkeypatch) -> dict[str, list[str]]:
    """Stub DNS so no test performs a real lookup.

    Everything resolves to a public address unless a test maps it otherwise.
    """
    mapping: dict[str, list[str]] = {}
    monkeypatch.setattr(lea_http, "_resolve_host", lambda host: mapping.get(host, [PUBLIC_IP]))
    return mapping


@pytest.fixture(autouse=True)
def sleeps(monkeypatch) -> list[float]:
    """Record backoff/pacing sleeps instead of waiting them out.

    Without this the retry tests would take ~7s each and the suite would be too
    slow to run on every edit.
    """
    recorded: list[float] = []
    monkeypatch.setattr(lea_search, "_sleep", lambda s: recorded.append(s))
    return recorded


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Live-ish settings: robots off (so no extra request), fixtures in a temp dir."""
    return Settings(
        search_provider="duckduckgo",
        respect_robots=False,
        fixture_dir=tmp_path / "fixtures",
        http_timeout_s=2.0,
    )


def _wrap(url: str) -> str:
    """Build the DuckDuckGo redirector wrapper the parser has to unwrap."""
    return "//duckduckgo.com/l/?uddg=" + quote(url, safe="") + "&rut=deadbeef"


DDG_PAGE = f"""<!DOCTYPE html><html><body>
<div id="links" class="results">
  <div class="result results_links web-result">
    <div class="links_main result__body">
      <h2 class="result__title"><a class="result__a" href="{_wrap('https://ramp.com/')}">Ramp — Spend Management</a></h2>
      <a class="result__snippet" href="{_wrap('https://ramp.com/')}">Corporate cards and spend management for finance teams.</a>
    </div>
  </div>
  <div class="result results_links web-result">
    <div class="links_main result__body">
      <h2 class="result__title"><a class="result__a" href="{_wrap('https://techcrunch.com/2024/04/ramp-series-d/')}">Ramp raises $150M Series D</a></h2>
      <a class="result__snippet" href="{_wrap('https://techcrunch.com/2024/04/ramp-series-d/')}">Fintech Ramp announced a $150 million round at a $7.65B valuation.</a>
    </div>
  </div>
</div></body></html>"""

DDG_LITE_PAGE = f"""<!DOCTYPE html><html><body><table>
<tr><td>1.&nbsp;</td><td><a class="result-link" href="{_wrap('https://ramp.com/about')}">About Ramp</a></td></tr>
<tr><td></td><td class="result-snippet">Ramp was founded in 2019 and is headquartered in New York, NY.</td></tr>
<tr><td>2.&nbsp;</td><td><a class="result-link" href="{_wrap('https://ramp.com/careers')}">Careers at Ramp</a></td></tr>
<tr><td></td><td class="result-snippet">Open roles across engineering, sales and finance.</td></tr>
</table></body></html>"""

COMPANY_HTML = """<!DOCTYPE html><html lang="en"><head>
<title>Acme Robotics — Warehouse Automation</title>
<meta name="description" content="Acme Robotics builds autonomous warehouse robots.">
<meta property="og:title" content="Acme Robotics">
<meta property="og:site_name" content="Acme">
<script src="https://www.googletagmanager.com/gtag/js?id=G-ABCDEF"></script>
<script src="https://cdn.segment.com/analytics.js/v1/k/analytics.min.js"></script>
<script src="https://js.stripe.com/v3/"></script>
<script src="https://js.hs-scripts.com/999.js"></script>
<script src="https://widget.intercom.io/widget/acme"></script>
<script src="https://boards.greenhouse.io/embed/job_board/js?for=acme"></script>
<script src="https://app.posthog.com/static/array.js"></script>
<script src="/_next/static/chunks/main.js"></script>
</head><body>
<nav><a href="/pricing">Pricing</a></nav>
<header><span>skip me</span></header>
<main>
  <h1>Autonomous warehouses</h1>
  <p>Acme Robotics builds robots that move pallets. Read our <a href="/story">founding story</a> for more.</p>
  <ul><li>Fleet management</li><li>Real-time routing</li></ul>
  <p>Acme is headquartered in Austin, TX and was founded in 2018.</p>
  <p>Visit us at 500 Congress Avenue, Suite 400.</p>
</main>
<footer><p>footer noise</p></footer>
<script>console.log("ignored")</script>
</body></html>"""


# --------------------------------------------------------------------------
# DuckDuckGo parsing + uddg unwrap
# --------------------------------------------------------------------------


def test_unwrap_ddg_href_extracts_real_url():
    wrapped = _wrap("https://ramp.com/about?utm=1")
    assert unwrap_ddg_href(wrapped) == "https://ramp.com/about?utm=1"


def test_unwrap_ddg_href_handles_protocol_relative_and_plain():
    assert unwrap_ddg_href("https://example.com/x") == "https://example.com/x"
    assert unwrap_ddg_href("//example.com/x") == "https://example.com/x"
    assert unwrap_ddg_href("/l/?uddg=" + quote("https://ex.org/a", safe="")) == "https://ex.org/a"
    assert unwrap_ddg_href("") == ""


def test_parse_ddg_html_titles_urls_snippets():
    results = parse_ddg_html(DDG_PAGE)
    assert len(results) == 2
    assert results[0].title == "Ramp — Spend Management"
    # The wrapper must be gone; the agent needs the real target to fetch.
    assert results[0].url == "https://ramp.com/"
    assert "duckduckgo.com/l/" not in results[0].url
    assert "Corporate cards" in results[0].snippet
    assert results[1].url == "https://techcrunch.com/2024/04/ramp-series-d/"


@respx.mock
def test_search_drops_duckduckgo_ad_slots(settings: Settings):
    """Sponsored rows are y.js redirects with no uddg to unwrap — they must not
    reach the agent as if they were research."""
    ad = "//duckduckgo.com/y.js?ad_domain=melio.com&ad_provider=bingv7aa&click_metadata=" + "x" * 300
    page = f"""<div id="links">
      <div class="result results_links_deep"><a class="result__a" href="{ad}">Sponsored</a>
        <a class="result__snippet">an ad</a></div>
      <div class="result web-result"><a class="result__a" href="{_wrap('https://ramp.com/')}">Ramp</a>
        <a class="result__snippet">real result</a></div>
    </div>"""
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, text=page))

    obs = WebSearchTool().run({"query": "Ramp"}, settings)

    assert obs.ok is True
    assert [r["url"] for r in obs.data["results"]] == ["https://ramp.com/"]
    assert "duckduckgo.com" not in obs.summary


def test_parse_ddg_lite_layout():
    results = parse_ddg_lite(DDG_LITE_PAGE)
    assert [r.url for r in results] == ["https://ramp.com/about", "https://ramp.com/careers"]
    assert results[0].title == "About Ramp"
    assert "founded in 2019" in results[0].snippet


@respx.mock
def test_search_duckduckgo_end_to_end(settings: Settings):
    route = respx.post(DDG_HTML_URL).mock(
        return_value=httpx.Response(200, text=DDG_PAGE, headers={"content-type": "text/html"})
    )
    obs = WebSearchTool().run({"query": "Ramp company", "num_results": 5}, settings)

    assert route.called
    assert isinstance(obs, Observation)
    assert obs.ok is True
    assert obs.tool is ToolName.WEB_SEARCH
    assert obs.data["provider"] == "duckduckgo"
    assert [r["url"] for r in obs.data["results"]] == [
        "https://ramp.com/",
        "https://techcrunch.com/2024/04/ramp-series-d/",
    ]
    assert obs.sources == [r["url"] for r in obs.data["results"]]
    assert obs.summary.startswith("1. Ramp — Spend Management — https://ramp.com/")
    assert "2. Ramp raises $150M Series D" in obs.summary


@respx.mock
def test_search_duckduckgo_falls_back_to_lite(settings: Settings):
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, text="<html><body>nothing</body></html>"))
    lite = respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(200, text=DDG_LITE_PAGE))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert lite.called
    assert obs.ok is True
    assert obs.data["results"][0]["url"] == "https://ramp.com/about"


@respx.mock
def test_search_degrades_gracefully_on_http_error(settings: Settings):
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(503, text="nope"))
    respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(503, text="nope"))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is False
    assert obs.error
    assert obs.data["results"] == []
    assert "search failed" in obs.summary


@respx.mock
def test_search_never_raises_on_transport_error(settings: Settings):
    respx.post(DDG_HTML_URL).mock(side_effect=httpx.ConnectError("boom"))
    respx.post(DDG_LITE_URL).mock(side_effect=httpx.ConnectError("boom"))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)
    assert obs.ok is False and obs.error


def test_search_rejects_missing_query(settings: Settings):
    obs = WebSearchTool().run({}, settings)
    assert obs.ok is False
    assert "query" in (obs.error or "")


def test_search_snippet_truncated_to_settings(settings: Settings, tmp_path: Path):
    settings.max_snippet_chars = 20
    long_snippet = "x" * 500
    html = (
        '<div class="result"><a class="result__a" href="https://a.com/">T</a>'
        f'<a class="result__snippet">{long_snippet}</a></div>'
    )
    with respx.mock:
        respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, text=html))
        obs = WebSearchTool().run({"query": "q"}, settings)
    assert obs.ok is True
    snippet_line = obs.summary.splitlines()[1].strip()
    assert snippet_line.endswith("…")
    assert len(snippet_line.rstrip("…")) <= 20


# --------------------------------------------------------------------------
# DuckDuckGo block handling: retry, rotation, pacing, degraded fallback
# --------------------------------------------------------------------------


CHALLENGE_PAGE = (
    "<html><body><h1>Sorry, we detected unusual activity from your network.</h1>"
    "<p>Please try again later.</p></body></html>"
)


@pytest.mark.parametrize(
    "status,text,expected",
    [
        (429, "whatever", "rate limited"),
        (403, "whatever", "forbidden"),
        (202, "whatever", "accepted-but-empty"),
        (503, "whatever", "upstream error"),
        (200, CHALLENGE_PAGE, "challenge page"),
        (200, "<html></html>", "suspiciously short"),
    ],
)
def test_ddg_block_reason_names_the_block(status: int, text: str, expected: str):
    res = lea_http.HttpResult(ok=status < 400, url=DDG_HTML_URL, status=status, text=text)
    reason = ddg_block_reason(res)
    assert reason is not None and expected in reason


def test_ddg_block_reason_is_none_for_a_real_page():
    res = lea_http.HttpResult(ok=True, url=DDG_HTML_URL, status=200, text=DDG_PAGE + "x" * 2000)
    assert ddg_block_reason(res) is None


@respx.mock
def test_ddg_retries_rotate_endpoint_and_user_agent(settings: Settings, sleeps: list[float]):
    """First endpoint is challenged; the retry hits the other endpoint with a
    different UA and succeeds."""
    html_route = respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, text=CHALLENGE_PAGE))
    lite_route = respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(200, text=DDG_LITE_PAGE))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is True
    assert obs.data["results"][0]["url"] == "https://ramp.com/about"
    assert html_route.called and lite_route.called

    ua_html = html_route.calls[0].request.headers["user-agent"]
    ua_lite = lite_route.calls[0].request.headers["user-agent"]
    assert ua_html != ua_lite  # rotation, not the same UA twice
    assert ua_html in lea_http.BROWSER_UAS and ua_lite in lea_http.BROWSER_UAS
    assert sleeps, "a retry must back off before the next attempt"


@respx.mock
def test_hard_block_short_circuits_instead_of_burning_the_budget(
    settings: Settings, sleeps: list[float], monkeypatch
):
    """A bot-check does not clear by waiting, so it must fail over fast.

    Each wasted attempt costs a full request timeout against a hanging endpoint,
    which is what made a blocked live run take over two minutes.
    """
    # Pacing off, so any recorded sleep is unambiguously retry backoff.
    monkeypatch.setattr(lea_search, "DDG_MIN_REQUEST_GAP_S", 0)
    html_route = respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(429, text="slow down"))
    lite_route = respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(202, text=CHALLENGE_PAGE))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is False
    total = html_route.call_count + lite_route.call_count
    assert total == lea_search.DDG_BLOCK_MAX_ATTEMPTS, f"burned {total} attempts on a hard block"
    assert total < lea_search.DDG_MAX_ATTEMPTS
    # both backends were still tried once before giving up
    assert html_route.called and lite_route.called
    # and no time was spent backing off between them
    assert sleeps == [], f"backed off on a hard block: {sleeps}"


@respx.mock
def test_a_hanging_endpoint_short_circuits_like_a_block(settings: Settings, monkeypatch):
    """A stalled connection is how a blocked DDG usually presents.

    This was the real cost: each timeout burned a full request timeout, and the
    transient path retried three times before giving up.
    """
    monkeypatch.setattr(lea_search, "DDG_MIN_REQUEST_GAP_S", 0)
    html_route = respx.post(DDG_HTML_URL).mock(side_effect=httpx.ReadTimeout("stalled"))
    lite_route = respx.post(DDG_LITE_URL).mock(side_effect=httpx.ConnectTimeout("stalled"))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is False
    assert html_route.call_count + lite_route.call_count == lea_search.DDG_BLOCK_MAX_ATTEMPTS
    assert "timed out" in (obs.error or "")


def test_timeouts_are_flagged_on_the_result():
    assert lea_http.HttpResult(ok=False, url="x", timed_out=True).timed_out is True
    assert lea_http.HttpResult(ok=True, url="x").timed_out is False


@respx.mock
def test_transient_errors_still_use_the_full_retry_budget(settings: Settings, sleeps: list[float]):
    """A reset or 5xx may genuinely be luck — those still get retried with backoff."""
    html_route = respx.post(DDG_HTML_URL).mock(side_effect=httpx.ConnectError("reset"))
    lite_route = respx.post(DDG_LITE_URL).mock(side_effect=httpx.ConnectError("reset"))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is False
    assert html_route.call_count + lite_route.call_count == lea_search.DDG_MAX_ATTEMPTS
    assert [s for s in sleeps if s >= min(lea_search.DDG_RETRY_BACKOFF_S) * (1 - lea_search.DDG_JITTER)]


def test_search_timeout_is_tighter_than_the_page_timeout():
    """A hanging search must surrender to the cache well before a page fetch would."""
    assert lea_search.DDG_REQUEST_TIMEOUT_S < Settings().http_timeout_s


def test_blocked_run_latency_budget_is_bounded():
    """Guards the demo-visible wall clock: two searches must not exceed ~30s."""
    worst_case_one_search = lea_search.DDG_BLOCK_MAX_ATTEMPTS * lea_search.DDG_REQUEST_TIMEOUT_S
    two_searches = 2 * (worst_case_one_search + lea_search.DDG_MIN_REQUEST_GAP_S)
    assert two_searches <= 30.0, f"a fully blocked run could take {two_searches:.0f}s"


@respx.mock
def test_blocked_error_is_actionable(settings: Settings):
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(429, text="slow down"))
    respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(200, text=CHALLENGE_PAGE))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is False
    err = obs.error or ""
    assert "blocked this client" in err
    # the operator needs to know all three ways out
    assert "retry in a minute" in err
    assert "TAVILY_API_KEY" in err
    assert "--offline" in err
    # and which signal produced the verdict
    assert "rate limited" in err or "challenge page" in err


@respx.mock
def test_ddg_paces_successive_queries(settings: Settings, sleeps: list[float]):
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, text=DDG_PAGE))
    tool = WebSearchTool()

    first = tool.run({"query": "query one"}, settings)
    assert not sleeps, "the first query should not wait"

    second = tool.run({"query": "query two"}, settings)
    assert first.ok and second.ok
    assert sleeps, "a back-to-back second query must self-throttle"
    assert max(sleeps) <= lea_search.DDG_MIN_REQUEST_GAP_S * (1 + lea_search.DDG_JITTER)


def test_pacing_does_not_sleep_in_offline_mode(settings: Settings, sleeps: list[float]):
    settings.offline = True
    settings.search_provider = "fixture"
    _write_fixture(settings, DDG_HTML_URL, {"q": "Ramp company", "kl": "us-en"}, DDG_PAGE)

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is True
    assert sleeps == [], "offline demos must never stall on backoff or pacing"


@respx.mock
def test_offline_ddg_does_not_retry(settings: Settings, sleeps: list[float]):
    """With no fixture, an offline DDG run fails immediately instead of looping."""
    settings.offline = True
    settings.search_provider = "duckduckgo"
    obs = WebSearchTool().run({"query": "nothing recorded"}, settings)
    assert obs.ok is False
    assert sleeps == []


@respx.mock
def test_blocked_search_falls_back_to_fixture_as_degraded(settings: Settings):
    """The demo-saving path: live DDG is blocked but the query was recorded."""
    _write_fixture(settings, DDG_HTML_URL, {"q": "Ramp company", "kl": "us-en"}, DDG_PAGE)
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(429, text="slow down"))
    respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(429, text="slow down"))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    # A blocked provider with a warm cache must still produce usable results.
    assert obs.ok is True
    assert obs.data["degraded"] is True
    assert obs.data["served_from"] == "fixture"
    assert "rate limited" in obs.data["degraded_reason"]
    assert obs.data["results"][0]["url"] == "https://ramp.com/"
    assert obs.sources
    # the summary must admit the results are cached, not fresh
    assert obs.summary.startswith("[cached]")
    assert "recorded fixture" in obs.summary


@respx.mock
def test_successful_search_is_not_marked_degraded(settings: Settings):
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, text=DDG_PAGE))
    obs = WebSearchTool().run({"query": "Ramp company"}, settings)
    assert obs.ok is True
    assert obs.data["degraded"] is False
    assert "served_from" not in obs.data


@respx.mock
def test_blocked_search_without_fixture_still_fails_cleanly(settings: Settings):
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(429, text="slow down"))
    respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(429, text="slow down"))

    obs = WebSearchTool().run({"query": "never recorded"}, settings)

    assert obs.ok is False
    assert obs.data["degraded"] is False
    assert obs.data["results"] == []


@respx.mock
def test_keyed_provider_failure_also_falls_back_to_fixture(settings: Settings):
    """The degraded path is provider-agnostic, not DDG-only."""
    settings.search_provider = "tavily"
    settings.tavily_api_key = "tv"
    _write_fixture(settings, DDG_HTML_URL, {"q": "Ramp company", "kl": "us-en"}, DDG_PAGE)
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(500, text="boom"))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is True
    assert obs.data["degraded"] is True
    assert obs.data["provider"] == "tavily"


# --------------------------------------------------------------------------
# keyed providers: shape mapping
# --------------------------------------------------------------------------


def test_parse_tavily_shape():
    payload = {
        "results": [
            {"title": "Ramp", "url": "https://ramp.com/", "content": "Spend management."},
            {"title": "News", "url": "https://tc.com/a", "content": "Series D."},
        ]
    }
    results = parse_tavily(payload)
    assert [(r.title, r.url, r.snippet) for r in results] == [
        ("Ramp", "https://ramp.com/", "Spend management."),
        ("News", "https://tc.com/a", "Series D."),
    ]


def test_parse_serper_shape():
    payload = {"organic": [{"title": "Ramp", "link": "https://ramp.com/", "snippet": "Cards."}]}
    results = parse_serper(payload)
    assert results[0].url == "https://ramp.com/"
    assert results[0].snippet == "Cards."


def test_parse_brave_shape():
    payload = {"web": {"results": [{"title": "Ramp", "url": "https://ramp.com/", "description": "Cards."}]}}
    results = parse_brave(payload)
    assert results[0].title == "Ramp"
    assert results[0].snippet == "Cards."


def test_parsers_tolerate_junk_payloads():
    assert parse_tavily({}) == []
    assert parse_serper({"organic": None}) == []
    assert parse_brave({"web": {"results": ["not a dict"]}}) == []
    assert parse_ddg_html("") == []
    assert parse_ddg_lite("<html></html>") == []


@respx.mock
def test_search_tavily_end_to_end(settings: Settings):
    settings.search_provider = "tavily"
    settings.tavily_api_key = "tv-key"
    route = respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"title": "Ramp", "url": "https://ramp.com/", "content": "Cards."}]}
        )
    )
    obs = WebSearchTool().run({"query": "Ramp", "num_results": 3}, settings)

    sent = json.loads(route.calls[0].request.content)
    assert sent == {"api_key": "tv-key", "query": "Ramp", "max_results": 3, "search_depth": "basic"}
    assert obs.ok is True
    assert obs.data["provider"] == "tavily"
    assert obs.data["results"][0]["url"] == "https://ramp.com/"


@respx.mock
def test_search_serper_sends_api_key_header(settings: Settings):
    settings.search_provider = "serper"
    settings.serper_api_key = "sk-serper"
    route = respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(
            200, json={"organic": [{"title": "Ramp", "link": "https://ramp.com/", "snippet": "Cards."}]}
        )
    )
    obs = WebSearchTool().run({"query": "Ramp"}, settings)

    assert route.calls[0].request.headers["x-api-key"] == "sk-serper"
    assert obs.ok is True and obs.sources == ["https://ramp.com/"]


@respx.mock
def test_search_brave_sends_subscription_token(settings: Settings):
    settings.search_provider = "brave"
    settings.brave_api_key = "brv"
    route = respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(
            200, json={"web": {"results": [{"title": "Ramp", "url": "https://ramp.com/", "description": "C"}]}}
        )
    )
    obs = WebSearchTool().run({"query": "Ramp"}, settings)

    assert route.calls[0].request.headers["x-subscription-token"] == "brv"
    assert obs.ok is True


def test_search_reports_missing_key(settings: Settings):
    settings.search_provider = "tavily"
    settings.tavily_api_key = None
    obs = WebSearchTool().run({"query": "Ramp"}, settings)
    assert obs.ok is False
    assert "TAVILY_API_KEY" in (obs.error or "")


# --------------------------------------------------------------------------
# offline / fixture cache
# --------------------------------------------------------------------------


def _write_fixture(
    settings: Settings, url: str, body, text: str, status: int = 200, headers=None, query: str = ""
) -> None:
    d = Path(settings.fixture_dir)
    d.mkdir(parents=True, exist_ok=True)
    key = lea_http.fixture_key(url, body)
    payload = {
        "url": url,
        "status": status,
        "headers": headers or {"content-type": "text/html; charset=utf-8"},
        "text": text,
        "recorded_at": "2024-01-01T00:00:00+00:00",
    }
    if query:
        payload["query"] = query
    (d / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "recorded,asked",
    [
        # the stub planner's real query shapes, against recorded wording
        ("vanta compliance automation company", "Vanta company overview"),
        ("vanta compliance automation company", '"Vanta" official site'),
        ("vanta funding round", "Vanta funding OR hiring OR news 2025"),
        ("ramp corporate cards company", "Ramp company overview"),
        ("ramp careers hiring", "Ramp careers"),
        ("vanta careers hiring", "Vanta hiring roles"),
    ],
)
def test_offline_search_matches_a_reworded_query(settings: Settings, recorded: str, asked: str):
    """The regression unit tests missed: --offline runs through the `fixture`
    PROVIDER, which must use the same fuzzy lookup as the degraded fallback."""
    settings.offline = True
    _write_fixture(settings, DDG_HTML_URL, {"q": recorded, "kl": "us-en"}, DDG_PAGE, query=recorded)

    obs = WebSearchTool().run({"query": asked}, settings)

    assert obs.ok is True, f"{asked!r} did not resolve to recorded {recorded!r}: {obs.error}"
    assert obs.data["results"]


@pytest.mark.parametrize(
    "recorded,asked",
    [
        # sharing only a research word must never cross companies
        ("vanta careers hiring", "Ramp careers"),
        ("ramp funding valuation", "Vanta funding"),
        ("vanta compliance automation company", "best sourdough starter recipe"),
    ],
)
def test_offline_search_refuses_a_different_subject(settings: Settings, recorded: str, asked: str):
    settings.offline = True
    _write_fixture(settings, DDG_HTML_URL, {"q": recorded, "kl": "us-en"}, DDG_PAGE, query=recorded)

    obs = WebSearchTool().run({"query": asked}, settings)

    assert obs.ok is False, f"{asked!r} wrongly matched recorded {recorded!r}"


@respx.mock
def test_degraded_fallback_fires_on_a_hard_provider_error(settings: Settings):
    """Not only on empty results — a blocked provider is the case that matters."""
    _write_fixture(
        settings, DDG_HTML_URL, {"q": "vanta compliance automation company", "kl": "us-en"},
        DDG_PAGE, query="vanta compliance automation company",
    )
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(202, text=CHALLENGE_PAGE))
    respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(202, text=CHALLENGE_PAGE))

    # reworded query: exercises fuzzy matching and the degraded path together
    obs = WebSearchTool().run({"query": "Vanta company overview"}, settings)

    assert obs.ok is True, obs.error
    assert obs.data["degraded"] is True
    assert obs.data["served_from"] == "fixture"
    assert "blocked this client" in obs.data["degraded_reason"]
    assert obs.data["results"]


@respx.mock
def test_degraded_fallback_fires_on_a_transport_error(settings: Settings):
    _write_fixture(
        settings, DDG_HTML_URL, {"q": "ramp corporate cards company", "kl": "us-en"},
        DDG_PAGE, query="ramp corporate cards company",
    )
    respx.post(DDG_HTML_URL).mock(side_effect=httpx.ConnectError("no route to host"))
    respx.post(DDG_LITE_URL).mock(side_effect=httpx.ConnectError("no route to host"))

    obs = WebSearchTool().run({"query": "Ramp company overview"}, settings)

    assert obs.ok is True
    assert obs.data["degraded"] is True


@pytest.mark.parametrize(
    "recorded,asked",
    [
        ("https://www.vanta.com/", "https://vanta.com"),
        ("https://www.vanta.com/", "https://www.vanta.com"),
        ("https://www.vanta.com/", "http://vanta.com/"),
        ("https://vanta.com/careers", "https://www.vanta.com/careers/"),
        ("https://ramp.com", "https://RAMP.com/"),
    ],
)
def test_fixture_lookup_ignores_www_scheme_and_trailing_slash(
    settings: Settings, recorded: str, asked: str
):
    """A demo that fails over a `www.` prefix looks broken."""
    settings.offline = True
    _write_fixture(settings, recorded, None, COMPANY_HTML)

    obs = FetchPageTool().run({"url": asked}, settings)

    assert obs.ok is True, f"{asked!r} missed a fixture recorded as {recorded!r}: {obs.error}"
    assert obs.data["title"]


def test_canonical_url_still_separates_distinct_pages():
    canon = lea_http.canonical_url
    assert canon("https://vanta.com/careers") != canon("https://vanta.com/pricing")
    assert canon("https://vanta.com") != canon("https://ramp.com")
    assert canon("https://a.com/x?q=1") != canon("https://a.com/x?q=2")


# --------------------------------------------------------------------------
# SSRF guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,needle",
    [
        ("http://127.0.0.1:8080/admin", "loopback"),
        ("http://localhost/", "loopback"),
        ("http://10.0.0.5/internal", "private"),
        ("http://192.168.1.1/router", "private"),
        ("http://172.16.0.9/", "private"),
        ("http://169.254.169.254/latest/meta-data/", "metadata"),
        ("http://[fd00:ec2::254]/latest/meta-data/", "metadata"),
        ("http://metadata.google.internal/computeMetadata/v1/", "metadata"),
        ("http://0.0.0.0/", "unspecified"),
        ("http://[::1]/", "loopback"),
    ],
)
def test_ssrf_guard_refuses_internal_literals(settings: Settings, url: str, needle: str):
    """An LLM picking these out of a hostile page must not reach them."""
    obs = FetchPageTool().run({"url": url}, settings)

    assert obs.ok is False
    assert needle in (obs.error or "").lower(), obs.error
    assert "refused" in (obs.error or "")
    assert obs.data["text"] == ""


def test_ssrf_guard_refuses_a_public_name_resolving_inward(
    settings: Settings, stub_dns: dict[str, list[str]]
):
    """The reason the check resolves DNS: the literal looks perfectly public."""
    stub_dns["spoof.example.com"] = ["127.0.0.1"]

    obs = FetchPageTool().run({"url": "http://spoof.example.com/"}, settings)

    assert obs.ok is False
    assert "loopback" in (obs.error or "")
    assert "spoof.example.com resolves to" in (obs.error or "")


def test_ssrf_guard_refuses_when_any_resolved_ip_is_internal(
    settings: Settings, stub_dns: dict[str, list[str]]
):
    """One bad A record is enough — round-robin DNS must not be a bypass."""
    stub_dns["mixed.example.com"] = [PUBLIC_IP, "10.1.2.3"]

    obs = FetchPageTool().run({"url": "http://mixed.example.com/"}, settings)
    assert obs.ok is False
    assert "private" in (obs.error or "")


@pytest.mark.parametrize("scheme_url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/"])
def test_ssrf_guard_refuses_non_http_schemes(settings: Settings, scheme_url: str):
    obs = FetchPageTool().run({"url": scheme_url}, settings)
    assert obs.ok is False


@respx.mock
def test_ssrf_guard_allows_a_public_host(settings: Settings):
    respx.get("https://acme.example/").mock(
        return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
    )
    obs = FetchPageTool().run({"url": "https://acme.example/"}, settings)
    assert obs.ok is True


@respx.mock
def test_ssrf_guard_refuses_a_redirect_into_the_private_range(
    settings: Settings, stub_dns: dict[str, list[str]]
):
    """The hop that matters: the first URL is public, the destination is not."""
    hop = respx.get("https://public.example/start").mock(
        return_value=httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
    )
    metadata = respx.get("http://169.254.169.254/latest/meta-data/").mock(
        return_value=httpx.Response(200, text="SECRET-CREDENTIALS")
    )

    obs = FetchPageTool().run({"url": "https://public.example/start"}, settings)

    assert hop.called
    assert not metadata.called, "the agent followed a redirect into cloud metadata"
    assert obs.ok is False
    assert "metadata" in (obs.error or "")
    assert "redirected from" in (obs.error or "")
    assert "SECRET" not in obs.data.get("text", "")


@respx.mock
def test_ssrf_guard_refuses_a_later_hop_in_the_chain(
    settings: Settings, stub_dns: dict[str, list[str]]
):
    stub_dns["inside.example"] = ["10.0.0.7"]
    respx.get("https://public.example/a").mock(
        return_value=httpx.Response(302, headers={"location": "https://public.example/b"})
    )
    respx.get("https://public.example/b").mock(
        return_value=httpx.Response(302, headers={"location": "https://inside.example/secret"})
    )
    inside = respx.get("https://inside.example/secret").mock(return_value=httpx.Response(200, text="x"))

    obs = FetchPageTool().run({"url": "https://public.example/a"}, settings)

    assert not inside.called
    assert obs.ok is False
    assert "private" in (obs.error or "")


@respx.mock
def test_public_redirects_are_still_followed(settings: Settings):
    respx.get("https://public.example/old").mock(
        return_value=httpx.Response(301, headers={"location": "https://public.example/new"})
    )
    respx.get("https://public.example/new").mock(
        return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
    )

    obs = FetchPageTool().run({"url": "https://public.example/old"}, settings)

    assert obs.ok is True
    assert obs.data["url"] == "https://public.example/new", "post-redirect URL must be tracked"


@respx.mock
def test_redirect_loops_are_bounded(settings: Settings):
    respx.get("https://public.example/loop").mock(
        return_value=httpx.Response(302, headers={"location": "https://public.example/loop"})
    )
    obs = FetchPageTool().run({"url": "https://public.example/loop"}, settings)
    assert obs.ok is False
    assert "too many redirects" in (obs.error or "")


def test_allow_private_hosts_escape_hatch(settings: Settings, monkeypatch):
    monkeypatch.setattr(lea_http, "ALLOW_PRIVATE_HOSTS", True)
    with respx.mock:
        route = respx.get("http://127.0.0.1:9000/").mock(
            return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
        )
        obs = FetchPageTool().run({"url": "http://127.0.0.1:9000/"}, settings)
    assert route.called
    assert obs.ok is True


def test_unresolvable_host_is_left_to_the_transport(settings: Settings, monkeypatch):
    """A DNS outage must not be reported as a security refusal."""
    def boom(host):
        raise OSError("name resolution failed")

    monkeypatch.setattr(lea_http, "_resolve_host", boom)
    assert lea_http.url_rejection("https://nowhere.example/") is None


def test_detect_tech_is_guarded_too(settings: Settings):
    obs = DetectTechTool().run({"url": "http://169.254.169.254/latest/meta-data/"}, settings)
    assert obs.ok is False
    assert obs.data["tech"] == []


def test_offline_mode_never_hits_network(settings: Settings):
    settings.offline = True
    with respx.mock(assert_all_called=False) as mock:
        blocked = mock.route(host__in=["html.duckduckgo.com", "example.com"]).mock(
            side_effect=AssertionError("network was used in offline mode")
        )
        obs = FetchPageTool().run({"url": "https://example.com/"}, settings)
        assert not blocked.called
    assert obs.ok is False
    assert "offline" in (obs.error or "")


def test_offline_search_reads_fixture(settings: Settings):
    settings.offline = True
    settings.search_provider = "fixture"
    _write_fixture(settings, DDG_HTML_URL, {"q": "Ramp company", "kl": "us-en"}, DDG_PAGE)

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)
    assert obs.ok is True
    assert obs.data["results"][0]["url"] == "https://ramp.com/"


def test_fixture_key_is_stable_and_body_sensitive():
    a = lea_http.fixture_key("https://x.com/", {"q": "one", "z": 2})
    b = lea_http.fixture_key("https://x.com/", {"z": 2, "q": "one"})
    c = lea_http.fixture_key("https://x.com/", {"q": "two"})
    assert a == b and a != c
    assert len(a) == 64


@respx.mock
def test_record_fixtures_writes_replayable_file(settings: Settings):
    settings.record_fixtures = True
    respx.get("https://example.com/").mock(
        return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
    )
    first = FetchPageTool().run({"url": "https://example.com/"}, settings)
    assert first.ok is True

    written = list(Path(settings.fixture_dir).glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert set(payload) >= {"url", "status", "headers", "text", "recorded_at"}

    # And the recording replays without the network.
    settings.record_fixtures = False
    settings.offline = True
    replay = FetchPageTool().run({"url": "https://example.com/"}, settings)
    assert replay.ok is True
    assert replay.data["title"] == "Acme Robotics — Warehouse Automation"
    assert replay.data["from_fixture"] is True


@respx.mock
def test_challenge_pages_are_never_recorded(settings: Settings):
    """Recording a bot-challenge page would poison every later --offline run."""
    settings.record_fixtures = True
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(202, text=CHALLENGE_PAGE))
    respx.post(DDG_LITE_URL).mock(return_value=httpx.Response(202, text=CHALLENGE_PAGE))

    obs = WebSearchTool().run({"query": "Ramp company"}, settings)

    assert obs.ok is False
    assert list(Path(settings.fixture_dir).glob("*.json")) == []


@respx.mock
def test_successful_search_is_recorded(settings: Settings):
    settings.record_fixtures = True
    respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, text=DDG_PAGE))

    assert WebSearchTool().run({"query": "Ramp company"}, settings).ok is True

    written = list(Path(settings.fixture_dir).glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["query"] == "ramp company"
    assert parse_ddg_html(payload["text"]), "recorded a page that does not parse"


@respx.mock
def test_recorded_fixtures_are_capped(settings: Settings):
    settings.record_fixtures = True
    huge = "<html><body>" + "<p>padding</p>" * 80_000 + "</body></html>"
    assert len(huge) > lea_http.MAX_FIXTURE_TEXT_CHARS
    respx.get("https://huge.example/").mock(
        return_value=httpx.Response(200, text=huge, headers={"content-type": "text/html"})
    )
    assert FetchPageTool().run({"url": "https://huge.example/"}, settings).ok is True

    payload = json.loads(next(Path(settings.fixture_dir).glob("*.json")).read_text())
    assert len(payload["text"]) == lea_http.MAX_FIXTURE_TEXT_CHARS
    assert payload["truncated_from"] == len(huge)


def test_robots_disallow_blocks_fetch(settings: Settings):
    settings.respect_robots = True
    with respx.mock:
        respx.get("https://blocked.example/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        )
        page = respx.get("https://blocked.example/secret").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        obs = FetchPageTool().run({"url": "https://blocked.example/secret"}, settings)
        assert not page.called
    assert obs.ok is False
    assert "robots" in (obs.error or "")


def test_robots_failure_fails_open(settings: Settings):
    settings.respect_robots = True
    with respx.mock:
        respx.get("https://open.example/robots.txt").mock(side_effect=httpx.ConnectError("no robots"))
        respx.get("https://open.example/").mock(
            return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
        )
        obs = FetchPageTool().run({"url": "https://open.example/"}, settings)
    assert obs.ok is True


# --------------------------------------------------------------------------
# fetch: extraction, truncation, errors
# --------------------------------------------------------------------------


def test_extract_meta_pulls_title_description_and_og():
    meta = extract_meta(HTMLParser(COMPANY_HTML))
    assert meta["title"] == "Acme Robotics — Warehouse Automation"
    assert meta["description"] == "Acme Robotics builds autonomous warehouse robots."
    assert meta["og:title"] == "Acme Robotics"


def test_extract_text_drops_chrome_keeps_link_text_and_order():
    text = extract_text(COMPANY_HTML)
    assert "console.log" not in text
    assert "skip me" not in text  # <header> dropped
    assert "footer noise" not in text  # <footer> dropped
    assert "Pricing" not in text  # <nav> dropped
    assert "founding story" in text  # link text preserved inline
    assert "- Fleet management" in text  # list markers
    # document order survives the walk
    assert text.index("Autonomous warehouses") < text.index("Fleet management")
    assert text.index("Fleet management") < text.index("Austin, TX")


def test_truncate_marks_the_cut():
    out = truncate("y" * 500, 100)
    assert out.endswith("[truncated]")
    assert len(out) < 200
    assert truncate("short", 100) == "short"


@respx.mock
def test_fetch_page_observation_shape(settings: Settings):
    respx.get("https://acme.example/").mock(
        return_value=httpx.Response(
            200,
            text=COMPANY_HTML,
            headers={"content-type": "text/html; charset=utf-8", "server": "cloudflare"},
        )
    )
    obs = FetchPageTool().run({"url": "https://acme.example/", "reason": "company overview"}, settings)

    assert obs.ok is True
    assert obs.tool is ToolName.FETCH_PAGE
    assert obs.data["title"] == "Acme Robotics — Warehouse Automation"
    assert "Acme Robotics builds robots" in obs.data["text"]
    # raw html is kept for tech_detect to consume without a second round trip
    assert "<script" in obs.data["html"]
    assert obs.sources == ["https://acme.example/"]
    assert obs.summary.startswith("Acme Robotics — Warehouse Automation")
    assert any("Austin" in loc or "Congress Avenue" in loc for loc in obs.data["locations"])


@respx.mock
def test_fetch_page_truncates_to_max_page_chars(settings: Settings):
    settings.max_page_chars = 200
    body = "<html><body><main>" + "<p>lorem ipsum dolor sit amet.</p>" * 400 + "</main></body></html>"
    respx.get("https://long.example/").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/html"})
    )
    obs = FetchPageTool().run({"url": "https://long.example/"}, settings)

    assert obs.ok is True
    assert obs.data["text"].endswith("[truncated]")
    assert len(obs.data["text"]) <= 200 + len("\n…[truncated]")


@respx.mock
def test_fetch_page_404_is_not_ok_with_actionable_error(settings: Settings):
    respx.get("https://acme.example/nope").mock(return_value=httpx.Response(404, text="Not Found"))
    obs = FetchPageTool().run({"url": "https://acme.example/nope"}, settings)

    assert obs.ok is False
    assert obs.error and obs.error.startswith("404")
    assert "try a different URL" in obs.error
    assert obs.data["text"] == ""
    assert obs.sources == []


@respx.mock
@pytest.mark.parametrize("status,needle", [(403, "blocked"), (429, "rate limited"), (500, "server error")])
def test_fetch_page_maps_status_codes_to_advice(settings: Settings, status: int, needle: str):
    respx.get("https://acme.example/x").mock(return_value=httpx.Response(status, text=""))
    obs = FetchPageTool().run({"url": "https://acme.example/x"}, settings)
    assert obs.ok is False
    assert needle in (obs.error or "")


@respx.mock
def test_fetch_page_rejects_non_html_content(settings: Settings):
    respx.get("https://acme.example/report.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})
    )
    obs = FetchPageTool().run({"url": "https://acme.example/report.pdf"}, settings)
    assert obs.ok is False
    assert "not HTML" in (obs.error or "")


def test_fetch_page_requires_a_url(settings: Settings):
    obs = FetchPageTool().run({"reason": "no url given"}, settings)
    assert obs.ok is False
    assert "url" in (obs.error or "")


@respx.mock
def test_fetch_page_adds_missing_scheme(settings: Settings):
    route = respx.get("https://acme.example/").mock(
        return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
    )
    obs = FetchPageTool().run({"url": "acme.example"}, settings)
    assert route.called
    assert obs.ok is True


# --------------------------------------------------------------------------
# tech detection
# --------------------------------------------------------------------------


def test_signature_table_is_broad_and_well_formed():
    assert len(SIGNATURES) >= 35
    names = [s.name for s in SIGNATURES]
    assert len(names) == len(set(names))
    for sig in SIGNATURES:
        assert sig.category and sig.name
        assert sig.scripts or sig.html or sig.headers or sig.cookies
        assert 0.0 < sig.confidence <= 1.0
    categories = {s.category for s in SIGNATURES}
    assert {"analytics", "marketing", "support", "hosting", "framework", "auth", "payments", "docs", "jobs"} <= categories


def test_detect_from_html_matches_script_srcs():
    tech = detect_from_html(COMPANY_HTML, {})
    found = {t.name for t in tech}
    assert {"Google Analytics 4", "Segment", "Stripe", "HubSpot", "Intercom", "Greenhouse", "PostHog", "Next.js"} <= found

    stripe = next(t for t in tech if t.name == "Stripe")
    assert stripe.category == "payments"
    assert "js.stripe.com" in stripe.evidence
    assert "script src" in stripe.evidence
    assert 0.0 < stripe.confidence <= 1.0


def test_detect_from_html_matches_headers_and_cookies():
    headers = {
        "Server": "cloudflare",
        "cf-ray": "87ab-EWR",
        "x-powered-by": "Next.js",
        "x-vercel-id": "iad1::abc",
        "set-cookie": "hubspotutk=abc123; Path=/",
    }
    tech = detect_from_html("<html></html>", headers)
    by_name = {t.name: t for t in tech}
    assert "Cloudflare" in by_name and "header" in by_name["Cloudflare"].evidence
    assert "Vercel" in by_name
    assert "Next.js" in by_name
    assert "cookie" in by_name["HubSpot"].evidence


def test_detect_from_html_matches_html_markers():
    html = """<html><body>
      <div id="__NEXT_DATA__"></div>
      <div data-wf-page="abc"></div>
      <div class="wp-content"></div>
      <a href="https://jobs.lever.co/acme">Jobs</a>
      <div ng-version="17.0.0"></div>
    </body></html>"""
    names = {t.name for t in detect_from_html(html, {})}
    assert {"Next.js", "Webflow", "WordPress", "Lever", "Angular"} <= names


def test_detect_from_html_is_quiet_on_a_bare_page():
    tech = detect_from_html("<html><body><p>hello</p></body></html>", {})
    assert tech == []


def test_detect_from_html_never_raises_on_garbage():
    assert detect_from_html("", {}) == []
    assert detect_from_html("<<<not html>>>", {"weird": None}) is not None


def test_header_evidence_outranks_html_evidence():
    # Next.js is detectable both ways; the header should win and carry more confidence.
    both = detect_from_html('<div id="__NEXT_DATA__"></div>', {"x-powered-by": "Next.js"})
    html_only = detect_from_html('<div id="__NEXT_DATA__"></div>', {})
    nx_both = next(t for t in both if t.name == "Next.js")
    nx_html = next(t for t in html_only if t.name == "Next.js")
    assert "header" in nx_both.evidence
    assert nx_both.confidence > nx_html.confidence


@respx.mock
def test_detect_tech_tool_observation_shape(settings: Settings):
    respx.get("https://acme.example/").mock(
        return_value=httpx.Response(
            200,
            text=COMPANY_HTML,
            headers={"content-type": "text/html", "server": "cloudflare", "cf-ray": "1a2b-EWR"},
        )
    )
    obs = DetectTechTool().run({"url": "https://acme.example/"}, settings)

    assert obs.ok is True
    assert obs.tool is ToolName.DETECT_TECH
    entries = obs.data["tech"]
    assert entries and all({"category", "name", "evidence", "confidence"} <= set(e) for e in entries)
    assert "Greenhouse" in {e["name"] for e in entries}
    assert obs.sources == ["https://acme.example/"]
    assert "[payments] Stripe" in obs.summary


@respx.mock
def test_detect_tech_reuses_the_fetched_response(settings: Settings):
    """fetch_page then detect_tech on the same URL must cost ONE request.

    The loop always does these back to back, so detect_tech has to stand on its own
    without re-hitting the site.
    """
    route = respx.get("https://acme.example/").mock(
        return_value=httpx.Response(
            200, text=COMPANY_HTML, headers={"content-type": "text/html", "server": "cloudflare"}
        )
    )
    page = FetchPageTool().run({"url": "https://acme.example/"}, settings)
    tech = DetectTechTool().run({"url": "https://acme.example/"}, settings)

    assert page.ok is True and tech.ok is True
    assert route.call_count == 1, "detect_tech re-fetched a page already in cache"
    assert "Greenhouse" in {t["name"] for t in tech.data["tech"]}


def test_detect_tech_works_standalone_offline_after_a_fetch(settings: Settings):
    """Offline, detect_tech must work on an already-fetched URL from the fixture."""
    settings.offline = True
    _write_fixture(settings, "https://acme.example/", None, COMPANY_HTML)

    page = FetchPageTool().run({"url": "https://acme.example/"}, settings)
    tech = DetectTechTool().run({"url": "https://acme.example/"}, settings)

    assert page.ok is True
    assert tech.ok is True
    assert tech.data["tech"], "tech detection produced nothing on a fetched page"


def test_detect_tech_description_does_not_require_a_prior_fetch(settings: Settings):
    spec = next(s for s in tool_specs(build_registry(settings)) if s["name"] == "detect_tech")
    text = spec["description"].lower()
    assert "does not need fetch_page" in text or "not need fetch_page" in text
    assert set(spec["args_schema"]) == {"url"}


@respx.mock
def test_detect_tech_tool_reports_fetch_failure(settings: Settings):
    respx.get("https://acme.example/gone").mock(return_value=httpx.Response(404, text=""))
    obs = DetectTechTool().run({"url": "https://acme.example/gone"}, settings)
    assert obs.ok is False
    assert obs.data["tech"] == []
    assert "404" in (obs.error or "")


@respx.mock
def test_detect_tech_tool_ok_with_empty_list_when_nothing_matches(settings: Settings):
    respx.get("https://plain.example/").mock(
        return_value=httpx.Response(200, text="<html><body>hi</body></html>", headers={"content-type": "text/html"})
    )
    obs = DetectTechTool().run({"url": "https://plain.example/"}, settings)
    assert obs.ok is True
    assert obs.data["tech"] == []


# --------------------------------------------------------------------------
# ATS slug extraction (tech_detect) — the key fetch_jobs runs on
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet,expected",
    [
        ('<a href="https://jobs.ashbyhq.com/vanta">Open roles</a>', ("ashby", "vanta")),
        ('fetch("https://api.ashbyhq.com/posting-api/job-board/vanta")', ("ashby", "vanta")),
        ('<a href="https://jobs.ashbyhq.com/ramp/a1b2c3">Role</a>', ("ashby", "ramp")),
        ('<a href="https://boards.greenhouse.io/ramp">Jobs</a>', ("greenhouse", "ramp")),
        ('<a href="https://job-boards.greenhouse.io/duolingo/jobs/77">x</a>', ("greenhouse", "duolingo")),
        ('<script src="https://boards.greenhouse.io/embed/job_board/js?for=acme">', ("greenhouse", "acme")),
        ('"https://boards-api.greenhouse.io/v1/boards/figma/jobs"', ("greenhouse", "figma")),
        ('<a href="https://jobs.lever.co/outreach">Careers</a>', ("lever", "outreach")),
        ('"https://api.lever.co/v0/postings/matchgroup?mode=json"', ("lever", "matchgroup")),
        ('<iframe src="https://apply.workable.com/epignosis/"></iframe>', ("workable", "epignosis")),
        ('<a href="https://acme.workable.com/jobs">Jobs</a>', ("workable", "acme")),
    ],
)
def test_extract_ats_board_slug_from_every_url_shape(snippet: str, expected: tuple[str, str]):
    boards = extract_ats_boards(snippet)
    assert (boards[0]["platform"], boards[0]["slug"]) == expected
    assert boards[0]["board_url"].endswith(expected[1])


def test_extract_ats_board_reads_the_url_as_well_as_the_body():
    boards = extract_ats_boards("", "https://jobs.ashbyhq.com/linear")
    assert boards[0]["slug"] == "linear"


def test_extract_ats_board_sees_through_escaped_json_urls():
    """Careers pages hide the board URL inside a hydration payload."""
    for escaped in (
        r'{"board":"https:\/\/jobs.lever.co\/outreach"}',
        r'{"board":"https://jobs.lever.co/outreach"}',
    ):
        assert extract_ats_boards(escaped)[0]["slug"] == "outreach"


def test_extract_ats_board_ignores_url_grammar_as_a_slug():
    """`boards.greenhouse.io/embed/...` must not yield the org slug 'embed'."""
    boards = extract_ats_boards('<script src="https://boards.greenhouse.io/embed/job_board.js"></script>')
    assert boards == []


def test_extract_ats_board_deduplicates_and_finds_nothing_in_noise():
    html = '<a href="https://jobs.ashbyhq.com/vanta">a</a><a href="https://jobs.ashbyhq.com/vanta">b</a>'
    assert len(extract_ats_boards(html)) == 1
    assert extract_ats_boards("<p>we are hiring!</p>") == []


@respx.mock
def test_detect_tech_publishes_the_board_and_remembers_it(settings: Settings):
    """detect_tech is where the slug is discovered; fetch_jobs reads it later."""
    respx.get("https://acme.example/careers").mock(
        return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
    )
    obs = DetectTechTool().run({"url": "https://acme.example/careers"}, settings)

    assert obs.ok is True
    assert obs.data["ats_boards"] == [
        {
            "ats": "Greenhouse",
            "platform": "greenhouse",
            "slug": "acme",
            "board_url": "https://job-boards.greenhouse.io/acme",
            "evidence": obs.data["ats_boards"][0]["evidence"],
        }
    ]
    # The summary has to name the tool the planner should reach for next.
    assert "fetch_jobs" in obs.summary
    assert recall_ats_boards("acme.example")[0]["slug"] == "acme"


# --------------------------------------------------------------------------
# fetch_jobs — parsing each ATS into one shape
# --------------------------------------------------------------------------

ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/{slug}"
GREENHOUSE_DEPTS_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/departments"
GREENHOUSE_JOBS_URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER_URL = "https://api.lever.co/v0/postings/{slug}?mode=json"
WORKABLE_URL = "https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true"

# Field names below are copied from live responses captured while building this
# tool; the recorded fixtures in examples/fixtures are the same shapes at scale.
ASHBY_PAYLOAD = {
    "apiVersion": "1",
    "jobs": [
        {
            "title": "Account Executive, Mid-Market", "department": "Sales", "team": "Commercial",
            "location": "San Francisco, CA", "isListed": True, "isRemote": False,
            "publishedAt": "2026-09-15T20:18:58.727+00:00",
            "descriptionHtml": "<p>" + "x" * 5000 + "</p>",
        },
        {
            "title": "Senior Backend Engineer", "department": "Engineering", "team": "Platform",
            "location": "Remote (US)", "isListed": True, "isRemote": True,
            "publishedAt": "2026-09-10T09:00:00.000+00:00",
            "descriptionPlain": "y" * 5000,
        },
        {
            "title": "Draft role nobody can see", "department": "Engineering",
            "location": "Dublin, Ireland", "isListed": False, "publishedAt": "2026-09-01T00:00:00Z",
        },
    ],
}

GREENHOUSE_PAYLOAD = {
    "departments": [
        {
            "id": 1, "name": "Engineering", "parent_id": None, "child_ids": [2],
            "jobs": [
                {"id": 10, "title": "Staff Engineer", "location": {"name": "New York, NY"},
                 "first_published": "2026-09-14T10:00:00-04:00"},
                {"id": 11, "title": "Engineering Manager", "location": {"name": "Remote - US"},
                 "first_published": "2026-08-02T10:00:00-04:00"},
            ],
        },
        # A child department repeats a parent's job: counted once, not twice.
        {"id": 2, "name": "Infrastructure", "parent_id": 1, "child_ids": [],
         "jobs": [{"id": 10, "title": "Staff Engineer", "location": {"name": "New York, NY"}}]},
        {"id": 3, "name": "Sales", "parent_id": None, "child_ids": [],
         "jobs": [{"id": 12, "title": "Enterprise AE", "location": {"name": "London, UK"},
                   "first_published": "2026-09-16T10:00:00-04:00"}]},
        {"id": 4, "name": "Nobody Is Hiring Here", "parent_id": None, "child_ids": [], "jobs": []},
    ]
}

LEVER_PAYLOAD = [
    {"text": "Solutions Consultant", "categories": {"department": "Sales", "team": "Pre-Sales",
                                                    "location": "Seattle, WA"},
     "createdAt": 1789748062697, "workplaceType": "onsite", "descriptionPlain": "z" * 5000},
    {"text": "Data Engineer", "categories": {"team": "Data", "location": "Remote - United States"},
     "createdAt": 1789748062697, "workplaceType": "remote"},
]

WORKABLE_PAYLOAD = {
    "name": "Acme Widgets",
    "jobs": [
        {"title": "Partner Marketing Lead", "department": "Marketing", "city": "Athens",
         "country": "Greece", "telecommuting": False, "published_on": "2026-09-07",
         "description": "<p>" + "q" * 5000 + "</p>"},
        {"title": "Cloud Engineer", "department": "", "function": "Engineering", "city": "Boston",
         "country": "United States", "telecommuting": True, "published_on": "2026-09-01"},
    ],
}


def test_parse_ashby_flattens_and_drops_unlisted():
    roles = parse_ashby(ASHBY_PAYLOAD)
    assert [r.title for r in roles] == ["Account Executive, Mid-Market", "Senior Backend Engineer"]
    assert roles[0].department == "Sales" and roles[0].location == "San Francisco, CA"
    assert roles[0].posted == "2026-09-15"
    assert roles[1].remote is True


def test_parse_greenhouse_deduplicates_jobs_shared_by_parent_and_child():
    roles = parse_greenhouse(GREENHOUSE_PAYLOAD)
    assert len(roles) == 3, "a job filed under a parent AND a child was double counted"
    assert {r.department for r in roles} == {"Engineering", "Sales"}
    assert {r.title for r in roles} == {"Staff Engineer", "Engineering Manager", "Enterprise AE"}


def test_parse_greenhouse_also_reads_the_flat_jobs_shape():
    roles = parse_greenhouse(
        {"jobs": [{"id": 1, "title": "Designer", "location": {"name": "Berlin"}}], "meta": {"total": 1}}
    )
    assert len(roles) == 1 and roles[0].location == "Berlin"
    assert roles[0].department == ""  # /jobs genuinely does not say


def test_parse_lever_falls_back_from_department_to_team():
    roles = parse_lever(LEVER_PAYLOAD)
    assert [r.department for r in roles] == ["Sales", "Data"]
    assert roles[0].posted == "2026-09-18"  # epoch milliseconds
    assert roles[1].remote is True


def test_parse_workable_builds_a_location_and_falls_back_to_function():
    roles = parse_workable(WORKABLE_PAYLOAD)
    assert roles[0].location == "Athens, Greece"
    assert roles[1].department == "Engineering"
    assert roles[1].remote is True


def test_parsers_never_choke_on_junk():
    for parse in (parse_ashby, parse_greenhouse, parse_lever, parse_workable):
        assert parse({}) == [] or parse({}) is not None
        assert parse({"jobs": [None, "nope", 7]}) == []
        assert parse([]) == []


# --------------------------------------------------------------------------
# fetch_jobs — aggregation
# --------------------------------------------------------------------------


def test_aggregate_counts_departments_and_orders_them_by_size():
    roles = (
        [Role("E%d" % i, "Engineering", "New York, NY") for i in range(4)]
        + [Role("S%d" % i, "Sales", "London") for i in range(6)]
        + [Role("D1", "", "new york, ny")]
    )
    stats = aggregate(roles)
    assert stats["open_roles"] == 11
    assert list(stats["departments"].items()) == [("Sales", 6), ("Engineering", 4), ("Unspecified", 1)]
    assert sum(stats["departments"].values()) == stats["open_roles"], "counts must reconcile"
    # Locations dedupe case-insensitively, keeping the first spelling, by frequency.
    assert stats["locations"] == ["London", "New York, NY"]


def test_aggregate_samples_titles_across_departments_not_just_the_biggest():
    roles = [Role("Eng %d" % i, "Engineering") for i in range(40)] + [Role("AE", "Sales")]
    titles = aggregate(roles)["titles"]
    assert len(titles) == 15, "title sample is not capped"
    assert "AE" in titles, "a one-role department was crowded out of the sample"


def test_aggregate_flags_recent_postings():
    from datetime import date, timedelta as td

    fresh = (date.today() - td(days=3)).isoformat()
    stale = (date.today() - td(days=400)).isoformat()
    stats = aggregate([Role("A", "Sales", posted=fresh), Role("B", "Sales", posted=stale)])
    assert stats["posted_last_30d"] == 1
    assert stats["latest_posted"] == fresh


def test_aggregate_handles_an_empty_board():
    stats = aggregate([])
    assert stats == {
        "open_roles": 0, "departments": {}, "locations": [], "titles": [],
        "remote_roles": 0, "latest_posted": "", "posted_last_30d": 0,
    }


def test_slugify_strips_punctuation_and_suffixes():
    assert slugify("Match Group, Inc.") == "matchgroup"
    assert slugify("Ramp") == "ramp"
    assert slugify("!!!") == ""


# --------------------------------------------------------------------------
# fetch_jobs — end to end per platform
# --------------------------------------------------------------------------


@respx.mock
def test_fetch_jobs_ashby_end_to_end(settings: Settings):
    respx.get(ASHBY_URL.format(slug="vanta")).mock(
        return_value=httpx.Response(200, json=ASHBY_PAYLOAD)
    )
    obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/vanta"}, settings)

    assert obs.ok is True and obs.tool is ToolName.FETCH_JOBS
    assert obs.data["open_roles"] == 2
    assert obs.data["ats"] == "Ashby" and obs.data["slug"] == "vanta"
    assert obs.data["departments"] == {"Engineering": 1, "Sales": 1}
    assert obs.data["board_url"] == "https://jobs.ashbyhq.com/vanta"
    assert obs.sources == ["https://jobs.ashbyhq.com/vanta"]
    assert obs.data["resolved_by"] == "board_url"
    assert "Vanta — 2 open roles on Ashby" in obs.summary
    # Descriptions are megabytes of HTML the LLM must never be handed.
    assert "descriptionHtml" not in json.dumps(obs.data)
    assert len(json.dumps(obs.data)) < 4000


@respx.mock
def test_fetch_jobs_greenhouse_end_to_end(settings: Settings):
    respx.get(GREENHOUSE_DEPTS_URL.format(slug="duolingo")).mock(
        return_value=httpx.Response(200, json=GREENHOUSE_PAYLOAD)
    )
    obs = FetchJobsTool().run({"company_or_url": "https://boards.greenhouse.io/duolingo"}, settings)

    assert obs.ok is True
    assert obs.data["open_roles"] == 3
    assert obs.data["departments"] == {"Engineering": 2, "Sales": 1}
    assert obs.data["company"] == "Duolingo"


@respx.mock
def test_fetch_jobs_lever_end_to_end(settings: Settings):
    respx.get(LEVER_URL.format(slug="outreach")).mock(
        return_value=httpx.Response(200, json=LEVER_PAYLOAD)
    )
    obs = FetchJobsTool().run({"company_or_url": "https://jobs.lever.co/outreach"}, settings)
    assert obs.ok is True
    assert obs.data["open_roles"] == 2 and obs.data["ats"] == "Lever"


@respx.mock
def test_fetch_jobs_workable_end_to_end(settings: Settings):
    respx.get(WORKABLE_URL.format(slug="acme")).mock(
        return_value=httpx.Response(200, json=WORKABLE_PAYLOAD)
    )
    obs = FetchJobsTool().run({"company_or_url": "https://apply.workable.com/acme"}, settings)
    assert obs.ok is True
    assert obs.data["open_roles"] == 2 and obs.data["ats"] == "Workable"


@respx.mock
def test_fetch_jobs_observation_data_shape_is_stable(settings: Settings):
    """Downstream extraction reads these keys by name; locking the contract."""
    respx.get(ASHBY_URL.format(slug="vanta")).mock(
        return_value=httpx.Response(200, json=ASHBY_PAYLOAD)
    )
    data = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/vanta"}, settings).data
    assert set(data) == {
        "company", "ats", "platform", "slug", "board_url", "api_url", "resolved_by",
        "truncated", "from_fixture", "open_roles", "departments", "locations",
        "titles", "remote_roles", "latest_posted", "posted_last_30d",
    }
    assert isinstance(data["open_roles"], int)
    assert isinstance(data["departments"], dict)
    assert isinstance(data["locations"], list) and isinstance(data["titles"], list)


# --------------------------------------------------------------------------
# fetch_jobs — degradation
# --------------------------------------------------------------------------


@respx.mock
def test_a_live_board_with_zero_roles_is_a_real_answer(settings: Settings):
    """'Not hiring' is a finding, not a failure — it must not read as an error."""
    respx.get(WORKABLE_URL.format(slug="hotjar")).mock(
        return_value=httpx.Response(200, json={"name": "Hotjar", "jobs": []})
    )
    obs = FetchJobsTool().run({"company_or_url": "https://apply.workable.com/hotjar"}, settings)

    assert obs.ok is True
    assert obs.error is None
    assert obs.data["open_roles"] == 0
    assert obs.data["departments"] == {} and obs.data["titles"] == []
    assert "0 open roles" in obs.summary and "not hiring publicly" in obs.summary


@respx.mock
def test_greenhouse_falls_back_to_the_flat_jobs_endpoint(settings: Settings):
    """An empty /departments may just be the wrong view of a live board."""
    depts = respx.get(GREENHOUSE_DEPTS_URL.format(slug="acme")).mock(
        return_value=httpx.Response(200, json={"departments": []})
    )
    jobs = respx.get(GREENHOUSE_JOBS_URL.format(slug="acme")).mock(
        return_value=httpx.Response(
            200, json={"jobs": [{"id": 1, "title": "Designer", "location": {"name": "Berlin"}}]}
        )
    )
    obs = FetchJobsTool().run({"company_or_url": "https://boards.greenhouse.io/acme"}, settings)

    assert depts.called and jobs.called
    assert obs.ok is True and obs.data["open_roles"] == 1


@respx.mock
def test_a_missing_or_private_board_degrades_cleanly(settings: Settings):
    respx.get(ASHBY_URL.format(slug="nosuchorg")).mock(return_value=httpx.Response(404, text="Not Found"))
    obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/nosuchorg"}, settings)

    assert obs.ok is False
    assert "no public Ashby board" in (obs.error or "")
    assert obs.data["open_roles"] is None and obs.data["departments"] == {}
    assert obs.sources == []


@respx.mock
def test_a_404_stops_the_platform_instead_of_trying_its_other_endpoint(settings: Settings):
    """A wrong slug is wrong on every endpoint; do not spend a second request."""
    depts = respx.get(GREENHOUSE_DEPTS_URL.format(slug="nosuchorg")).mock(
        return_value=httpx.Response(404, json={"status": 404, "error": "Job not found"})
    )
    jobs = respx.get(GREENHOUSE_JOBS_URL.format(slug="nosuchorg")).mock(
        return_value=httpx.Response(404, json={"status": 404, "error": "Job not found"})
    )
    obs = FetchJobsTool().run({"company_or_url": "https://boards.greenhouse.io/nosuchorg"}, settings)

    assert obs.ok is False
    assert depts.called and not jobs.called


@respx.mock
def test_an_oversized_board_is_refused_rather_than_parsed(settings: Settings):
    from lea.tools import jobs as lea_jobs

    huge = '{"jobs":[' + ",".join(['{"title":"x"}'] * 200) + "]}"
    respx.get(ASHBY_URL.format(slug="huge")).mock(
        return_value=httpx.Response(200, text=huge, headers={"content-type": "application/json"})
    )
    original = lea_jobs.MAX_RESPONSE_CHARS
    lea_jobs.MAX_RESPONSE_CHARS = 100
    try:
        obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/huge"}, settings)
    finally:
        lea_jobs.MAX_RESPONSE_CHARS = original

    assert obs.ok is False
    assert "over the" in (obs.error or "") and "limit" in (obs.error or "")


@respx.mock
def test_a_long_board_is_counted_in_full_but_reported_in_brief(settings: Settings):
    """Hundreds of roles must not become hundreds of lines in the prompt."""
    payload = {
        "jobs": [
            {"title": f"Engineer {i}", "department": f"Team {i % 30}", "location": f"City {i % 40}",
             "isListed": True, "descriptionHtml": "<p>" + "x" * 2000 + "</p>"}
            for i in range(400)
        ]
    }
    respx.get(ASHBY_URL.format(slug="big")).mock(return_value=httpx.Response(200, json=payload))
    obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/big"}, settings)

    assert obs.ok is True
    assert obs.data["open_roles"] == 400  # the count is exact
    assert len(obs.data["titles"]) == 15
    assert len(obs.data["locations"]) <= 12
    assert len(obs.data["departments"]) <= 25
    assert len(obs.summary) < 400


@respx.mock
def test_unparseable_json_fails_without_raising(settings: Settings):
    respx.get(ASHBY_URL.format(slug="broken")).mock(
        return_value=httpx.Response(200, text="<html>bot check</html>")
    )
    obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/broken"}, settings)
    assert obs.ok is False and "unparseable JSON" in (obs.error or "")


def test_fetch_jobs_rejects_a_missing_argument(settings: Settings):
    obs = FetchJobsTool().run({}, settings)
    assert obs.ok is False and "company_or_url" in (obs.error or "")
    assert FetchJobsTool().run({"company_or_url": "   "}, settings).ok is False


@respx.mock
def test_fetch_jobs_never_raises_on_a_transport_error(settings: Settings):
    respx.get(ASHBY_URL.format(slug="vanta")).mock(side_effect=httpx.ConnectError("boom"))
    obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/vanta"}, settings)
    assert isinstance(obs, Observation) and obs.ok is False


# --------------------------------------------------------------------------
# fetch_jobs — resolving which board to read
# --------------------------------------------------------------------------


@respx.mock
def test_fetch_jobs_uses_the_board_detect_tech_already_found(settings: Settings):
    """The whole point of the tech_detect handoff: no second page scrape."""
    page = respx.get("https://acme.example/careers").mock(
        return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
    )
    board = respx.get(GREENHOUSE_DEPTS_URL.format(slug="acme")).mock(
        return_value=httpx.Response(200, json=GREENHOUSE_PAYLOAD)
    )
    DetectTechTool().run({"url": "https://acme.example/careers"}, settings)
    page_calls = page.call_count

    obs = FetchJobsTool().run({"company_or_url": "acme.example"}, settings)

    assert obs.ok is True and board.called
    assert obs.data["resolved_by"] == "detected_ats"
    assert page.call_count == page_calls, "fetch_jobs re-scraped a page detect_tech already read"


@respx.mock
def test_fetch_jobs_finds_the_board_on_the_careers_page(settings: Settings):
    respx.get("https://acme.example/careers").mock(
        return_value=httpx.Response(200, text=COMPANY_HTML, headers={"content-type": "text/html"})
    )
    respx.get(GREENHOUSE_DEPTS_URL.format(slug="acme")).mock(
        return_value=httpx.Response(200, json=GREENHOUSE_PAYLOAD)
    )
    obs = FetchJobsTool().run({"company_or_url": "https://acme.example"}, settings)
    assert obs.ok is True and obs.data["resolved_by"] == "careers_page"


@respx.mock
def test_slug_guessing_is_capped_at_two_platforms(settings: Settings):
    """No ATS detected must not mean brute-forcing every platform we know."""
    respx.get("https://acme.example/careers").mock(return_value=httpx.Response(404))
    respx.get("https://acme.example/jobs").mock(return_value=httpx.Response(404))
    tried = {
        "ashby": respx.get(ASHBY_URL.format(slug="acme")).mock(return_value=httpx.Response(404)),
        "greenhouse": respx.get(GREENHOUSE_DEPTS_URL.format(slug="acme")).mock(
            return_value=httpx.Response(404)
        ),
        "lever": respx.get(LEVER_URL.format(slug="acme")).mock(return_value=httpx.Response(404)),
        "workable": respx.get(WORKABLE_URL.format(slug="acme")).mock(return_value=httpx.Response(404)),
    }
    obs = FetchJobsTool().run({"company_or_url": "acme.example"}, settings)

    assert obs.ok is False
    assert sum(1 for route in tried.values() if route.called) == 2
    assert not tried["lever"].called and not tried["workable"].called


@respx.mock
def test_a_bare_company_name_is_slugified_into_a_board_guess(settings: Settings):
    respx.get(ASHBY_URL.format(slug="matchgroup")).mock(
        return_value=httpx.Response(200, json=ASHBY_PAYLOAD)
    )
    obs = FetchJobsTool().run({"company_or_url": "Match Group, Inc."}, settings)
    assert obs.ok is True
    assert obs.data["slug"] == "matchgroup" and obs.data["resolved_by"] == "slug_guess"
    assert obs.data["company"] == "Match Group, Inc."


def test_a_bare_name_does_not_trigger_a_careers_page_fetch(settings: Settings):
    with respx.mock(assert_all_called=False) as mock:
        pages = mock.route(path="/careers").mock(
            side_effect=AssertionError("guessed a domain out of a bare company name")
        )
        mock.route(host__in=("api.ashbyhq.com", "boards-api.greenhouse.io")).mock(
            return_value=httpx.Response(404)
        )
        FetchJobsTool().run({"company_or_url": "Acme"}, settings)
        assert not pages.called


def test_resolve_candidates_prefers_a_board_url_over_everything(settings: Settings):
    candidates, _ = resolve_candidates("https://jobs.lever.co/outreach", settings)
    assert [(c.platform, c.slug, c.how) for c in candidates] == [("lever", "outreach", "board_url")]


# --------------------------------------------------------------------------
# fetch_jobs — SSRF, offline, fixtures
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target", ["http://localhost/careers", "http://169.254.169.254/", "http://127.0.0.1:8080"]
)
def test_fetch_jobs_careers_probe_is_ssrf_guarded(settings: Settings, target: str):
    with respx.mock(assert_all_called=False) as mock:
        blocked = mock.route(host__in=("localhost", "169.254.169.254", "127.0.0.1")).mock(
            side_effect=AssertionError("SSRF guard did not fire")
        )
        # The slug-guess fallback still runs afterwards; it must go nowhere either.
        mock.route().mock(return_value=httpx.Response(404))
        obs = FetchJobsTool().run({"company_or_url": target}, settings)
        assert not blocked.called
    assert obs.ok is False


def test_fetch_jobs_careers_probe_is_guarded_on_a_rebinding_host(
    settings: Settings, stub_dns: dict[str, list[str]]
):
    stub_dns["evil.example"] = ["127.0.0.1"]
    with respx.mock(assert_all_called=False) as mock:
        blocked = mock.route(host="evil.example").mock(
            side_effect=AssertionError("SSRF guard did not fire on a public name resolving inward")
        )
        obs = FetchJobsTool().run({"company_or_url": "evil.example"}, settings)
        assert not blocked.called
    assert obs.ok is False


def test_fetch_jobs_offline_serves_a_fixture(settings: Settings):
    settings.offline = True
    _write_fixture(
        settings,
        ASHBY_URL.format(slug="vanta"),
        None,
        json.dumps(ASHBY_PAYLOAD),
        headers={"content-type": "application/json"},
    )
    with respx.mock(assert_all_called=False) as mock:
        blocked = mock.route().mock(side_effect=AssertionError("network was used in offline mode"))
        obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/vanta"}, settings)
        assert not blocked.called

    assert obs.ok is True
    assert obs.data["open_roles"] == 2 and obs.data["from_fixture"] is True


def test_fetch_jobs_offline_without_a_fixture_says_so(settings: Settings):
    settings.offline = True
    obs = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/vanta"}, settings)
    assert obs.ok is False and "offline" in (obs.error or "")


def test_prune_descriptions_keeps_the_counts_and_drops_the_prose():
    raw = json.dumps(ASHBY_PAYLOAD)
    pruned = prune_descriptions(raw)
    assert len(pruned) < len(raw) / 5
    reparsed = json.loads(pruned)  # still valid JSON, unlike a truncated blob
    assert len(reparsed["jobs"]) == 3
    assert "descriptionHtml" not in pruned and "descriptionPlain" not in pruned
    assert parse_ashby(reparsed) == parse_ashby(ASHBY_PAYLOAD)
    assert prune_descriptions("not json at all") == "not json at all"


@respx.mock
def test_recording_a_board_writes_a_small_replayable_fixture(settings: Settings):
    settings.record_fixtures = True
    respx.get(ASHBY_URL.format(slug="vanta")).mock(return_value=httpx.Response(200, json=ASHBY_PAYLOAD))
    live = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/vanta"}, settings)
    assert live.ok is True

    written = [p for p in Path(settings.fixture_dir).glob("*.json")]
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert "descriptionHtml" not in payload["text"], "job descriptions were recorded"
    assert len(payload["text"]) < 1500

    settings.record_fixtures = False
    settings.offline = True
    replay = FetchJobsTool().run({"company_or_url": "https://jobs.ashbyhq.com/vanta"}, settings)
    assert replay.data["open_roles"] == live.data["open_roles"]


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_build_registry_returns_four_runnable_tools(settings: Settings):
    registry = build_registry(settings)
    assert set(registry) == {
        ToolName.WEB_SEARCH, ToolName.FETCH_PAGE, ToolName.DETECT_TECH, ToolName.FETCH_JOBS
    }
    assert ToolName.FINISH not in registry
    for name, tool in registry.items():
        assert tool.name is name
        assert tool.description
        assert isinstance(tool.args_schema, dict) and tool.args_schema
        assert callable(tool.run)


def test_tool_specs_include_finish(settings: Settings):
    specs = tool_specs(build_registry(settings))
    names = [s["name"] for s in specs]
    assert names == ["web_search", "fetch_page", "detect_tech", "fetch_jobs", "finish"]
    for spec in specs:
        assert set(spec) == {"name", "description", "args_schema"}
        assert spec["description"]
    finish = specs[-1]
    assert "reason" in finish["args_schema"]


def test_registry_tools_satisfy_the_tool_protocol(settings: Settings):
    from lea.tools.base import Tool

    for tool in build_registry(settings).values():
        assert isinstance(tool, Tool)


@pytest.fixture
def demo_settings() -> Settings:
    """Points at the corpus actually committed to the repo."""
    s = Settings(offline=True, search_provider="fixture")
    if not Path(s.fixture_dir).is_dir() or not list(Path(s.fixture_dir).glob("*.json")):
        pytest.skip("no fixtures committed to the repo")
    return s


@pytest.mark.parametrize(
    "query,expected_host",
    [
        ("vanta compliance automation company", "vanta.com"),
        ("ramp corporate cards company", "ramp.com"),
        # reworded — an LLM will not reproduce a recorded query verbatim
        ("Vanta security compliance platform", "vanta.com"),
        ("Vanta hiring roles", "vanta.com"),
        ("Ramp careers", "ramp.com"),
    ],
)
def test_offline_demo_search_answers_demo_queries(demo_settings: Settings, query: str, expected_host: str):
    obs = build_registry(demo_settings)[ToolName.WEB_SEARCH].run({"query": query}, demo_settings)
    assert obs.ok is True, obs.error
    assert obs.data["results"]
    assert any(expected_host in u for u in obs.sources)


def test_offline_search_still_refuses_an_unrelated_query(demo_settings: Settings):
    """Fuzzy matching must not answer anything with whatever it has lying around."""
    obs = build_registry(demo_settings)[ToolName.WEB_SEARCH].run(
        {"query": "best sourdough starter recipe"}, demo_settings
    )
    assert obs.ok is False


@pytest.mark.parametrize(
    "url,expected_tech",
    [
        ("https://www.vanta.com/", {"Webflow", "HubSpot", "Google Analytics 4"}),
        # the hiring signal that makes the demo worth watching
        ("https://www.vanta.com/careers", {"Ashby"}),
        ("https://ramp.com/", {"Next.js", "Vercel"}),
    ],
)
def test_offline_demo_pages_fetch_and_detect(demo_settings: Settings, url: str, expected_tech: set):
    registry = build_registry(demo_settings)

    page = registry[ToolName.FETCH_PAGE].run({"url": url}, demo_settings)
    assert page.ok is True, page.error
    assert len(page.data["text"]) > 500, "fixture produced almost no readable text"
    assert page.data["title"]

    tech = registry[ToolName.DETECT_TECH].run({"url": url}, demo_settings)
    assert tech.ok is True
    assert expected_tech <= {t["name"] for t in tech.data["tech"]}


@pytest.mark.parametrize(
    "target,ats,least",
    [
        # Live recordings: three of the four platforms, at real board sizes.
        ("https://jobs.ashbyhq.com/vanta", "Ashby", 50),
        ("https://jobs.ashbyhq.com/ramp", "Ashby", 50),
        ("https://boards.greenhouse.io/duolingo", "Greenhouse", 50),
        ("https://jobs.lever.co/outreach", "Lever", 20),
        # Hand-authored, so the fourth parser is exercised offline too.
        ("https://apply.workable.com/acmewidgets", "Workable", 5),
    ],
)
def test_offline_demo_boards_produce_headcount_signal(
    demo_settings: Settings, target: str, ats: str, least: int
):
    """The whole point: --offline answers 'how big are they?' with a number."""
    obs = build_registry(demo_settings)[ToolName.FETCH_JOBS].run(
        {"company_or_url": target}, demo_settings
    )
    assert obs.ok is True, obs.error
    assert obs.data["ats"] == ats
    assert obs.data["open_roles"] >= least
    assert sum(obs.data["departments"].values()) == obs.data["open_roles"]
    assert obs.data["titles"] and len(obs.data["titles"]) <= 15
    assert obs.data["from_fixture"] is True


def test_offline_demo_resolves_a_board_from_a_bare_company_name(demo_settings: Settings):
    """Ramp's careers page is too big to fixture whole, so the slug guess carries it."""
    obs = build_registry(demo_settings)[ToolName.FETCH_JOBS].run(
        {"company_or_url": "Ramp"}, demo_settings
    )
    assert obs.ok is True, obs.error
    assert obs.data["slug"] == "ramp" and obs.data["open_roles"] > 0


def test_committed_board_fixtures_carry_no_job_descriptions():
    """A recorded board must be counts and titles, never posting HTML."""
    d = Path(Settings().fixture_dir)
    if not d.is_dir():
        pytest.skip("no fixtures committed to the repo")
    offenders = []
    for p in d.glob("*.json"):
        raw = p.read_text()
        if "job-board" in raw[:300] or "v0/postings" in raw[:300]:
            if "descriptionHtml" in raw or "descriptionPlain" in raw:
                offenders.append(p.name)
    assert not offenders, f"job descriptions leaked into fixtures: {offenders}"


def test_committed_fixtures_stay_small():
    """Fixtures live in the repo; a multi-megabyte blob makes it unreviewable."""
    d = Path(Settings().fixture_dir)
    if not d.is_dir():
        pytest.skip("no fixtures committed to the repo")
    oversized = {
        p.name: p.stat().st_size
        for p in d.glob("*.json")
        # JSON escaping inflates the stored text somewhat past the raw cap
        if p.stat().st_size > lea_http.MAX_FIXTURE_TEXT_CHARS * 2
    }
    assert not oversized, f"fixtures exceed the size budget: {oversized}"
