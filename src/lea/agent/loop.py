"""The autonomous research loop.

There is deliberately no research *plan* in this file. The loop only:
  decide -> guard -> dispatch -> observe -> update -> repeat
The LLM picks every step; `policy` says what is affordable and what repeats work.
Grep this module for a company name, a hardcoded URL or an ordered list of queries
and you will not find one — that is the point.

Every module that lands in parallel (tools, prompts, llm, enrich) is imported
*inside* a function through a small `_get_*` indirection. Two reasons: a half-built
sibling module cannot break import of the CLI, and tests can monkeypatch one seam
instead of the whole world.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlsplit

from lea.agent.policy import (
    STOP_BLOCKED,
    STOP_COMPLETE,
    STOP_ERROR,
    Budget,
    allowed_tools,
    derive_gaps,
    gap_labels,
    MODELS_HINT,
    is_redundant,
    learned_anything,
    llm_error_detail,
    llm_failure_stop_reason,
    looks_like_stale_model,
    premature_finish_reason,
    should_stop,
    stop_class,
    transient_failure_hint,
    truncate_words,
)
from lea.config import Settings
from lea.models import (
    AgentAction,
    CompanyRecord,
    EnrichmentResult,
    ICPFit,
    ICPProfile,
    Observation,
    OutreachHook,
    RunStats,
    TechStack,
    ToolName,
    TraceStep,
)

OnStep = Callable[[TraceStep], None]
Evidence = list[tuple[str, str]]

# Observation text longer than this is trimmed before it goes into the trace; the
# full text still reaches extraction via the evidence list.
OBS_PREVIEW_CHARS = 600

# Hosts that are never a company's own domain, so we don't mistake a search hit
# for the target's website.
_AGGREGATORS = {
    "linkedin.com", "crunchbase.com", "wikipedia.org", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "youtube.com", "github.com", "medium.com",
    "bloomberg.com", "reuters.com", "techcrunch.com", "forbes.com", "pitchbook.com",
    "glassdoor.com", "indeed.com", "google.com", "bing.com", "duckduckgo.com",
    "yahoo.com", "apollo.io", "zoominfo.com", "producthunt.com", "reddit.com",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Seams (monkeypatched in tests, lazily imported in production)
# --------------------------------------------------------------------------


def _get_registry(settings: Settings) -> dict[ToolName, Any]:
    from lea.tools.registry import build_registry  # type: ignore

    return build_registry(settings)


def _get_llm(settings: Settings) -> Any:
    from lea.llm.factory import get_llm  # type: ignore

    return get_llm(settings)


def _get_prompts() -> tuple[str, Callable[..., str]]:
    from lea.prompts import AGENT_SYSTEM, build_agent_context  # type: ignore

    return AGENT_SYSTEM, build_agent_context


def _call_extract(
    target: str, evidence: Evidence, llm: Any, settings: Settings, base: CompanyRecord
) -> CompanyRecord:
    from lea.enrich.extract import extract_record  # type: ignore

    return extract_record(target, evidence, llm, settings, base=base)


def _call_icp(
    record: CompanyRecord, icp: ICPProfile, evidence: Evidence, llm: Any, settings: Settings
) -> ICPFit:
    from lea.enrich.icp import score_icp  # type: ignore

    return score_icp(record, icp, evidence, llm, settings)


def _call_hooks(
    record: CompanyRecord, fit: Optional[ICPFit], evidence: Evidence, llm: Any, settings: Settings
) -> list[OutreachHook]:
    from lea.enrich.hooks import generate_hooks  # type: ignore

    return generate_hooks(record, fit, evidence, llm, settings)


def _default_icp(settings: Settings) -> ICPProfile:
    from lea.enrich.icp import load_or_default_icp  # type: ignore

    return load_or_default_icp(None, settings)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _host(url: str) -> str:
    if not url:
        return ""
    raw = url if "://" in url else "http://" + url
    host = (urlsplit(raw).netloc or "").lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _registrable(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _is_aggregator(host: str) -> bool:
    return _registrable(host) in _AGGREGATORS


def _truncate(text: str, limit: int = OBS_PREVIEW_CHARS) -> str:
    # Word-boundary: these strings are error messages as often as page text, and a
    # message cut mid-token is one a user cannot search for.
    return truncate_words(text, limit)


def _stringify_results(results: Iterable[Any]) -> str:
    """Flatten search hits into plain text extraction can read."""
    lines: list[str] = []
    for r in results:
        if isinstance(r, dict):
            title = r.get("title") or ""
            url = r.get("url") or r.get("link") or ""
            snippet = r.get("snippet") or r.get("content") or r.get("description") or ""
        else:
            title = getattr(r, "title", "") or ""
            url = getattr(r, "url", "") or ""
            snippet = getattr(r, "snippet", "") or ""
        lines.append(f"{title}\n{url}\n{snippet}".strip())
    return "\n\n".join(l for l in lines if l)


def _evidence_from(obs: Observation, action: AgentAction) -> Evidence:
    """Pull citable (source_url, text) pairs out of a tool observation.

    Tools are owned by another agent, so we probe a few plausible payload keys and
    always fall back to the summary rather than assuming a shape.
    """
    data = obs.data or {}
    source = obs.sources[0] if obs.sources else ""
    if not source:
        source = (
            str(action.args.get("url") or action.args.get("page") or "")
            or (f"search:{action.args.get('query', '')}" if action.args.get("query") else "")
            or action.tool.value
        )

    for key in ("text", "content", "markdown", "body", "page_text", "html_text"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return [(source, val)]

    results = data.get("results") or data.get("hits") or data.get("items")
    if isinstance(results, list) and results:
        text = _stringify_results(results)
        if text.strip():
            return [(source, text)]

    tech = data.get("tech") or data.get("technologies") or data.get("detected")
    if isinstance(tech, list) and tech:
        return [(source, "Detected technologies: " + _stringify_tech(tech))]

    if obs.summary.strip():
        return [(source, obs.summary)]
    return []


def _stringify_tech(items: Iterable[Any]) -> str:
    out: list[str] = []
    for t in items:
        if isinstance(t, dict):
            out.append(str(t.get("name") or t.get("tech") or t))
        else:
            out.append(str(getattr(t, "name", t)))
    return ", ".join(o for o in out if o)


def _tech_entries(data: dict[str, Any], source: str) -> list[TechStack]:
    raw = data.get("tech") or data.get("technologies") or data.get("detected") or []
    entries: list[TechStack] = []
    if not isinstance(raw, list):
        return entries
    for item in raw:
        try:
            if isinstance(item, TechStack):
                entries.append(item)
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("tech") or "").strip()
                if not name:
                    continue
                entries.append(
                    TechStack(
                        category=str(item.get("category") or "other"),
                        name=name,
                        evidence=str(item.get("evidence") or source),
                        confidence=float(item.get("confidence") or 0.5),
                    )
                )
            elif isinstance(item, str) and item.strip():
                entries.append(TechStack(category="other", name=item.strip(), evidence=source))
        except Exception:
            continue
    return entries


def looks_like_domain(target: str) -> bool:
    """Did the user type a domain/URL rather than a company name?"""
    t = (target or "").strip()
    if not t:
        return False
    if "://" in t or "/" in t:
        return True
    return "." in t and " " not in t


def _seed_record(record: CompanyRecord, target: str) -> None:
    """Seed ONLY what the user themselves told us.

    Careful: `merge_records` gives the base precedence for every scalar it already
    has, so anything seeded here permanently pre-empts extraction — including
    Agent D's `sanitize_company_name` / `choose_domain`. So the bar for seeding is
    not "probably right", it is "authoritative", and the user's own input is the
    only thing that clears it:

      "Vanta"    -> company_name, because that is what they called it.
      "ramp.com" -> domain, because that is the site they pointed at. The NAME is
                    left unset so extraction can sanitize a real one ("Ramp"),
                    rather than us asserting the domain string as the name.

    Everything else the loop learns mid-run is a guess and stays out of the record.
    """
    candidate = (target or "").strip()
    if not candidate:
        return

    if looks_like_domain(candidate):
        host = _host(candidate)
        if host and not _is_aggregator(host):
            record.domain = record.domain or host
        return

    # A plain name typed by the user beats anything we could infer from a page.
    record.company_name = record.company_name or candidate


def _tentative_domain(obs: Observation) -> str:
    """A *guess* at the company's domain from an observation.

    Used only to keep the prompt's gap list honest as the run progresses. It is
    deliberately NOT written to the record: a host that merely isn't a known
    aggregator can still be a news site or a vendor page, and `choose_domain` at
    synthesis does this properly by matching the host against the target.
    """
    data = obs.data or {}
    explicit = data.get("domain") or data.get("site") or data.get("url")
    host = _host(str(explicit)) if explicit else ""
    if not host:
        for url in obs.sources or []:
            h = _host(url)
            if h and not _is_aggregator(h):
                host = h
                break
    return host if host and not _is_aggregator(host) else ""


def _update_record(record: CompanyRecord, action: AgentAction, obs: Observation) -> None:
    """Post-observation updates to the AUTHORITATIVE record.

    Only two things qualify, and both are safe under `merge_records` because it
    unions lists instead of letting the base win:

      sources     — URLs we actually read.
      tech_stack  — detect_tech matched real markers in real HTML and carries its
                    own evidence string, which beats anything extraction infers.

    Scalar identity fields (company_name, domain, industry, ...) are NOT set here.
    Setting them would win the merge and silence the sanitizers. That is exactly
    how a page <title> ended up as the company name.
    """
    data = obs.data or {}

    for url in obs.sources:
        if url and url not in record.sources:
            record.sources.append(url)

    if action.tool == ToolName.DETECT_TECH and obs.ok:
        source = obs.sources[0] if obs.sources else str(action.args.get("url", ""))
        for entry in _tech_entries(data, source):
            if not any(e.name.lower() == entry.name.lower() for e in record.tech_stack):
                record.tech_stack.append(entry)


def provider_label(llm: Any, settings: Settings) -> str:
    """What to show as the LLM in the footer stats.

    For an OpenAI-compatible provider this carries the capability rung the client
    actually settled on — `google:gemini-2.0-flash (json_object)`. The rung is the
    single best predictor of output quality on free models, so hiding it means a
    user reads flat ICP reasoning as the agent's judgement rather than as the
    endpoint refusing tool-calling. getattr-guarded so anthropic/stub are untouched.
    """
    name = getattr(llm, "name", None) or settings.resolved_llm()
    rung = getattr(llm, "capability", None)
    if not rung:
        return str(name)
    model = getattr(llm, "model", "") or ""
    return f"{name}:{model} ({rung})" if model else f"{name} ({rung})"


def _steering_view(record: CompanyRecord, tentative_domain: str) -> CompanyRecord:
    """What the model is shown: the record plus the loop's working guesses.

    Keeping this separate from the record is the whole point — guesses are good
    enough to steer the next step ("stop hunting for the domain") but not good
    enough to publish.
    """
    if record.domain or not tentative_domain:
        return record
    view = record.model_copy(deep=True)
    view.domain = tentative_domain
    return view


def _emit(step: TraceStep, trace: list[TraceStep], on_step: Optional[OnStep]) -> None:
    trace.append(step)
    if on_step is None:
        return
    try:
        on_step(step)
    except Exception:
        # A broken renderer must never take down a research run.
        pass


def _summarize_obs(obs: Observation, action: AgentAction) -> str:
    if obs.summary.strip():
        return _truncate(obs.summary)
    ev = _evidence_from(obs, action)
    return _truncate(ev[0][1]) if ev else ""


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_agent(
    target: str,
    settings: Settings,
    icp: ICPProfile | None = None,
    on_step: OnStep | None = None,
) -> EnrichmentResult:
    """Research one company autonomously and return the record plus the full trace."""
    t0 = time.monotonic()
    started_at = _now()

    budget = Budget.from_settings(settings)
    trace: list[TraceStep] = []
    evidence: Evidence = []
    record = CompanyRecord(input=target)
    _seed_record(record, target)

    stop_reason = ""
    pushbacks = 0  # times we've refused an empty-handed FINISH
    tentative_domain = ""  # a guess, for steering only — never written to the record
    llm_errors: list[str] = []  # failed LLM calls, for diagnosing a dead endpoint
    registry: dict[ToolName, Any] = {}
    llm: Any = None

    # --- bootstrap -------------------------------------------------------
    try:
        registry = _get_registry(settings)
    except Exception as exc:
        _emit(
            TraceStep(
                n=1, tool=ToolName.FINISH, ok=False,
                thought="bootstrap: build tool registry",
                observation=f"tool registry unavailable: {exc}",
                error=str(exc),
            ),
            trace, on_step,
        )
        stop_reason = f"{STOP_ERROR}: tool registry unavailable: {exc}"

    if not stop_reason:
        try:
            llm = _get_llm(settings)
        except Exception as exc:
            _emit(
                TraceStep(
                    n=len(trace) + 1, tool=ToolName.FINISH, ok=False,
                    thought="bootstrap: construct LLM client",
                    observation=f"LLM unavailable: {exc}",
                    error=str(exc),
                ),
                trace, on_step,
            )
            stop_reason = f"{STOP_ERROR}: LLM unavailable: {exc}"

    system = ""
    build_context: Optional[Callable[..., str]] = None
    if not stop_reason:
        try:
            system, build_context = _get_prompts()
        except Exception as exc:
            stop_reason = f"{STOP_ERROR}: prompts unavailable: {exc}"

    # --- decide / act / observe -----------------------------------------
    while not stop_reason:
        # Gaps and stop conditions read the steering view, so a domain we've merely
        # inferred still narrows the to-do list without contaminating the output.
        steering = _steering_view(record, tentative_domain)
        reason = should_stop(steering, budget, trace)
        if reason:
            stop_reason = reason
            break

        gaps = derive_gaps(steering)
        specs = allowed_tools(budget, registry)

        try:
            context = build_context(target, steering, trace, budget.remaining(), gap_labels(gaps))  # type: ignore[misc]
            action, usage = llm.decide(system, context, specs)
            tokens_in, tokens_out = int(usage.tokens_in), int(usage.tokens_out)
        except Exception as exc:
            # A failed decision still burns a step, otherwise a persistently broken
            # client would spin forever.
            budget.consume(None)
            _emit(
                TraceStep(
                    n=len(trace) + 1, tool=ToolName.FINISH, ok=False,
                    thought="decide next step",
                    observation=f"the model failed to produce a usable action: {exc}",
                    error=str(exc),
                ),
                trace, on_step,
            )
            continue

        # A failed LLM call arrives disguised as a calm decision to FINISH. Unmask it
        # before the finish handling below, or a 404ing endpoint reads as "the agent
        # judged the evidence sufficient" — the most confusing failure in the system.
        llm_error = llm_error_detail(action)
        if llm_error:
            llm_errors.append(llm_error)
            budget.consume(None)
            note = f" — {MODELS_HINT}." if looks_like_stale_model(llm_error) else ""
            _emit(
                TraceStep(
                    n=len(trace) + 1, tool=ToolName.FINISH, ok=False,
                    thought="decide next step",
                    observation=f"{llm_error}{note}", error=llm_error,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                ),
                trace, on_step,
            )
            continue

        if action.tool == ToolName.FINISH:
            # Guardrail, not a plan: refuse to finish having learned nothing while
            # budget remains. Two failed searches is a transient block, not a
            # conclusion about the company — finishing there yields an all-None
            # record dressed up with a neutral score. Bounded by MAX_FINISH_PUSHBACKS
            # so an insistent model can still stop.
            objection = premature_finish_reason(budget, trace, pushbacks)
            if objection:
                pushbacks += 1
                budget.consume(None)
                _emit(
                    TraceStep(
                        n=len(trace) + 1, tool=ToolName.FINISH, args=action.args, ok=False,
                        thought=action.thought, observation=objection, error="premature finish",
                        tokens_in=tokens_in, tokens_out=tokens_out,
                    ),
                    trace, on_step,
                )
                continue

            _emit(
                TraceStep(
                    n=len(trace) + 1, tool=ToolName.FINISH, ok=True,
                    thought=action.thought, args=action.args,
                    observation="agent judged the evidence sufficient; moving to synthesis.",
                    tokens_in=tokens_in, tokens_out=tokens_out,
                ),
                trace, on_step,
            )
            budget.consume(ToolName.FINISH)
            stop_reason = (
                f"{STOP_COMPLETE}: agent called finish"
                if learned_anything(trace)
                else f"{STOP_BLOCKED}: agent finished without a single successful research step"
            )
            break

        # Redundancy guard. We do NOT skip silently: the reason is written into the
        # trace as a failed step, and the trace is what the next prompt shows the
        # model. Blocking without feedback just produces the same action again.
        redundant = is_redundant(action, trace)
        if redundant:
            budget.consume(None)  # costs a step, not a search/fetch
            _emit(
                TraceStep(
                    n=len(trace) + 1, tool=action.tool, args=action.args, ok=False,
                    thought=action.thought, observation=redundant, error="redundant",
                    tokens_in=tokens_in, tokens_out=tokens_out,
                ),
                trace, on_step,
            )
            continue

        tool = registry.get(action.tool)
        if tool is None:
            budget.consume(None)
            _emit(
                TraceStep(
                    n=len(trace) + 1, tool=action.tool, args=action.args, ok=False,
                    thought=action.thought,
                    observation=f"tool '{action.tool.value}' is not available; choose from the offered tools.",
                    error="unknown tool",
                    tokens_in=tokens_in, tokens_out=tokens_out,
                ),
                trace, on_step,
            )
            continue

        # --- dispatch ----------------------------------------------------
        t_step = time.monotonic()
        try:
            obs = tool.run(action.args, settings)
            if not isinstance(obs, Observation):  # defensive: a tool returning junk
                raise TypeError(f"tool returned {type(obs).__name__}, expected Observation")
        except Exception as exc:
            # Tools are contractually not supposed to raise, but one bad step must
            # never kill a run: convert it into a failed step and keep going.
            obs = Observation(
                tool=action.tool, ok=False,
                summary=f"{type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
        latency = obs.latency_ms or int((time.monotonic() - t_step) * 1000)
        budget.consume(action.tool)

        if obs.ok:
            for src, text in _evidence_from(obs, action):
                if text and text.strip():
                    evidence.append((src, text))
            _update_record(record, action, obs)
            if not tentative_domain:
                tentative_domain = _tentative_domain(obs)

        step = TraceStep(
            n=len(trace) + 1, tool=action.tool, args=action.args, ok=obs.ok,
            thought=action.thought,
            observation=_summarize_obs(obs, action),
            sources=list(obs.sources), error=obs.error,
            latency_ms=latency, tokens_in=tokens_in, tokens_out=tokens_out,
        )
        if not obs.ok:
            # The trace is the next prompt's context, so the *specific* error has to
            # land in it along with a nudge that a failed tool is not an answer.
            # Without this the model reads any failure as "research is impossible".
            step.observation = _truncate(step.observation + transient_failure_hint(step), 900)
        _emit(step, trace, on_step)

    # If the brain never answered, say so specifically. "The agent gave up" and "your
    # model id 404s" look identical from the outside, and only one of them is the
    # user's to fix.
    if llm_errors and not learned_anything(trace) and stop_class(stop_reason) != STOP_ERROR:
        stop_reason = llm_failure_stop_reason(
            llm_errors,
            provider=str(getattr(llm, "name", "") or settings.resolved_llm()),
            model=str(getattr(llm, "model", "") or ""),
        )

    # --- synthesis -------------------------------------------------------
    # Gated on evidence. Scoring an empty record produces a neutral-looking tier that
    # reads as a real assessment of the company; refusing to score is the honest
    # output. A blocked run must look blocked.
    if not learned_anything(trace) and not evidence:
        if stop_class(stop_reason) not in (STOP_ERROR, STOP_BLOCKED):
            stop_reason = f"{STOP_BLOCKED}: no evidence gathered — every research step failed"
        record.unknowns = gap_labels(derive_gaps(record))
        _emit(
            TraceStep(
                n=len(trace) + 1, tool=ToolName.FINISH, ok=False,
                thought="synthesize: nothing to synthesize.",
                observation=(
                    "skipped extraction, ICP scoring and hooks: no research step succeeded, so "
                    "there is no evidence to reason over. Scoring here would invent a result."
                ),
                error="no evidence",
            ),
            trace, on_step,
        )
    else:
        profile = icp
        if profile is None:
            try:
                profile = _default_icp(settings)
            except Exception:
                profile = None

        record = _synthesis_extract(target, record, evidence, llm, settings, trace, on_step)
        fit = _synthesis_icp(record, profile, evidence, llm, settings, trace, on_step)
        if fit is not None:
            record.icp_fit = fit
        hooks = _synthesis_hooks(record, fit, evidence, llm, settings, trace, on_step)
        if hooks:
            record.hooks = hooks

        # Honesty pass: whatever is still missing goes on the record, not into a guess.
        if not record.unknowns:
            record.unknowns = gap_labels(derive_gaps(record))
        for src, _ in evidence:
            if src and not src.startswith("search:") and src not in record.sources:
                record.sources.append(src)

    stats = RunStats(
        steps=len(trace),
        searches=budget.searches,
        fetches=budget.fetches,
        tokens_in=sum(s.tokens_in for s in trace),
        tokens_out=sum(s.tokens_out for s in trace),
        wall_ms=int((time.monotonic() - t0) * 1000),
        stop_reason=stop_reason or f"{STOP_COMPLETE}: run finished",
        llm_provider=provider_label(llm, settings),
        search_provider=settings.resolved_search(),
    )

    return EnrichmentResult(
        record=record, trace=trace, stats=stats,
        started_at=started_at, finished_at=_now(),
    )


# --------------------------------------------------------------------------
# Synthesis steps
# --------------------------------------------------------------------------


def _synthesis_extract(
    target: str, record: CompanyRecord, evidence: Evidence, llm: Any,
    settings: Settings, trace: list[TraceStep], on_step: Optional[OnStep],
) -> CompanyRecord:
    n_sources = len({src for src, _ in evidence})
    t = time.monotonic()
    thought = f"synthesize: extract the structured record from {n_sources} source(s)."
    try:
        if llm is None or not evidence:
            raise RuntimeError("no evidence gathered")
        out = _call_extract(target, evidence, llm, settings, record)
        if not isinstance(out, CompanyRecord):
            raise TypeError("extract_record did not return a CompanyRecord")
        filled = [
            f for f in ("company_name", "domain", "industry", "employee_range", "hq_city")
            if getattr(out, f, None)
        ]
        _emit(
            TraceStep(
                n=len(trace) + 1, tool=ToolName.FINISH, ok=True, thought=thought,
                observation=f"resolved {len(filled)} core field(s): {', '.join(filled) or 'none'}; "
                            f"{len(out.signals)} signal(s), {len(out.tech_stack)} tech entries.",
                latency_ms=int((time.monotonic() - t) * 1000),
            ),
            trace, on_step,
        )
        return out
    except Exception as exc:
        _emit(
            TraceStep(
                n=len(trace) + 1, tool=ToolName.FINISH, ok=False, thought=thought,
                observation=f"extraction failed, keeping fields gathered during the run: {exc}",
                error=str(exc), latency_ms=int((time.monotonic() - t) * 1000),
            ),
            trace, on_step,
        )
        return record


def _synthesis_icp(
    record: CompanyRecord, profile: Optional[ICPProfile], evidence: Evidence, llm: Any,
    settings: Settings, trace: list[TraceStep], on_step: Optional[OnStep],
) -> Optional[ICPFit]:
    t = time.monotonic()
    name = getattr(profile, "name", "default")
    thought = f"synthesize: score the record against the '{name}' ICP."
    try:
        if profile is None or llm is None:
            raise RuntimeError("no ICP profile available")
        fit = _call_icp(record, profile, evidence, llm, settings)
        if not isinstance(fit, ICPFit):
            raise TypeError("score_icp did not return an ICPFit")
        _emit(
            TraceStep(
                n=len(trace) + 1, tool=ToolName.FINISH, ok=True, thought=thought,
                observation=f"tier {fit.tier} · {fit.score:.0f}/100 across {len(fit.criteria)} criteria.",
                latency_ms=int((time.monotonic() - t) * 1000),
            ),
            trace, on_step,
        )
        return fit
    except Exception as exc:
        _emit(
            TraceStep(
                n=len(trace) + 1, tool=ToolName.FINISH, ok=False, thought=thought,
                observation=f"ICP scoring failed: {exc}", error=str(exc),
                latency_ms=int((time.monotonic() - t) * 1000),
            ),
            trace, on_step,
        )
        return None


def _synthesis_hooks(
    record: CompanyRecord, fit: Optional[ICPFit], evidence: Evidence, llm: Any,
    settings: Settings, trace: list[TraceStep], on_step: Optional[OnStep],
) -> list[OutreachHook]:
    t = time.monotonic()
    thought = "synthesize: draft outreach hooks grounded in the evidence."
    try:
        if llm is None:
            raise RuntimeError("no LLM available")
        hooks = list(_call_hooks(record, fit, evidence, llm, settings))
        if hooks:
            summary = f"drafted {len(hooks)} hook(s): " + "; ".join(h.angle for h in hooks[:3])
        else:
            summary = "no hook was well-grounded enough to draft."
        _emit(
            TraceStep(
                n=len(trace) + 1, tool=ToolName.FINISH, ok=True, thought=thought,
                observation=summary,
                latency_ms=int((time.monotonic() - t) * 1000),
            ),
            trace, on_step,
        )
        return hooks
    except Exception as exc:
        _emit(
            TraceStep(
                n=len(trace) + 1, tool=ToolName.FINISH, ok=False, thought=thought,
                observation=f"hook generation failed: {exc}", error=str(exc),
                latency_ms=int((time.monotonic() - t) * 1000),
            ),
            trace, on_step,
        )
        return []


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------


def run_batch(
    targets: Iterable[str],
    settings: Settings,
    icp: ICPProfile | None = None,
    on_result: Callable[[EnrichmentResult], None] | None = None,
    on_step: OnStep | None = None,
) -> list[EnrichmentResult]:
    """Enrich many targets sequentially. One target blowing up never aborts the batch."""
    results: list[EnrichmentResult] = []
    for target in targets:
        started = _now()
        try:
            result = run_agent(target, settings, icp=icp, on_step=on_step)
        except Exception as exc:
            result = EnrichmentResult(
                record=CompanyRecord(input=target, unknowns=["run failed before any research"]),
                trace=[
                    TraceStep(
                        n=1, tool=ToolName.FINISH, ok=False,
                        thought="run this target",
                        observation=f"{type(exc).__name__}: {exc}",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                ],
                stats=RunStats(
                    stop_reason=f"error: {type(exc).__name__}: {exc}",
                    llm_provider=settings.resolved_llm(),
                    search_provider=settings.resolved_search(),
                ),
                started_at=started, finished_at=_now(),
            )
        results.append(result)
        if on_result is not None:
            try:
                on_result(result)
            except Exception:
                pass
    return results


__all__ = ["run_agent", "run_batch"]
