"""Rendering the agent's work so a human can audit it.

The pitch of this project is "it decides its own next step". That claim is only
credible if you can watch it decide, so every step is printed as it happens:
the thought, the call, the observation, whether it worked. Same data is written to
JSON and to a markdown research log, and `replay` re-renders a saved run with no
network at all — which is how you demo this on a call in an airport.

Styling stays inside the 16 basic terminal colors. No 256-color or truecolor
assumptions, no emoji, so it looks the same over ssh as it does in iTerm.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.box import ROUNDED, SIMPLE
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lea.agent.policy import STOP_BLOCKED, STOP_COMPLETE, STOP_ERROR, stop_class
from lea.models import CompanyRecord, EnrichmentResult, ToolName, TraceStep

# Below this, an ICP score is not an assessment — it is a shrug with a number on it.
LOW_CONFIDENCE = 0.15

_OUTCOME_STYLE = {
    STOP_COMPLETE: ("COMPLETE", "green"),
    "exhausted": ("OUT OF BUDGET", "yellow"),
    STOP_BLOCKED: ("BLOCKED", "red"),
    STOP_ERROR: ("ERROR", "red"),
}

# tool -> (label, color, glyph). Glyphs are plain ASCII/box-drawing, not emoji.
_TOOL_STYLE: dict[str, tuple[str, str, str]] = {
    ToolName.WEB_SEARCH.value: ("search", "cyan", ">"),
    ToolName.FETCH_PAGE.value: ("fetch", "blue", "v"),
    ToolName.DETECT_TECH.value: ("tech", "magenta", "#"),
    ToolName.FINISH.value: ("synth", "green", "*"),
}

_TIER_COLOR = {"A": "bright_green", "B": "green", "C": "yellow", "D": "red"}

OBS_PREVIEW_LINES = 3
OBS_LINE_WIDTH = 96


def _style_for(step: TraceStep) -> tuple[str, str, str]:
    return _TOOL_STYLE.get(step.tool.value, ("step", "white", "-"))


def _call_signature(step: TraceStep) -> str:
    """`web_search("ramp.com funding 2025")` — the call as you'd write it in code."""
    args = step.args or {}
    if not args:
        return f"{step.tool.value}()"
    primary = None
    for key in ("query", "url", "q", "reason", "page"):
        if isinstance(args.get(key), str) and args[key].strip():
            primary = args[key].strip()
            break
    if primary is None:
        primary = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
        return f"{step.tool.value}({primary})"
    if len(primary) > 78:
        primary = primary[:77] + "…"
    extras = [k for k in args if k not in ("query", "url", "q", "reason", "page")]
    tail = f", +{len(extras)} arg" + ("s" if len(extras) > 1 else "") if extras else ""
    return f'{step.tool.value}("{primary}"{tail})'


def _observation_lines(
    text: str, limit: int = OBS_PREVIEW_LINES, width: int = OBS_LINE_WIDTH
) -> list[str]:
    """Collapse an observation into a few readable lines.

    We wrap by hand rather than letting rich reflow, so the preview can never spill
    past `limit` lines and every line keeps its gutter prefix.
    """
    raw = " ".join((text or "").split("\n"))
    raw = " ".join(raw.split())
    if not raw:
        return []
    width = max(30, width)
    lines: list[str] = []
    while raw and len(lines) < limit:
        chunk, raw = raw[:width], raw[width:]
        if raw and not raw.startswith(" ") and " " in chunk:
            cut = chunk.rfind(" ")
            chunk, raw = chunk[:cut], chunk[cut + 1:] + raw
        lines.append(chunk.strip())
    if raw.strip():
        lines[-1] = lines[-1].rstrip() + "…"
    return lines


# --------------------------------------------------------------------------
# Live step rendering
# --------------------------------------------------------------------------


def render_step(step: TraceStep, console: Console) -> None:
    """Print one plan->act->observe cycle as it completes."""
    label, color, glyph = _style_for(step)
    mark = Text("ok", style="green") if step.ok else Text("fail", style="red")

    header = Text()
    header.append(f"{step.n:>3} ", style="bold white")
    header.append(f"{glyph} ", style=color)
    header.append(f"{label:<6}", style=f"bold {color}")
    header.append("  ")
    header.append_text(mark)
    if step.latency_ms:
        header.append(f"  {step.latency_ms}ms", style="dim")
    if step.tokens_in or step.tokens_out:
        header.append(f"  {step.tokens_in}+{step.tokens_out} tok", style="dim")
    console.print(header)

    # Padding keeps the indent on wrapped continuation lines too, so a long thought
    # doesn't fall back to column 0 and break the visual gutter.
    if step.thought:
        console.print(Padding(Text(step.thought.strip(), style="dim italic"), (0, 0, 0, 6)))

    console.print(Padding(Text(_call_signature(step), style=color), (0, 0, 0, 6)))

    avail = max(30, (console.width or 80) - 10)
    body = step.observation or step.error or ""
    for line in _observation_lines(body, width=avail):
        out = Text("      ")
        out.append("│ ", style="dim")
        out.append(line, style="" if step.ok else "red")
        console.print(out, overflow="ignore", crop=True)

    if step.sources:
        shown = ", ".join(step.sources[:2])
        if len(step.sources) > 2:
            shown += f" (+{len(step.sources) - 2} more)"
        src = Text("      ")
        src.append("│ ", style="dim")
        src.append(shown[:avail], style="dim underline")
        console.print(src, overflow="ignore", crop=True)

    console.print()


def render_header(target: str, console: Console, settings: Any | None = None) -> None:
    """Banner printed before the first step, so the trace has context."""
    line = Text()
    line.append("researching ", style="dim")
    line.append(target, style="bold white")
    if settings is not None:
        try:
            line.append(
                f"   [llm={settings.resolved_llm()} search={settings.resolved_search()}"
                f"{' offline' if settings.offline else ''}]",
                style="dim",
            )
        except Exception:
            pass
    console.print()
    console.print(line)
    console.print()


# --------------------------------------------------------------------------
# Final report
# --------------------------------------------------------------------------


def _fmt(value: Any, dash: str = "unknown") -> str:
    if value is None or value == "" or value == []:
        return dash
    return str(value)


def _employees(rec: CompanyRecord) -> str:
    if rec.employee_range and rec.employee_estimate:
        return f"{rec.employee_range} (~{rec.employee_estimate})"
    return _fmt(rec.employee_range or rec.employee_estimate)


def _hq(rec: CompanyRecord) -> str:
    parts = [p for p in (rec.hq_city, rec.hq_country) if p]
    return ", ".join(parts) if parts else "unknown"


def successful_research_steps(result: EnrichmentResult) -> list[TraceStep]:
    """Steps that actually learned something. Synthesis steps don't count."""
    return [s for s in result.trace if s.ok and s.tool != ToolName.FINISH]


def is_ungrounded(result: EnrichmentResult) -> bool:
    """True when nothing in this result is backed by a source the agent read.

    Such a run must never be rendered as an assessment of the company — the fields
    are all None and any score attached to them is an artifact of the scorer's
    defaults, not a finding.
    """
    if stop_class(result.stats.stop_reason) in (STOP_BLOCKED, STOP_ERROR):
        return True
    return not result.record.sources and not successful_research_steps(result)


def _is_low_confidence(result: EnrichmentResult) -> bool:
    fit = result.record.icp_fit
    if fit is None:
        return False
    return fit.confidence < LOW_CONFIDENCE or not result.record.sources


def _no_research_panel(result: EnrichmentResult) -> Panel:
    """Replaces the ICP header when the run established nothing.

    Deliberately loud and score-free: the failure mode this guards against is a
    reader skimming a tier letter and believing the agent assessed the company.
    """
    rec = result.record
    failures = [s for s in result.trace if not s.ok and s.error != "premature finish"]
    body = Text()
    body.append("NO RESEARCH SUCCEEDED\n", style="bold red")
    body.append(
        f"Nothing was established about {rec.input}. "
        f"{len(successful_research_steps(result))} of {result.stats.steps} steps returned usable "
        f"data, and 0 sources were read.\n",
        style="white",
    )
    body.append("This is not an assessment of the company — no score is shown, because "
                "there is nothing to score.\n", style="dim")
    body.append("\nwhy: ", style="bold white")
    body.append(result.stats.stop_reason, style="red")
    if failures:
        err = (failures[0].error or failures[0].observation or "").strip()
        body.append("\nfirst error: ", style="bold white")
        body.append(err[:180], style="dim")
    body.append("\n\nwhat to try:\n", style="bold white")
    body.append("  - retry: search providers rate-limit, and the block is usually transient\n", style="dim")
    body.append("  - set TAVILY_API_KEY (or SERPER/BRAVE) for search that does not get blocked\n", style="dim")
    body.append("  - run with --offline to use the recorded fixtures instead of the network\n", style="dim")
    body.append("  - lea doctor  to see which providers and keys are actually in play", style="dim")
    return Panel(body, box=ROUNDED, border_style="red", padding=(0, 1))


def _header_panel(result: EnrichmentResult) -> Panel:
    rec = result.record
    fit = rec.icp_fit
    title = Text(rec.company_name or rec.input, style="bold white")
    body = Text()
    body.append_text(title)
    if rec.domain:
        body.append(f"   {rec.domain}", style="cyan")
    body.append("\n")
    if fit is not None:
        weak = _is_low_confidence(result)
        # A score the scorer itself has no confidence in is rendered as provisional,
        # never as a clean tier letter someone can skim and quote.
        color = "yellow" if weak else _TIER_COLOR.get(fit.tier, "white")
        body.append("ICP ", style="dim")
        body.append(f"tier {fit.tier}", style=f"bold {color}")
        body.append(f"  {fit.score:.0f}/100", style=color)
        body.append(f"   confidence {fit.confidence:.0%}", style="dim")
        if weak:
            body.append("\n")
            body.append(
                "PROVISIONAL — scored with little or no direct evidence; do not treat this "
                "tier as a finding.",
                style="bold yellow",
            )
        if fit.summary:
            body.append("\n")
            body.append(fit.summary.strip(), style="")
        if fit.disqualifiers_hit:
            body.append("\n")
            body.append("disqualifiers: " + ", ".join(fit.disqualifiers_hit), style="red")
    else:
        body.append("ICP fit not scored", style="dim")
    if rec.description:
        body.append("\n\n")
        body.append(rec.description.strip(), style="dim")
    return Panel(body, box=ROUNDED, border_style="blue", padding=(0, 1))


def _firmographics_table(rec: CompanyRecord) -> Table:
    t = Table(box=SIMPLE, show_header=False, pad_edge=False, expand=False)
    t.add_column("field", style="dim", no_wrap=True)
    t.add_column("value", style="white", overflow="fold")
    rows = [
        ("input", rec.input),
        ("company", _fmt(rec.company_name)),
        ("domain", _fmt(rec.domain)),
        ("industry", _fmt(rec.industry)),
        ("sub-industry", _fmt(rec.sub_industry, "-")),
        ("employees", _employees(rec)),
        ("hq", _hq(rec)),
        ("founded", _fmt(rec.founded_year)),
    ]
    for k, v in rows:
        t.add_row(k, v)
    return t


def _signals_group(rec: CompanyRecord) -> Group:
    items: list[Any] = []
    if not rec.signals:
        return Group(Text("  none found", style="dim"))
    for s in rec.signals:
        line = Text("  ")
        line.append(f"[{s.kind}] ", style="bold yellow")
        line.append(s.headline.strip(), style="white")
        if s.date:
            line.append(f"  ({s.date})", style="dim")
        items.append(line)
        if s.detail:
            d = Text("      ")
            d.append(s.detail.strip(), style="dim")
            items.append(d)
        if s.source_url:
            u = Text("      ")
            u.append(s.source_url, style="dim underline")
            items.append(u)
    return Group(*items)


def _tech_group(rec: CompanyRecord) -> Group:
    if not rec.tech_stack:
        return Group(Text("  none detected", style="dim"))
    by_cat: dict[str, list[str]] = {}
    for t in rec.tech_stack:
        by_cat.setdefault(t.category or "other", []).append(t.name)
    items = []
    for cat in sorted(by_cat):
        line = Text("  ")
        line.append(f"{cat:<14}", style="magenta")
        line.append(", ".join(sorted(set(by_cat[cat]))), style="white")
        items.append(line)
    return Group(*items)


def _icp_table(rec: CompanyRecord) -> Optional[Table]:
    fit = rec.icp_fit
    if fit is None or not fit.criteria:
        return None
    t = Table(box=SIMPLE, show_header=True, header_style="bold dim", expand=False)
    t.add_column("criterion", style="white", no_wrap=True)
    t.add_column("score", justify="right")
    t.add_column("wt", justify="right", style="dim")
    t.add_column("reasoning", style="dim", overflow="fold")
    for c in fit.criteria:
        color = "green" if c.score >= 7 else "yellow" if c.score >= 4 else "red"
        t.add_row(c.key, Text(f"{c.score:.1f}", style=color), f"{c.weight:g}", c.reasoning or "-")
    return t


def _hooks_group(rec: CompanyRecord) -> Group:
    if not rec.hooks:
        return Group(Text("  no hooks generated", style="dim"))
    items: list[Any] = []
    for h in rec.hooks:
        head = Text("  ")
        head.append(h.angle.strip(), style="bold green")
        items.append(head)
        quote = Text("    ")
        quote.append("“" + h.hook.strip() + "”", style="italic white")
        items.append(quote)
        if h.grounded_in:
            g = Text("    ")
            g.append("grounded in: ", style="dim")
            g.append(h.grounded_in.strip(), style="dim")
            if h.source_url:
                g.append(f"  {h.source_url}", style="dim underline")
            items.append(g)
        items.append(Text(""))
    return Group(*items)


def _section(console: Console, title: str) -> None:
    t = Text()
    t.append(title.upper(), style="bold white")
    console.print(t)


def render_result(result: EnrichmentResult, console: Console) -> None:
    """The final report: everything the agent established, and what it did not."""
    rec = result.record
    ungrounded = is_ungrounded(result)
    console.print(_no_research_panel(result) if ungrounded else _header_panel(result))
    console.print()

    _section(console, "firmographics")
    console.print(_firmographics_table(rec))

    _section(console, "signals")
    console.print(_signals_group(rec))
    console.print()

    _section(console, "tech stack")
    console.print(_tech_group(rec))
    console.print()

    icp_table = _icp_table(rec)
    if icp_table is not None and not ungrounded:
        _section(console, "icp scoring")
        if _is_low_confidence(result):
            console.print(Text("  provisional — graded without direct evidence", style="yellow"))
        console.print(icp_table)

    _section(console, "outreach hooks")
    console.print(_hooks_group(rec))

    unknowns = Text()
    unknowns.append("unknowns: ", style="bold yellow")
    unknowns.append(", ".join(rec.unknowns) if rec.unknowns else "none", style="dim")
    console.print(unknowns)

    if rec.sources:
        src = Text()
        src.append("sources:  ", style="bold white")
        src.append(f"{len(rec.sources)} cited", style="dim")
        console.print(src)
        for u in rec.sources[:10]:
            console.print(Text("  " + u, style="dim underline"))
        if len(rec.sources) > 10:
            console.print(Text(f"  (+{len(rec.sources) - 10} more)", style="dim"))

    console.print()
    console.print(_stats_line(result))


def _stats_line(result: EnrichmentResult) -> Text:
    s = result.stats
    if not (s.stop_reason or "").strip():
        label, color = "UNKNOWN", "dim"
    else:
        label, color = _OUTCOME_STYLE.get(stop_class(s.stop_reason), ("DONE", "white"))
    t = Text()
    t.append(f"{s.steps} steps", style="bold white")
    t.append(f" · {s.searches} searches · {s.fetches} pages", style="dim")
    t.append(f" · {s.tokens_in}+{s.tokens_out} tokens", style="dim")
    t.append(f" · {s.wall_ms / 1000:.1f}s", style="dim")
    t.append(f" · llm={s.llm_provider} search={s.search_provider}", style="dim")
    t.append("\noutcome: ", style="dim")
    # The class is spelled out so "finished" is never confused with "gave up".
    t.append(label, style=f"bold {color}")
    t.append(f" — {s.stop_reason}", style="dim italic")
    if "(prose)" in (s.llm_provider or ""):
        # Bottom rung: the endpoint honoured neither tool-calling nor JSON mode, so
        # every structured field was scraped back out of free text. Flat ICP
        # reasoning here is a property of the endpoint, not of the company.
        t.append(
            "\nnote: this model supports neither tool-calling nor JSON mode, so structured "
            "output was parsed from prose — expect weaker ICP reasoning and hooks. "
            "`lea doctor --models` lists stronger ids your provider serves.",
            style="yellow",
        )
    return t


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_json(result: EnrichmentResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_result(path: str | Path) -> EnrichmentResult:
    raw = Path(path).read_text(encoding="utf-8")
    return EnrichmentResult.model_validate(json.loads(raw))


def _md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").strip()


def save_trace_markdown(result: EnrichmentResult, path: str | Path) -> Path:
    """A research log a human can read top-to-bottom, or paste into a README."""
    rec = result.record
    fit = rec.icp_fit
    s = result.stats
    L: list[str] = []

    name = rec.company_name or rec.input
    L.append(f"# Research log: {name}")
    L.append("")
    L.append(f"*Input:* `{rec.input}`  ")
    L.append(f"*Started:* {result.started_at}  ")
    L.append(f"*Finished:* {result.finished_at or 'n/a'}  ")
    L.append(
        f"*Run:* {s.steps} steps · {s.searches} searches · {s.fetches} pages · "
        f"{s.tokens_in + s.tokens_out} tokens · {s.wall_ms / 1000:.1f}s  "
    )
    L.append(f"*Providers:* llm=`{s.llm_provider}` search=`{s.search_provider}`  ")
    label, _ = _OUTCOME_STYLE.get(stop_class(s.stop_reason), ("DONE", ""))
    L.append(f"*Outcome:* **{label}** — {s.stop_reason}")
    L.append("")

    if is_ungrounded(result):
        L.append("> **NO RESEARCH SUCCEEDED.** No source was read, so the fields below are "
                 "empty and no ICP score is reported. This is not an assessment of the "
                 "company. Retry, set `TAVILY_API_KEY`, or run with `--offline` to use "
                 "recorded fixtures.")
        L.append("")

    L.append("## Result")
    L.append("")
    L.append("| field | value |")
    L.append("| --- | --- |")
    L.append(f"| company | {_md_escape(_fmt(rec.company_name))} |")
    L.append(f"| domain | {_md_escape(_fmt(rec.domain))} |")
    L.append(f"| industry | {_md_escape(_fmt(rec.industry))} |")
    L.append(f"| employees | {_md_escape(_employees(rec))} |")
    L.append(f"| hq | {_md_escape(_hq(rec))} |")
    L.append(f"| founded | {_md_escape(_fmt(rec.founded_year))} |")
    if fit is not None:
        L.append(f"| ICP | tier {fit.tier} — {fit.score:.0f}/100 |")
    L.append("")
    if rec.description:
        L.append(f"> {_md_escape(rec.description)}")
        L.append("")

    if rec.signals:
        L.append("## Signals")
        L.append("")
        for sig in rec.signals:
            date = f" ({sig.date})" if sig.date else ""
            src = f" — [source]({sig.source_url})" if sig.source_url else ""
            L.append(f"- **[{sig.kind}]** {_md_escape(sig.headline)}{date}{src}")
            if sig.detail:
                L.append(f"  - {_md_escape(sig.detail)}")
        L.append("")

    if rec.tech_stack:
        L.append("## Tech stack")
        L.append("")
        by_cat: dict[str, list[str]] = {}
        for t in rec.tech_stack:
            by_cat.setdefault(t.category or "other", []).append(t.name)
        for cat in sorted(by_cat):
            L.append(f"- **{cat}**: {', '.join(sorted(set(by_cat[cat])))}")
        L.append("")

    if fit is not None and fit.criteria:
        L.append("## ICP scoring")
        L.append("")
        L.append("| criterion | score | weight | reasoning |")
        L.append("| --- | ---: | ---: | --- |")
        for c in fit.criteria:
            L.append(
                f"| {_md_escape(c.key)} | {c.score:.1f} | {c.weight:g} | {_md_escape(c.reasoning) or '-'} |"
            )
        L.append("")

    if rec.hooks:
        L.append("## Outreach hooks")
        L.append("")
        for h in rec.hooks:
            L.append(f"**{_md_escape(h.angle)}**")
            L.append("")
            L.append(f"> {_md_escape(h.hook)}")
            if h.grounded_in:
                src = f" ([source]({h.source_url}))" if h.source_url else ""
                L.append(f">")
                L.append(f"> *grounded in: {_md_escape(h.grounded_in)}{src}*")
            L.append("")

    L.append("## Agent trace")
    L.append("")
    L.append("Every step below was chosen by the model at runtime; the loop only enforces budgets.")
    L.append("")
    for step in result.trace:
        mark = "ok" if step.ok else "FAILED"
        label, _, _ = _TOOL_STYLE.get(step.tool.value, ("step", "", ""))
        L.append(f"### {step.n}. {label} — `{_call_signature(step)}` ({mark}, {step.latency_ms}ms)")
        L.append("")
        if step.thought:
            L.append(f"*{_md_escape(step.thought)}*")
            L.append("")
        obs = (step.observation or step.error or "").strip()
        if obs:
            for line in obs.splitlines()[:12]:
                L.append(f"> {line.strip()}")
            L.append("")
        if step.sources:
            L.append("Sources: " + ", ".join(f"<{u}>" for u in step.sources[:5]))
            L.append("")

    if rec.unknowns:
        L.append("## Still unknown")
        L.append("")
        for u in rec.unknowns:
            L.append(f"- {_md_escape(u)}")
        L.append("")

    if rec.sources:
        L.append("## Sources")
        L.append("")
        for u in rec.sources:
            L.append(f"- <{u}>")
        L.append("")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def replay(path: str | Path, console: Console, delay: float = 0.0) -> EnrichmentResult:
    """Re-render a saved run step by step. No network, no keys, fully deterministic."""
    result = load_result(path)
    render_header(result.record.input, console)
    for step in result.trace:
        render_step(step, console)
        if delay > 0:
            time.sleep(delay)
    render_result(result, console)
    return result


def summary_table(results: Iterable[EnrichmentResult]) -> Table:
    """Batch summary, best fit first."""
    rows = sorted(
        results,
        key=lambda r: (r.record.icp_fit.score if r.record.icp_fit else -1.0),
        reverse=True,
    )
    t = Table(box=ROUNDED, header_style="bold white", title="batch results (best fit first)")
    t.add_column("company", style="white", overflow="fold")
    t.add_column("domain", style="cyan", overflow="fold")
    t.add_column("tier", justify="center")
    t.add_column("score", justify="right")
    t.add_column("signals", justify="right", style="dim")
    t.add_column("steps", justify="right", style="dim")
    for r in rows:
        rec = r.record
        fit = rec.icp_fit
        tier = fit.tier if fit else "-"
        color = _TIER_COLOR.get(tier, "dim")
        t.add_row(
            rec.company_name or rec.input,
            rec.domain or "-",
            Text(tier, style=f"bold {color}"),
            f"{fit.score:.0f}" if fit else "-",
            str(len(rec.signals)),
            str(r.stats.steps),
        )
    return t


__all__ = [
    "render_step",
    "render_result",
    "render_header",
    "save_json",
    "save_trace_markdown",
    "load_result",
    "replay",
    "summary_table",
]
