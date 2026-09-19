"""`lea` — command line entry point.

Every command is a thin shell around `lea.agent.loop` plus `lea.trace`: parse flags,
build Settings, run, render. Imports of sibling packages happen inside the command
bodies so that a broken or half-finished module only breaks the command that needs
it, and `lea doctor` always runs.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lea.config import PROJECT_ROOT, Settings

HELP = """\
An autonomous research agent for B2B leads.

Give it a company name or domain and it decides its own next research step --
search, fetch, detect tech -- until it can fill in a structured record, score the
company against your ICP and draft grounded outreach hooks. Every step it takes is
printed as it happens, saved to JSON, and replayable offline.
"""

app = typer.Typer(
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
icp_app = typer.Typer(help="Build or inspect an ICP profile.", no_args_is_help=True)
app.add_typer(icp_app, name="icp")

console = Console()
err_console = Console(stderr=True)

# 1 is a usage/IO error. 2 means the command ran but established nothing, so a
# caller can tell "the agent failed to research" from "the agent found a bad fit".
EXIT_UNGROUNDED = 2

# Backstop for `doctor --models`: list_models has its own HTTP timeout, but a DNS
# black hole can outlast it and doctor must never hang.
MODELS_TIMEOUT_S = 12.0

# Same backstop for the Scrappa credit lookup. Shorter: it is a small call on a
# bare `doctor`, so it must never be the reason the command feels slow.
BALANCE_TIMEOUT_S = 4.0


# --------------------------------------------------------------------------
# .env + settings
# --------------------------------------------------------------------------


def load_dotenv(path: Optional[Path] = None) -> int:
    """Minimal KEY=VALUE parser. Not worth a dependency, and never overrides a
    variable the user already exported."""
    p = Path(path) if path else PROJECT_ROOT / ".env"
    if not p.is_file():
        return 0
    loaded = 0
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded += 1
    except OSError:
        return 0
    return loaded


def build_settings(
    *,
    out: Optional[Path] = None,
    max_steps: Optional[int] = None,
    offline: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    quiet: bool = False,
) -> Settings:
    load_dotenv()
    s = Settings.from_env()
    if out is not None:
        s.out_dir = Path(out)
    if max_steps is not None and max_steps > 0:
        s.max_steps = max_steps
    if offline:
        s.offline = True
    if provider:
        s.llm_provider = provider
    if model:
        s.model = model
    s.verbose = not quiet
    return s


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:60] or "target"


def warn_if_offline_but_live_llm(settings: Settings) -> None:
    """`--offline` stubs HTTP, not the model. Say so.

    The surprise is expensive rather than cosmetic: a free Gemini tier allows
    ~20 requests/day/model and one run spends ~11, so a user doing "offline"
    iteration can burn a day's quota without ever seeing a network call.
    Semantics are left alone on purpose — recorded pages plus real reasoning is a
    legitimate and much-used combination — so this only names the missing half.
    """
    if not settings.offline:
        return
    provider = settings.resolved_llm()
    if provider == "stub":
        return
    err_console.print(
        f"[yellow]note:[/yellow] --offline serves pages from fixtures, but the LLM "
        f"([bold]{provider}[/bold]) is still live and will use quota. "
        f"Add [cyan]--provider stub[/cyan] for a fully offline run."
    )


def _plausible_name(name: str) -> bool:
    """A filename guard, not a sanitizer. Rejects the obvious page-title shapes
    ("SOC 2, HIPAA, ISO 27001, PCI, and GDPR Compliance") so that a bad extraction
    can never decide what a file is called."""
    n = (name or "").strip()
    if not n or len(n) > 60:
        return False
    if any(ch in n for ch in "|,/"):
        return False
    return len(n.split()) <= 6


def output_slug(record: Any, target: str) -> str:
    """Filename stem for a run.

    When the user typed a plain name, that IS the filename — they should be able to
    predict it without reading the report. Only a domain/URL target defers to the
    resolved company name, and then only if it looks like a name at all.
    """
    from lea.agent.loop import _host, looks_like_domain

    if not looks_like_domain(target):
        return slugify(target)

    name = (getattr(record, "company_name", "") or "").strip()
    if name and _plausible_name(name):
        return slugify(name)
    # Fall back to the brand label, not the whole host: "ramp.com" -> "ramp",
    # "www.acme.co.uk" -> "acme". _host already drops the scheme, path and "www.".
    host = _host(target) or (target or "")
    return slugify(host.split(".")[0] or host)


# --------------------------------------------------------------------------
# enrich
# --------------------------------------------------------------------------


@app.command()
def enrich(
    target: str = typer.Argument(..., help="Company name, domain or URL, e.g. 'ramp.com'."),
    icp: Optional[Path] = typer.Option(None, "--icp", help="Path to an ICP yaml profile."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory (default ./out)."),
    json_out: bool = typer.Option(False, "--json", help="Print only the JSON record to stdout."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Override the step budget."),
    offline: bool = typer.Option(
        False, "--offline",
        help="Serve search and page fetches from examples/fixtures. The LLM still "
             "makes live calls — add --provider stub for a fully offline run.",
    ),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: anthropic | google | groq | openrouter | cerebras | ollama | stub."),
    model: Optional[str] = typer.Option(None, "--model", help="Model id for the LLM provider."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress the live trace."),
) -> None:
    """Research ONE company and print the enriched record."""
    settings = build_settings(
        out=out, max_steps=max_steps, offline=offline,
        provider=provider, model=model, quiet=quiet,
    )
    from lea import trace as trace_mod
    from lea.agent.loop import run_agent

    warn_if_offline_but_live_llm(settings)
    profile = _load_icp(icp, settings) if icp else None

    live = not (quiet or json_out)
    if live:
        trace_mod.render_header(target, console, settings)

    on_step = (lambda step: trace_mod.render_step(step, console)) if live else None
    result = run_agent(target, settings, icp=profile, on_step=on_step)

    out_dir = Path(settings.out_dir)
    slug = output_slug(result.record, target)
    json_path = trace_mod.save_json(result, out_dir / f"{slug}.json")
    md_path = trace_mod.save_trace_markdown(result, out_dir / f"{slug}.trace.md")

    if json_out:
        # stdout stays machine-clean so this is pipeable into jq.
        sys.stdout.write(result.record.model_dump_json(indent=2) + "\n")
        err_console.print(f"[dim]wrote {json_path} and {md_path}[/dim]")
    else:
        if not quiet:
            trace_mod.render_result(result, console)
        console.print(f"[dim]wrote[/dim] {json_path}")
        console.print(f"[dim]wrote[/dim] {md_path}")

    # Output is written first, then we fail loudly: a run that established nothing
    # must not look like success to a script wrapping this command.
    if trace_mod.is_ungrounded(result):
        err_console.print(
            f"[red]no research succeeded for {target} — {result.stats.stop_reason}[/red]"
        )
        raise typer.Exit(code=EXIT_UNGROUNDED)


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------


@app.command()
def batch(
    csv_path: Path = typer.Argument(..., help="CSV of targets (one company per row)."),
    icp: Optional[Path] = typer.Option(None, "--icp", help="Path to an ICP yaml profile."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory (default ./out)."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only process the first N targets."),
    offline: bool = typer.Option(
        False, "--offline",
        help="Serve search and page fetches from examples/fixtures. The LLM still "
             "makes live calls — add --provider stub for a fully offline run.",
    ),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: anthropic | google | groq | openrouter | cerebras | ollama | stub."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Override the step budget."),
) -> None:
    """Research MANY companies from a CSV and write a scored results table."""
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    from lea import trace as trace_mod
    from lea.agent.loop import run_agent

    settings = build_settings(out=out, offline=offline, provider=provider, max_steps=max_steps)
    warn_if_offline_but_live_llm(settings)

    try:
        from lea.io_csv import read_targets

        targets = list(read_targets(str(csv_path)))
    except Exception as exc:
        err_console.print(f"[red]could not read targets from {csv_path}: {exc}[/red]")
        raise typer.Exit(code=1)

    if limit:
        targets = targets[:limit]
    if not targets:
        err_console.print(f"[yellow]no targets found in {csv_path}[/yellow]")
        raise typer.Exit(code=1)

    profile = _load_icp(icp, settings) if icp else None
    out_dir = Path(settings.out_dir)
    results = []

    console.print(f"[dim]enriching {len(targets)} target(s) -> {out_dir}[/dim]\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold white]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("starting", total=len(targets))
        for target in targets:
            progress.update(task, description=target[:40])
            # Per-target isolation: a single blowup must not abort the batch.
            try:
                result = run_agent(target, settings, icp=profile)
            except Exception as exc:  # pragma: no cover - run_agent is already defensive
                err_console.print(f"[red]{target}: {type(exc).__name__}: {exc}[/red]")
                progress.advance(task)
                continue
            results.append(result)
            try:
                slug = output_slug(result.record, target)
                trace_mod.save_json(result, out_dir / f"{slug}.json")
                trace_mod.save_trace_markdown(result, out_dir / f"{slug}.trace.md")
            except Exception as exc:
                err_console.print(f"[yellow]could not write output for {target}: {exc}[/yellow]")
            progress.advance(task)

    csv_out = out_dir / "results.csv"
    try:
        from lea.io_csv import write_csv

        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(results, str(csv_out))
        wrote_csv = True
    except Exception as exc:
        err_console.print(f"[yellow]could not write {csv_out}: {exc}[/yellow]")
        wrote_csv = False

    console.print()
    console.print(trace_mod.summary_table(results))
    if wrote_csv:
        console.print(f"\n[dim]wrote[/dim] {csv_out}")
    console.print(f"[dim]wrote[/dim] {len(results)} JSON record(s) + trace logs to {out_dir}")

    ungrounded = [r for r in results if trace_mod.is_ungrounded(r)]
    if ungrounded:
        err_console.print(
            f"[yellow]{len(ungrounded)}/{len(results)} target(s) established nothing "
            f"(e.g. {ungrounded[0].record.input}: {ungrounded[0].stats.stop_reason})[/yellow]"
        )
    # Only a total washout is a failed batch; partial results are still useful.
    if results and len(ungrounded) == len(results):
        err_console.print("[red]no target produced any research[/red]")
        raise typer.Exit(code=EXIT_UNGROUNDED)


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


@app.command("replay")
def replay_cmd(
    json_path: Path = typer.Argument(..., help="A saved out/<slug>.json result."),
    delay: float = typer.Option(0.4, "--delay", help="Seconds between steps (0 = instant)."),
) -> None:
    """Re-render a saved run, step by step. No network, no keys."""
    from lea import trace as trace_mod

    if not Path(json_path).is_file():
        err_console.print(f"[red]no such file: {json_path}[/red]")
        raise typer.Exit(code=1)
    try:
        trace_mod.replay(json_path, console, delay=delay)
    except Exception as exc:
        err_console.print(f"[red]could not replay {json_path}: {exc}[/red]")
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------
# icp sub-app
# --------------------------------------------------------------------------


def _load_icp(path: Optional[Path], settings: Settings):
    from lea.enrich.icp import load_or_default_icp

    return load_or_default_icp(str(path) if path else None, settings)


@icp_app.command("infer")
def icp_infer(
    examples: str = typer.Option(..., "--examples", help='Comma-separated best customers, e.g. "Ramp,Brex,Deel".'),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write the profile (default icp.yaml)."),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: anthropic | google | groq | openrouter | cerebras | ollama | stub."),
) -> None:
    """Derive an ICP profile from a handful of your best customers."""
    settings = build_settings(provider=provider)
    names = [e.strip() for e in examples.split(",") if e.strip()]
    if not names:
        err_console.print("[red]--examples needs at least one company name[/red]")
        raise typer.Exit(code=1)

    try:
        from lea.enrich.icp import infer_icp, save_icp
        from lea.llm.factory import get_llm

        llm = get_llm(settings)
        profile = infer_icp(names, llm, settings)
    except Exception as exc:
        err_console.print(f"[red]could not infer an ICP: {exc}[/red]")
        raise typer.Exit(code=1)

    target = Path(out) if out else PROJECT_ROOT / "icp.yaml"
    try:
        save_icp(profile, str(target))
    except Exception as exc:
        err_console.print(f"[yellow]inferred the profile but could not save it: {exc}[/yellow]")
        target = None

    _print_icp(profile)
    if target:
        console.print(f"\n[dim]wrote[/dim] {target}")


@icp_app.command("show")
def icp_show(
    icp: Optional[Path] = typer.Option(None, "--icp", help="Path to an ICP yaml profile."),
) -> None:
    """Print the ICP profile that would be used (yours, or the built-in default)."""
    settings = build_settings()
    try:
        profile = _load_icp(icp, settings)
    except Exception as exc:
        err_console.print(f"[red]could not load an ICP profile: {exc}[/red]")
        raise typer.Exit(code=1)
    _print_icp(profile)


def _print_icp(profile) -> None:
    head = Text()
    head.append(getattr(profile, "name", "default"), style="bold white")
    summary = getattr(profile, "summary", "")
    if summary:
        head.append("\n")
        head.append(summary.strip(), style="")
    derived = getattr(profile, "derived_from", None)
    if derived:
        head.append("\n")
        head.append("derived from: " + ", ".join(derived), style="dim")
    console.print(Panel(head, border_style="blue", title="ICP profile", title_align="left"))

    criteria = getattr(profile, "criteria", []) or []
    if criteria:
        t = Table(show_header=True, header_style="bold dim")
        t.add_column("criterion", style="white", no_wrap=True)
        t.add_column("weight", justify="right", style="dim")
        t.add_column("description", overflow="fold")
        t.add_column("rubric", style="dim", overflow="fold")
        for c in criteria:
            t.add_row(c.key, f"{c.weight:g}", c.description or "-", c.rubric or "-")
        console.print(t)

    dq = getattr(profile, "disqualifiers", []) or []
    if dq:
        console.print(Text("disqualifiers: ", style="bold red") + Text(", ".join(dq), style="dim"))


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _mark(ok: bool, yes: str = "present", no: str = "absent") -> Text:
    return Text(yes, style="green") if ok else Text(no, style="yellow")


# One line each, aimed at someone deciding which key to go get. "free" is the word
# that matters most here, so it leads.
_PROVIDER_EFFECT = {
    "anthropic": ("paid", "best quality"),
    "google": ("FREE", "generous limits"),
    "groq": ("FREE", "fastest"),
    "openrouter": ("FREE", "many models"),
    "cerebras": ("FREE", "very fast"),
    "ollama": ("FREE", "runs on your box"),
}


def cached_resolved_model(settings: Settings, provider: str) -> Optional[str]:
    """The self-healed model id, but ONLY if it is already cached.

    `doctor` is the command people run when something is wrong, so it must stay
    instant and work offline. The plain summary therefore never triggers a lookup:
    it reports a resolution that `--models` or an earlier run already paid for, and
    otherwise says nothing. Reaching into the resolution cache is deliberate and
    guarded — if that internal moves, doctor just shows the configured id.
    """
    try:
        from lea.llm import openai_compat as oc

        base_url, configured = settings.endpoint_for(provider)
        if not base_url:
            return None
        return oc._MODEL_RESOLUTION.get((base_url.rstrip("/"), configured))
    except Exception:
        return None


# provider -> (Settings attribute holding its key, env var, one-line note).
# The keyless fallback has no key attribute.
_SEARCH_PROVIDERS: dict[str, tuple[Optional[str], str, str]] = {
    "scrappa": ("scrappa_api_key", "SCRAPPA_API_KEY", "Google SERP · 500 free credits/mo"),
    "tavily": ("tavily_api_key", "TAVILY_API_KEY", "search API built for agents"),
    "serper": ("serper_api_key", "SERPER_API_KEY", "Google results"),
    "brave": ("brave_api_key", "BRAVE_API_KEY", "Brave index"),
    "duckduckgo": (None, "(no key)", "keyless · bot-blocks often"),
}


def search_provider_order() -> list[str]:
    """The real auto-selection order, asked of Settings instead of duplicated here.

    `doctor` is where someone decides which key to go and get, so a hardcoded
    "first choice" is exactly the kind of copy that rots the moment the order
    changes — which is what just happened to Tavily. This probes `resolved_search`
    with every key set, then removes the winner and asks again. Pure function, no
    I/O, and it tracks any future reordering for free.
    """
    probe = Settings(search_provider="auto", offline=False)
    pending = {p: attr for p, (attr, _, _) in _SEARCH_PROVIDERS.items() if attr}
    for attr in pending.values():
        setattr(probe, attr, "probe")

    order: list[str] = []
    while pending:
        winner = probe.resolved_search()
        if winner not in pending:
            break
        order.append(winner)
        setattr(probe, pending.pop(winner), None)

    fallback = probe.resolved_search()  # whatever wins with no keys at all
    if fallback not in order:
        order.append(fallback)
    # Anything config never picks still gets listed, just last.
    order.extend(p for p in _SEARCH_PROVIDERS if p not in order)
    return order


def fetch_scrappa_balance(settings: Settings) -> Optional[dict[str, Any]]:
    """Scrappa's remaining credits, or None.

    Only called when a key exists and we are online, so an offline or unkeyed
    `doctor` stays a purely local command at ~0.2s. The daemon-thread cap is the
    same discipline as `--models`: the helper has its own timeout, this is the
    backstop for a DNS black hole that outlasts it. Failure is silent — an
    unreachable balance endpoint is not worth a warning.
    """
    if settings.offline or not getattr(settings, "scrappa_api_key", None):
        return None

    import threading

    box: dict[str, Any] = {}

    def _lookup() -> None:
        try:
            from lea.tools.search import scrappa_balance

            box["balance"] = scrappa_balance(settings)
        except Exception:
            box["balance"] = None

    worker = threading.Thread(target=_lookup, daemon=True)
    worker.start()
    worker.join(timeout=BALANCE_TIMEOUT_S)
    got = box.get("balance")
    return got if isinstance(got, dict) else None


def _format_balance(bal: dict[str, Any]) -> Text:
    """`475 of 500 left this month` — enough context to plan a batch against."""
    balance = bal.get("balance")
    used_month = bal.get("used_this_month")
    used_24h = bal.get("used_last_24h")

    out = Text("       credits: ", style="dim")
    if isinstance(balance, int) and not isinstance(balance, bool):
        total = balance + used_month if isinstance(used_month, int) else None
        # Below roughly one batch's worth this stops being trivia and becomes the
        # reason a 50-company run is about to stop halfway.
        out.append(str(balance), style="bold red" if balance < 50 else "bold green")
        out.append(f" of {total} left this month" if total else " left", style="dim")
    else:
        out.append("unknown", style="dim")
    if isinstance(used_24h, int) and used_24h:
        out.append(f" · {used_24h} in last 24h", style="dim")
    out.append(" · ~1 per search", style="dim")
    return out


def _print_search_providers(settings: Settings) -> None:
    """Search providers in true priority order, with the active one marked."""
    active = settings.resolved_search()
    console.print(Text("search providers — tried in this order", style="bold white"))

    for rank, name in enumerate(search_provider_order(), start=1):
        attr, env, note = _SEARCH_PROVIDERS.get(name, (None, "(no key)", ""))
        is_active = name == active
        if attr is None:
            status, style = "always", "green"
        elif getattr(settings, attr, None):
            status, style = "present", "green"
        else:
            status, style = "absent", "yellow"

        line = Text()
        line.append("  > " if is_active else "    ", style="bold green")
        line.append(f"{rank}. ", style="dim")
        line.append(f"{name:<12}", style="bold green" if is_active else "white")
        line.append(f"{env:<18}", style="dim")
        line.append(f"{status:<8}", style=style)
        line.append(note, style="dim")
        console.print(line)

        if name == "scrappa" and status == "present":
            bal = fetch_scrappa_balance(settings)
            if bal:
                console.print(_format_balance(bal))

    if settings.offline:
        console.print(Text("  (offline: every search is served from fixtures)", style="dim"))
    elif settings.search_provider != "auto":
        console.print(Text(
            f"  LEA_SEARCH_PROVIDER pins this to '{settings.search_provider}', overriding the order.",
            style="dim",
        ))
    console.print(Text("values are never printed, only presence.", style="dim"))


def _print_llm_providers(settings: Settings) -> None:
    """The model-provider table: who you could use, who you are using, and why."""
    from lea.config import OPENAI_COMPAT_PRESETS

    resolved = settings.resolved_llm()

    # An aligned list, not a rich Table: five columns of this content wrap into
    # confetti at 80 chars, and `doctor` has to be readable in a default terminal.
    console.print(Text("model providers — any ONE of these powers the agent", style="bold white"))

    rows: list[tuple[str, str, str]] = [("anthropic", "Anthropic (Claude)", "ANTHROPIC_API_KEY")]
    for name, preset in OPENAI_COMPAT_PRESETS.items():
        rows.append((name, preset.get("label", name), preset.get("env", "") or "(no key)"))
    rows.append(("stub", "stub (built in)", "(no key)"))

    for name, label, env in rows:
        active = name == resolved
        cost, note = _PROVIDER_EFFECT.get(name, ("FREE", "heuristics only"))
        # Ollama needs no key, so "absent" would be a lie; the stub always works.
        if name == "ollama":
            status, style = "local", "cyan"
        elif name == "stub":
            status, style = "always", "green"
        elif settings.key_for(name):
            status, style = "present", "green"
        else:
            status, style = "absent", "yellow"

        line = Text()
        line.append("  > " if active else "    ", style="bold green")
        line.append(f"{label[:25]:<26}", style="bold green" if active else "white")
        line.append(f"{env:<19}", style="dim")
        line.append(f"{status:<8}", style=style)
        line.append(f"{cost:<5}", style="bold green" if cost == "FREE" else "dim")
        line.append(note, style="dim")
        console.print(line)

    active = Text()
    active.append("active: ", style="dim")
    active.append(resolved, style="bold green")
    if settings.is_openai_compat(resolved):
        base, model = settings.endpoint_for(resolved)
        active.append(f"  base_url={base}  model={model}", style="dim")
        # Free only: a warm cache means --models or a previous run already looked
        # this up. Never pay a network round trip for a bare `doctor`.
        healed = cached_resolved_model(settings, resolved)
        if healed and healed != model:
            active.append("\n         will actually use ", style="dim")
            active.append(healed, style="bold green")
            active.append(f" — the endpoint does not serve {model}", style="dim")
        elif healed:
            active.append("  (confirmed served)", style="green")
        else:
            active.append("\n         model ids rotate on free tiers — ", style="dim")
            active.append("lea doctor --models", style="cyan")
            active.append(" checks what yours serves today.", style="dim")
    elif resolved == "anthropic":
        active.append(f"  model={settings.model}", style="dim")
    else:
        active.append("  no LLM — reasoning is heuristic", style="dim")
    console.print(active)


def _print_models(settings: Settings) -> None:
    """What the configured endpoint actually serves today.

    Every default model id in the presets is an educated guess against a lineup that
    rotates, so this is the difference between "it's broken" and "set LEA_MODEL".
    """
    resolved = settings.resolved_llm()
    console.print()
    console.print(Text(f"models served by {resolved}", style="bold white"))

    if resolved in ("anthropic", "stub"):
        console.print(Text(
            f"  {resolved} does not expose a model list. "
            + ("Model ids are in Anthropic's docs." if resolved == "anthropic"
               else "The stub has no models — it is deterministic heuristics."),
            style="dim",
        ))
        return

    # A hung endpoint must not wedge doctor. list_models has its own HTTP timeout;
    # this is the backstop for a DNS black hole that outlasts it. A daemon thread
    # (not an executor) on purpose: an executor's shutdown joins its workers, so a
    # stuck lookup would block the very exit this is meant to protect.
    import threading

    from lea.llm.factory import list_models

    box: dict[str, Any] = {}

    def _lookup() -> None:
        try:
            box["models"] = list_models(settings) or []
        except Exception as exc:  # diagnostics never raise
            box["error"] = f"{type(exc).__name__}: {exc}"
        # Resolve inside the same guarded window. It shares its cache with the
        # agent, so this call also means the next run pays no lookup of its own.
        try:
            from lea.llm.openai_compat import resolve_model

            box["resolved"] = resolve_model(settings)
        except Exception as exc:
            box.setdefault("error", f"{type(exc).__name__}: {exc}")

    worker = threading.Thread(target=_lookup, daemon=True)
    worker.start()
    worker.join(timeout=MODELS_TIMEOUT_S)

    timed_out = worker.is_alive()
    models: list[str] = box.get("models") or []
    will_use: Optional[str] = box.get("resolved")
    if box.get("error"):
        console.print(Text(f"  lookup failed: {box['error']}", style="yellow"))

    base, current = settings.endpoint_for(resolved)

    if models:
        console.print(Text(f"  {len(models)} available:", style="dim"))
        for m in models:
            line = Text("    ")
            style = "bold green" if m == will_use else "white"
            line.append(m, style=style)
            if m == will_use:
                line.append("   <- will be used", style="dim")
            console.print(line)

    # Report what will actually happen. The agent self-heals a stale id, so the old
    # "runs will fail with a 404" warning would now be a lie in the common case.
    console.print()
    if will_use and will_use == current:
        console.print(Text(f"  {current} — served, will be used.", style="green"))
    elif will_use:
        console.print(Text(
            f"  {current} is not served here. {will_use} will be used automatically.",
            style="bold green",
        ))
        console.print(Text(
            f"  Nothing to do — runs self-heal. Set LEA_MODEL to pin a different one.",
            style="dim",
        ))
    else:
        why = (
            f"the lookup took longer than {MODELS_TIMEOUT_S:g}s"
            if timed_out
            else f"{base or 'the endpoint'} was unreachable, or your key was rejected"
        )
        console.print(Text(f"  no models returned — {why}.", style="yellow"))
        console.print(Text(
            f"  Current model id is '{current}' and no replacement could be found, so runs "
            "will fail. Check the key, then set LEA_MODEL to an id from your provider's "
            "dashboard.",
            style="bold yellow",
        ))


@app.command()
def doctor(
    models: bool = typer.Option(
        False, "--models", help="Also list the model ids the configured provider serves."
    ),
) -> None:
    """Show what is configured, what will happen with the keys you have, and what is missing."""
    loaded = load_dotenv()
    settings = Settings.from_env()

    console.print(Panel(Text("lead-enrichment-agent — environment check", style="bold white"),
                        border_style="blue"))

    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column("k", style="dim", no_wrap=True)
    t.add_column("v", overflow="fold")
    t.add_row(".env", f"{PROJECT_ROOT / '.env'} ({loaded} vars loaded)" if loaded else "not found (optional)")
    _resolved = settings.resolved_llm()
    t.add_row("llm provider", f"{_resolved}  [dim](configured: {settings.llm_provider})[/dim]")
    if _resolved == "stub":
        t.add_row("model", "[dim]n/a — the stub uses no model[/dim]")
    elif settings.is_openai_compat(_resolved):
        t.add_row("model", settings.endpoint_for(_resolved)[1])
    else:
        t.add_row("model", settings.model)
    t.add_row("search provider", f"{settings.resolved_search()}  [dim](configured: {settings.search_provider})[/dim]")
    t.add_row(
        "offline mode",
        "ON — pages from fixtures; the LLM is still live unless provider=stub"
        if settings.offline else "off",
    )
    t.add_row("out dir", str(settings.out_dir))
    console.print(t)
    console.print()

    _print_llm_providers(settings)
    console.print()

    _print_search_providers(settings)
    console.print()

    b = Table(show_header=True, header_style="bold dim", title="budgets per company", title_justify="left")
    b.add_column("limit", no_wrap=True)
    b.add_column("value", justify="right")
    b.add_row("max steps", str(settings.max_steps))
    b.add_row("max searches", str(settings.max_searches))
    b.add_row("max page fetches", str(settings.max_fetches))
    b.add_row("step timeout", f"{settings.step_timeout_s:g}s")
    b.add_row("total timeout", f"{settings.total_timeout_s:g}s")
    console.print(b)
    console.print()

    fixture_dir = Path(settings.fixture_dir)
    fixtures = sorted(fixture_dir.glob("**/*")) if fixture_dir.is_dir() else []
    files = [f for f in fixtures if f.is_file()]
    f_line = Text()
    f_line.append("fixtures: ", style="dim")
    if files:
        f_line.append(f"{len(files)} file(s) in {fixture_dir}", style="green")
    else:
        f_line.append(f"none in {fixture_dir} — `--offline` has nothing to serve yet", style="yellow")
    console.print(f_line)

    missing: list[str] = []
    for mod, what in (
        ("lea.tools.registry", "tools"),
        ("lea.llm.factory", "llm clients"),
        ("lea.prompts", "prompts"),
        ("lea.enrich.extract", "extraction"),
        ("lea.enrich.icp", "ICP scoring"),
        ("lea.enrich.hooks", "hook generation"),
        ("lea.io_csv", "CSV I/O"),
    ):
        try:
            __import__(mod)
        except Exception as exc:
            missing.append(f"{what} ({mod}): {type(exc).__name__}: {exc}")
    if missing:
        console.print()
        console.print(Text("modules not importable:", style="bold red"))
        for m in missing:
            console.print(Text("  " + m, style="red"))

    if models:
        _print_models(settings)

    console.print()
    resolved = settings.resolved_llm()
    verdict = Text()
    if resolved == "stub":
        verdict.append("Ready, keyless. ", style="bold yellow")
        verdict.append(
            "With no model-provider key the run uses the built-in stub brain: it still "
            "searches, fetches, scores and produces a full record and trace, but the "
            "step-by-step reasoning is heuristic rather than model-driven.\n\n",
        )
        verdict.append("To get real reasoning without paying, grab ONE free key ", style="bold white")
        verdict.append("and the agent picks it up automatically:\n", style="bold white")
        verdict.append("  GOOGLE_API_KEY      aistudio.google.com/apikey        (free tier)\n", style="green")
        verdict.append("  GROQ_API_KEY        console.groq.com/keys             (free tier)\n", style="green")
        verdict.append("  OPENROUTER_API_KEY  openrouter.ai/keys                (free models)\n", style="green")
        verdict.append("  CEREBRAS_API_KEY    cloud.cerebras.ai                 (free tier)\n", style="green")
        verdict.append("  or run Ollama locally and use --provider ollama       (no key at all)\n", style="green")
        verdict.append("Put it in .env or export it, then re-run `lea doctor`.", style="dim")
    elif resolved == "anthropic":
        verdict.append("Ready. ", style="bold green")
        verdict.append(
            f"The agent will reason with {settings.model} (Anthropic) and search via "
            f"{settings.resolved_search()}."
        )
    else:
        base, model = settings.endpoint_for(resolved)
        verdict.append("Ready. ", style="bold green")
        verdict.append(
            f"The agent will reason with {model} via {resolved} and search via "
            f"{settings.resolved_search()}.\n"
        )
        verdict.append(
            "Free model lineups rotate, so if runs fail with a 404 run `lea doctor --models` "
            "and set LEA_MODEL to an id listed there.",
            style="dim",
        )
    if settings.offline:
        verdict.append("\nOffline mode is ON, so no network calls will be made at all.", style="dim")
    console.print(Panel(verdict, border_style="blue"))
    console.print("\n[dim]try:[/dim] lea enrich ramp.com --offline")


def main() -> None:  # pragma: no cover - console-script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
