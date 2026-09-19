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
    offline: bool = typer.Option(False, "--offline", help="Serve all HTTP from examples/fixtures."),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: anthropic | stub."),
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
    offline: bool = typer.Option(False, "--offline", help="Serve all HTTP from examples/fixtures."),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: anthropic | stub."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Override the step budget."),
) -> None:
    """Research MANY companies from a CSV and write a scored results table."""
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    from lea import trace as trace_mod
    from lea.agent.loop import run_agent

    settings = build_settings(out=out, offline=offline, provider=provider, max_steps=max_steps)

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
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: anthropic | stub."),
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


@app.command()
def doctor() -> None:
    """Show what is configured, what will happen with the keys you have, and what is missing."""
    loaded = load_dotenv()
    settings = Settings.from_env()

    console.print(Panel(Text("lead-enrichment-agent — environment check", style="bold white"),
                        border_style="blue"))

    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column("k", style="dim", no_wrap=True)
    t.add_column("v", overflow="fold")
    t.add_row(".env", f"{PROJECT_ROOT / '.env'} ({loaded} vars loaded)" if loaded else "not found (optional)")
    t.add_row("llm provider", f"{settings.resolved_llm()}  [dim](configured: {settings.llm_provider})[/dim]")
    t.add_row("model", settings.model)
    t.add_row("search provider", f"{settings.resolved_search()}  [dim](configured: {settings.search_provider})[/dim]")
    t.add_row("offline mode", "ON — all HTTP served from fixtures" if settings.offline else "off")
    t.add_row("out dir", str(settings.out_dir))
    console.print(t)
    console.print()

    keys = Table(show_header=True, header_style="bold dim", title="API keys", title_justify="left")
    keys.add_column("key", no_wrap=True)
    keys.add_column("status")
    keys.add_column("effect", style="dim", overflow="fold")
    keys.add_row("ANTHROPIC_API_KEY", _mark(bool(settings.anthropic_api_key)),
                 "real reasoning; without it the deterministic stub brain is used")
    keys.add_row("TAVILY_API_KEY", _mark(bool(settings.tavily_api_key)), "best search quality (first choice)")
    keys.add_row("SERPER_API_KEY", _mark(bool(settings.serper_api_key)), "Google results (second choice)")
    keys.add_row("BRAVE_API_KEY", _mark(bool(settings.brave_api_key)), "Brave results (third choice)")
    console.print(keys)
    console.print("[dim]values are never printed, only presence.[/dim]")
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

    console.print()
    verdict = Text()
    if settings.anthropic_api_key:
        verdict.append("Ready. ", style="bold green")
        verdict.append(
            f"The agent will reason with {settings.model} and search via "
            f"{settings.resolved_search()}.",
        )
    else:
        verdict.append("Ready, keyless. ", style="bold yellow")
        verdict.append(
            "With no ANTHROPIC_API_KEY the run uses the deterministic stub brain: it still "
            "searches, fetches and produces a full record and trace, but the step-by-step "
            "reasoning is heuristic rather than model-driven. Set ANTHROPIC_API_KEY to turn "
            "it into a real reasoning agent.",
        )
    if settings.offline:
        verdict.append("\nOffline mode is ON, so no network calls will be made at all.", style="dim")
    console.print(Panel(verdict, border_style="blue"))
    console.print("\n[dim]try:[/dim] lea enrich ramp.com")


def main() -> None:  # pragma: no cover - console-script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
