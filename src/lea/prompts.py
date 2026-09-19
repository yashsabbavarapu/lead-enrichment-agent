"""The agent's prompt library.

Everything the LLM is ever told lives here, so the behaviour of the agent can be
read and tuned in one file. Nothing here imports from lea.agent / lea.enrich /
lea.cli — prompts are a leaf.

Renderers are defensive: they accept pydantic models, dicts, or None for the
state they render, because they are called from a loop that may be mid-failure.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "AGENT_SYSTEM",
    "EXTRACTION_SYSTEM",
    "ICP_INFERENCE_SYSTEM",
    "ICP_SCORING_SYSTEM",
    "HOOKS_SYSTEM",
    "build_agent_context",
    "build_extraction_context",
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM = """\
You are a GTM (go-to-market) research agent. You are handed one company and you \
research it on the open web, one step at a time, until you can hand a sales team a \
record they can act on.

HOW YOU WORK
- You choose your own next step. Nobody hands you a plan.
- Every step must close a SPECIFIC information gap that you can name. If you cannot \
say which unknown a step resolves, it is the wrong step.
- You have a hard budget of steps, searches and page fetches. Spending it is the only \
irreversible thing you can do here, so spend it like it is yours.
- Never repeat a search you already ran or fetch a URL you already fetched. The trace \
below shows you exactly what you have already done and what it gave you.

WHERE FACTS LIVE (use the cheapest source that can answer the question)
- Firmographics — what the company does, industry, HQ, founding year, size, positioning: \
the company's OWN site (homepage, /about, /company). Treat directory aggregators as a \
hint, never as the source of record.
- Signals — funding, launches, leadership changes, partnerships: news, press releases, \
the company blog. Prefer dated items; an undated claim is weak evidence.
- Hiring signals: the careers/jobs page. Open roles and their departments say more about \
where a company is investing than any press release.
- Technology: the rendered homepage and its scripts, via the tech-detection tool.

EVIDENCE DISCIPLINE
- Never invent a fact. If the web did not say it, it is unknown, and "unknown" is a \
perfectly good answer that a sales rep can work with. A confident wrong number is worse \
than a blank field.
- Every claim you will later make must be traceable to a page you actually fetched.
- Distinguish the company you were asked about from similarly-named companies. If the \
evidence is about a different entity, say so and keep looking.

STOPPING
- Call `finish` as soon as you have enough to write a useful record: who they are, \
roughly how big, what they sell, and at least one recent, dated signal.
- Do not burn remaining budget on marginal detail. Finishing early with an honest set of \
unknowns is a good outcome. Running out of budget mid-thought is not.

OUTPUT
Each turn you emit exactly one action: a short `thought` naming the gap you are closing, \
the `tool` to call, and its `args`.\
"""


EXTRACTION_SYSTEM = """\
You are an information extraction engine for a GTM research pipeline. You are given \
numbered evidence blocks, each with its source URL, and a schema to fill.

RULES
- Fill a field ONLY when an evidence block supports it. Every non-obvious field must be \
traceable to a specific block — if asked for reasoning or evidence, cite the source URL \
of the block you used.
- If the evidence does not establish a field, leave it null / empty. Do not guess, do not \
round a "feels about right" number, do not infer a headquarters from a top-level domain.
- Where the schema has an `unknowns` list, name the fields you could not establish.
- Where the schema has a `sources` list, include the URLs of the blocks you actually used.
- Prefer the company's own pages over third-party summaries when they disagree, and say \
which you used.
- Dates: copy them as published rather than normalising them into something the source \
did not say. Undated items should be marked undated.
- Numbers: copy the figure and its units exactly. "50-200 employees on LinkedIn" is an \
employee range, not an employee count.
- Do not repeat the same fact as several different signals.

Return only the structured object. No commentary.\
"""


ICP_INFERENCE_SYSTEM = """\
You derive an Ideal Customer Profile (ICP) rubric from a handful of a company's existing \
example customers. You are reverse-engineering what these accounts have in common that a \
new prospect could be measured against.

METHOD
1. Look across the examples for shared, OBSERVABLE properties: industry / sub-industry, \
company size band, business model (B2B SaaS, marketplace, services...), geography, \
technical maturity, and the trigger that plausibly made them buy.
2. Turn each shared property into one criterion with: a stable `key` (snake_case), a \
one-line `description`, a `weight`, and a `rubric` written in plain language that says \
what a HIGH score looks like and what a LOW score looks like.
3. Weight by discriminating power, not by how often you can observe it. A criterion that \
every company in the market would pass is worth little, however easy it is to check. \
Weights should be on a comparable scale and should not all be equal.
4. Name `disqualifiers` — properties that should rule an account out regardless of the \
rest (wrong region, competitor, too small to have the problem, regulated in a way that \
blocks purchase).
5. Keep the rubric to 4-7 criteria. A rubric nobody can apply consistently is worse than \
a short one.

HONESTY
- With only a few examples, some "patterns" are coincidence. Only promote a property to a \
criterion if you can say why it would plausibly drive fit, not merely that it recurs.
- Say in the summary what this rubric is NOT able to see.\
"""


ICP_SCORING_SYSTEM = """\
You score one company against a given ICP rubric. You are an evaluator, not an advocate.

METHOD
- Score EVERY criterion in the rubric on 0-10, using that criterion's own rubric text as \
the scale.
- For each criterion give: the score, one or two sentences of concrete reasoning that \
names the fact you used, and the source URL(s) that fact came from.
- Missing evidence is not a middling score by default — say so explicitly in the \
reasoning, score it low-to-middling, and let the overall `confidence` carry the doubt. \
Never treat absence of evidence as evidence of fit.
- Check the disqualifiers. If one is hit, list it, and let the score reflect it.
- Set `confidence` on how much evidence you actually had, independent of how good the fit \
looks.

CALIBRATION
- Be willing to score low. Most companies are not a good fit for most products; a rubric \
where everything lands in the 6-8 band is useless to a sales team.
- Reserve 9-10 for criteria the evidence clearly and specifically satisfies.
- Use 0-3 freely when the evidence points the other way.
- The overall summary should read like a qualification note: the one reason to pursue and \
the one reason to hesitate.\
"""


HOOKS_SYSTEM = """\
You write cold outreach opening hooks for a sales rep, grounded in the research record \
you are given.

WHAT TO WRITE
- 2-3 hooks. Each is 1-2 sentences, under about 40 words.
- Each hook must lean on ONE specific, named fact from the record — a funding round, a \
named open role, a product launch, a detected technology, a stated market. Name the fact \
in the `grounded_in` field and carry its `source_url`.
- Give each hook a short `angle` label describing the logic, e.g. \
"Series B -> new GTM headcount".
- The hook should connect the fact to a plausible problem the reader has right now. A \
fact with no "so what" is trivia, not a hook.

BANNED
- Flattery clichés: "I was impressed by...", "hope this finds you well", "I came across \
your company", "big fan of what you're building", "congrats on the raise!" as the whole \
idea.
- Anything that would read identically if you swapped in a different company name. If the \
sentence survives that swap, delete it.
- Facts not in the record. If the record's unknowns are large, write fewer hooks rather \
than inventing material. Two grounded hooks beat three with one fabrication.
- Questions that the reader can answer "no" to and be done.\
"""


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

_SKIP_WHEN_EMPTY = ("", None, [], {})

# Fields worth showing the agent as "known / still unknown".
_INTERESTING_FIELDS = (
    "company_name",
    "domain",
    "description",
    "industry",
    "sub_industry",
    "employee_range",
    "employee_estimate",
    "hq_city",
    "hq_country",
    "founded_year",
)


def _as_dict(obj: Any) -> dict[str, Any]:
    """Accept a pydantic model, a mapping, or None."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except (TypeError, ValueError):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _short(text: Any, limit: int = 220) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _fmt_args(args: Any) -> str:
    d = _as_dict(args)
    if not d:
        return ""
    parts = [f"{k}={_short(v, 90)!r}" for k, v in d.items() if v not in ("", None)]
    return ", ".join(parts)


def build_agent_context(
    target: str,
    record_so_far: Any = None,
    trace: Optional[Sequence[Any]] = None,
    remaining_budget: Optional[Mapping[str, Any]] = None,
    known_gaps: Optional[Iterable[str]] = None,
) -> str:
    """Render the running research state for one `decide` call.

    Compact on purpose: the model re-reads this every step, so it carries what it
    needs to avoid repeating itself and nothing else.
    """
    rec = _as_dict(record_so_far)
    steps = list(trace or [])
    budget = dict(remaining_budget or {})

    lines: list[str] = []
    lines.append(f"TARGET: {target}")

    # --- what we already know -------------------------------------------
    known: list[str] = []
    missing: list[str] = []
    for field in _INTERESTING_FIELDS:
        value = rec.get(field)
        if value in _SKIP_WHEN_EMPTY:
            missing.append(field)
        else:
            known.append(f"  {field}: {_short(value, 160)}")

    n_signals = len(rec.get("signals") or [])
    n_tech = len(rec.get("tech_stack") or [])
    if n_signals:
        known.append(f"  signals: {n_signals} captured")
    else:
        missing.append("signals")
    if n_tech:
        known.append(f"  tech_stack: {n_tech} detected")

    lines.append("")
    lines.append("KNOWN SO FAR:")
    lines.extend(known or ["  (nothing yet)"])

    lines.append("")
    lines.append("STILL UNKNOWN:")
    gaps = [g for g in (known_gaps or []) if g]
    gap_list = list(dict.fromkeys([*gaps, *missing]))
    lines.extend([f"  - {g}" for g in gap_list] or ["  (nothing outstanding)"])

    # --- what has already been tried ------------------------------------
    lines.append("")
    if steps:
        lines.append(f"STEPS ALREADY TAKEN ({len(steps)}) — do not repeat these:")
        for i, raw_step in enumerate(steps, start=1):
            s = _as_dict(raw_step)
            n = s.get("n", i)
            tool = s.get("tool")
            tool = getattr(tool, "value", tool) or "?"
            args = _fmt_args(s.get("args"))
            head = f"  {n}. {tool}({args})"
            if s.get("ok", True) is False or s.get("error"):
                lines.append(f"{head} -> FAILED: {_short(s.get('error') or 'unknown error', 160)}")
            else:
                lines.append(f"{head} -> {_short(s.get('observation'), 300) or '(no output)'}")
            srcs = s.get("sources") or []
            if srcs:
                lines.append(f"     sources: {', '.join(str(u) for u in srcs[:4])}")
    else:
        lines.append("STEPS ALREADY TAKEN (0): this is your first move.")

    # --- budget ----------------------------------------------------------
    lines.append("")
    lines.append("REMAINING BUDGET:")
    steps_left = budget.get("steps", budget.get("max_steps", "?"))
    searches_left = budget.get("searches", budget.get("max_searches", "?"))
    fetches_left = budget.get("fetches", budget.get("max_fetches", "?"))
    lines.append(f"  steps={steps_left}  searches={searches_left}  fetches={fetches_left}")
    for key, value in budget.items():
        if key not in ("steps", "searches", "fetches", "max_steps", "max_searches", "max_fetches"):
            lines.append(f"  {key}={value}")
    if isinstance(steps_left, int) and steps_left <= 1:
        lines.append("  NOTE: budget is nearly spent — call finish unless one step clearly matters more.")

    lines.append("")
    lines.append("Choose the single next action that closes the most valuable gap above.")
    return "\n".join(lines)


def build_extraction_context(
    target: str,
    evidence_blocks: Optional[Sequence[Any]] = None,
) -> str:
    """Render ``[(source_url, text), ...]`` as delimited, citable evidence blocks."""
    lines: list[str] = []
    lines.append(f"TARGET COMPANY: {target}")
    lines.append("")

    blocks = list(evidence_blocks or [])
    if not blocks:
        lines.append("EVIDENCE: (none was gathered — leave every field unknown)")
        return "\n".join(lines)

    lines.append(
        f"EVIDENCE ({len(blocks)} block(s)). Each block is delimited and labelled with the "
        "URL it came from; cite that URL when you use the block."
    )
    lines.append("")
    for i, block in enumerate(blocks, start=1):
        meta: dict[str, Any] = {}
        if isinstance(block, (tuple, list)) and len(block) >= 2:
            url, text = block[0], block[1]
        elif isinstance(block, Mapping):
            url = block.get("source_url") or block.get("url") or "unknown"
            text = block.get("text") or block.get("content") or ""
            # Structured hints a tool already resolved (locations, detected tech,
            # page title, publish date) are worth stating plainly rather than
            # making the reader dig them back out of the raw page text.
            meta = {
                k: v
                for k, v in block.items()
                if k not in ("source_url", "url", "text", "content") and v not in _SKIP_WHEN_EMPTY
            }
        else:
            url, text = "unknown", block
        body = str(text or "").strip() or "(empty)"
        lines.append(f"--- BLOCK {i} | source_url: {url} ---")
        for key, value in meta.items():
            if isinstance(value, Mapping):
                rendered = "; ".join(f"{k}={_short(v, 60)}" for k, v in list(value.items())[:10])
            elif isinstance(value, (list, tuple, set)):
                rendered = "; ".join(_short(v, 80) for v in list(value)[:10])
            else:
                rendered = _short(value, 240)
            if rendered:
                lines.append(f"{key}: {rendered}")
        lines.append(body)
        lines.append(f"--- END BLOCK {i} ---")
        lines.append("")

    lines.append(
        "Fill the schema from these blocks only. Any field the blocks do not establish "
        "stays null/empty and belongs in `unknowns`."
    )
    return "\n".join(lines)
