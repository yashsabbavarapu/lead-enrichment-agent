"""The agent's prompt library.

Everything the LLM is ever told lives here, so the behaviour of the agent can be
read and tuned in one file. Nothing here imports from lea.agent / lea.enrich /
lea.cli — prompts are a leaf.

Renderers are defensive: they accept pydantic models, dicts, or None for the
state they render, because they are called from a loop that may be mid-failure.
"""

from __future__ import annotations

import re
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

TWO TOOLS, TWO DIFFERENT JOBS — this is the thing agents get wrong
- `web_search` LOCATES documents. It gives you titles, URLs and ~150-character snippets. \
A snippet is an index card: it tells you which page is worth opening. It is NOT evidence, \
and it is not where facts live.
- `fetch_page` READS a document. This is where facts live. Nearly everything you are asked \
for is page-shaped, not snippet-shaped:
    headquarters city · founding year · employee count · what the company actually sells ·
    funding amount and round name · number of open roles · the date of an announcement
- Searching again for a page-shaped fact does not work. It returns the same kind of snippet \
you already have. If you catch yourself thinking "I still need the HQ / founding year / \
headcount / the funding details", that is a FETCH instruction, not a search instruction — \
open the company's /about or /company page, or open the article you already found.
- A run that ends with more searches than page reads has almost certainly failed: it \
collected links and never opened them.

WRITING A SEARCH QUERY — short, and anchored on the brand
YOUR FIRST SEARCH IS THE BARE COMPANY NAME. Nothing else. Not the name plus "software", \
not plus "company", not plus "app", not plus what they do. You do not know the domain \
yet, so the name is your only anchor, and every extra word weakens it. Search "Linear". \
Search "Notion". Search "Datadog". Then read the domain out of the results and anchor \
everything afterwards on THAT.

Search engines collapse on long, keyword-stuffed queries. Adding descriptive words to \
"help" does the opposite: it dilutes the brand anchor, and you get pages about the WORDS \
instead of pages about the COMPANY. Measured against this project's own search provider:

  "Datadog"                                    -> datadog.com + knowledge panel      GOOD
  "vanta funding round"                        -> Vanta's release, BusinessWire, CNBC GOOD
  "ramp corporate cards company"               -> gift cards, a card validator, Zazzle JUNK
  "ramp careers hiring"                        -> a pesto recipe, a drowning report    JUNK
  "vanta careers hiring"                       -> Brunswick, Coca-Cola, GAO careers    JUNK
  "Vanta SOC 2 compliance automation software" -> generic SOC 2 pages, no Vanta        JUNK
  "Linear issue tracker software company"      -> J.K. Rowling, Jira docs, a GitHub bug JUNK

- Two to four words. If a query runs longer than four, delete the adjectives first.
- Anchor on the NAME or the DOMAIN and almost nothing else: "<name> funding", not \
"<name> latest funding round news 2025".
- Never add category words — software, company, platform, tool, app, startup, solution. \
Those are exactly what the engine matches when the brand anchor is weak.

WHEN THE COMPANY NAME IS AN ORDINARY WORD — this is where runs go wrong
Linear, Ramp, Notion, Figma, Arc, Lattice, Stripe and many others are everyday words, and \
the engine will cheerfully return the word rather than the company.
- Anchor on the DOMAIN instead: "linear.app", "ramp.com".
- For anything on the company's own site, use a site: query. "site:linear.app about" and \
"site:ramp.com careers" beat any descriptive phrasing, and are the most reliable way to \
reach firmographics.
- Never pair an ordinary-word name with generic category words. "Linear issue tracker \
software company" is how a research run ends up citing a novelist's blog.

READING A RESULT SET BEFORE YOU OPEN ANYTHING
- Look at the DOMAINS first. If none of them relate to the target — not the company's own \
site, not a news outlet covering it, not a directory profile of it — then the query \
failed. Re-query with a tighter anchor: the bare domain, or site:<domain>.
- Do NOT open a junk result hoping it turns out relevant. It will not, and its URL ends \
up cited in a record a salesperson puts in front of a customer.
- Results that are clearly about a different entity sharing the name are the same \
failure. Re-anchor on the domain rather than reading them.

DEFAULT RHYTHM (a strong default, not a script)
- After a search returns plausible URLs, your next step should normally be to OPEN one of them.
- Two searches in a row are justified only when (a) the previous search returned nothing \
usable, or (b) you are starting a genuinely new line of enquiry — e.g. firmographics are \
done and you are now hunting funding or hiring news.
- You may override this default. If you do, say why in your `thought`.

WHERE EACH KIND OF FACT LIVES
- Firmographics — what they sell, industry, HQ, founding year, size: the company's OWN site \
(homepage, /about, /company, /contact, the footer). Read the page. Directory profiles \
(Crunchbase, LinkedIn) are a usable fallback, but still only when you OPEN them.
- Signals — funding, launches, leadership changes, partnerships: news articles, press \
releases, the company blog. Open the article; a headline in a snippet gives you the event \
but not the amount, the round, the investors or the date.
- Hiring: the careers/jobs page. Open roles and their departments say more about where a \
company is investing than any press release.
- Technology: run `detect_tech` on the homepage once you have fetched it.

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

THE RULE THAT OUTRANKS EVERY OTHER RULE HERE
Extract everything the evidence supports, and nothing it does not.
The two ways of getting this wrong are NOT equally bad:
- A field left null that a block actually stated: a small loss. A human re-reads a page.
- A field filled with something no block stated: a real harm. A sales rep opens a \
conversation with a "fact" that is false, in front of the customer.
So when the evidence is thin, leave it null. A gap is useful, and honest gaps are a \
documented feature of this product. A confident wrong value is the one thing we will \
not ship.

You have no knowledge of these companies beyond the blocks below. If you find yourself \
RECALLING a company rather than READING it, stop — that is memory, not evidence, and it \
does not belong in the record.

READ THE WHOLE BLOCK BEFORE DECLARING SOMETHING UNKNOWN
Evidence blocks are raw page text, not summaries. The fact you need is often mid-\
paragraph, in a footer address line, in an "our story" aside or in a headline, and \
phrased differently from the field name. Scan the whole block before answering null.
Each example below applies ONLY when that text actually appears in a block:
- "Headquartered in San Francisco, California"  -> hq_city "San Francisco", hq_country "United States"
- "228 Park Ave S, New York, NY 10003"          -> hq_city "New York", hq_country "United States"
- "Founded in 2018" / "since 2018" / "est. 2018" -> founded_year 2018
- "our team of 1,200 people" / "1,200 employees" -> employee_estimate 1200
- "Vanta raises $150M Series C led by Sequoia"   -> a funding Signal: amount $150M, round Series C
- "We have 140 open roles"                       -> a hiring Signal
- "compliance automation platform for SOC 2 and ISO 27001" -> industry from what they SELL

INDUSTRY MUST BE SPECIFIC
`industry` names the category the company SELLS INTO, in the words its own site uses.
"Software", "Technology", "SaaS", "B2B" and "Software & Technology" are umbrellas, not
industries — they describe almost every company in this pipeline and are useless to a
sales team segmenting accounts. Name the actual category:
  Security & Compliance · Fintech · Developer Tools · Data & Analytics · HR & Payroll ·
  Martech · Healthtech · Logistics · E-commerce · Insurtech · Legaltech
Use `sub_industry` for the sharper specialisation ("Compliance Automation", "Spend
Management"). If the evidence only supports an umbrella term, prefer leaving `industry`
null over recording one.
A US state or a postal code implies the country. A headline that names an event IS a \
signal, even if the article body is not in the evidence — record it with whatever detail \
the text does carry and cite the block.

THE QUOTE TEST — apply it to every value before you write it
Ask: can I point to the words in a block that say this? If you cannot quote them, the \
value is null. Not "probably", not "it is well known that" — quotable, or null.

HEADCOUNT FAILS THE QUOTE TEST ALMOST EVERY TIME
`employee_estimate` and `employee_range` are the values you are most likely to "know" \
from memory rather than read from a page. Almost no company page states its headcount, \
so for most companies the correct answer is null — and null is the answer we want.
- Write a headcount ONLY when a block states it and you could quote the sentence.
- An employee band is valid only if that band appears verbatim in a block. Do NOT \
produce a band because it looks right for a company of this size, funding stage, \
customer count or reputation. That is recall, and recall is not evidence here.
- Open roles are NOT a headcount. "93 open roles" is a hiring signal and says nothing \
about how many people work there. Do not scale, multiply or infer from it.
- Funding amount, valuation, customer count and office count are not headcount either.
- A number only counts as a headcount if the words beside it say so — "500 employees", \
"a team of 500", "we are 500 people". A bare "500" somewhere on a page is NOT a \
headcount: it may be a price, a customer count, a job id, a funding figure or a street \
number. Do not adopt a loose number because it is the only one available.
- `employee_range` takes a BAND quoted from a page ("X-Y", "X+"), never a single bare \
number. If you have one explicit headcount figure, put it in `employee_estimate` and \
leave `employee_range` null; do not invent a band around it.
- If no block states it: `employee_estimate` null AND `employee_range` null. Leaving \
both null is the correct, expected outcome for most companies.

`unknowns` IS PART OF A CORRECT ANSWER, NOT AN AFTERTHOUGHT
Name every field you could not establish. A short `unknowns` list against a thin \
evidence set is itself an error — it means you filled something you should not have. \
Returning an empty `unknowns` is a claim that the evidence established everything, so \
make it only when that is true.

RULES
- Fill a field when an evidence block supports it. Every non-obvious field must be \
traceable to a specific block — if asked for reasoning or evidence, cite the source URL \
of the block you used.
- If the evidence genuinely does not establish a field, leave it null / empty. Do not \
guess, do not round a "feels about right" number, do not infer a headquarters from a \
top-level domain. "Not stated anywhere in the blocks" is unknown; "stated in different \
words than the field name" is not.
- Where the schema has an `unknowns` list, name every field you could not establish.
- Where the schema has a `sources` list, include the URLs of the blocks you actually used.
- Prefer the company's own pages over third-party summaries when they disagree, and say \
which you used.
- Dates: copy them as published rather than normalising them into something the source \
did not say. Undated items should be marked undated.
- Numbers: copy the figure and its units exactly, as the block writes them. A band \
quoted from a page is an employee range; a single figure is an employee count. Never \
convert one into the other, and never write a number no block contains.
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

THE RECORD IS A REPORT, NOT A GUARANTEE
The record you are scoring was itself extracted from a handful of pages, and it is \
expected to have gaps. Read it the way you would read a colleague's notes.
- A null/empty field means NOBODY ESTABLISHED IT. It does not mean "average". Score that \
criterion as unevidenced and let `confidence` carry the doubt.
- Never fill a gap yourself. If the record has no headcount, you do not know the \
headcount — not from the company's name, its funding, its customer count or your own \
memory of it.
- Do not write that something "is stated", "is reported" or "is listed" unless the record \
carries that value WITH a source. Write "the record gives X (source: …)" for values that \
have one, and "the record does not establish X" for the rest. Claiming a source that is \
not there is worse than scoring low.
- Check the record's own `unknowns` list before reasoning about a field; if a field is \
named there, treat it as unevidenced no matter what else you believe.

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

#: Where each unknown actually lives. An unknown that is page-shaped should read
#: as a fetch instruction, not as an invitation to search again — the observed
#: failure was a model thinking "I need HQ, founded year, headcount" and issuing
#: another search, none of which can carry those facts.
_GAP_GUIDANCE = {
    "company_name": "the homepage <title> or /about — confirm it on the page",
    "domain": "the search results themselves carry this",
    "description": "the homepage hero text or /about — read the page, not the snippet",
    "industry": "how the company describes what it SELLS on its own homepage/about page",
    "sub_industry": "the product or solutions pages",
    "employee_range": "an /about, /company or careers page, or an opened directory profile — never a snippet",
    "employee_estimate": "an /about, /company or careers page, or an opened directory profile — never a snippet",
    "hq_city": "/about, /contact, or the page footer — a search snippet almost never carries this",
    "hq_country": "/about, /contact, or the page footer — a search snippet almost never carries this",
    "founded_year": "the 'our story' text on /about or /company, or an opened directory profile — you must open the page",
    "signals": "open a dated news article, press release or blog post — a headline alone gives you no amount, round or date",
    "tech_stack": "run detect_tech on the fetched homepage",
    "hooks": "derived later from the signals you gather — no step needed",
    "icp_fit": "scored later from the record — no step needed",
}

#: Gaps nothing can research directly; never nag the model about these.
_DERIVED_GAPS = frozenset({"hooks", "icp_fit", "unknowns", "sources", "input"})

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


_NAME_NOISE = frozenset({"inc", "llc", "ltd", "corp", "the", "com", "www", "app", "io", "co", "gmbh", "plc"})


def _target_tokens(target: str) -> list[str]:
    """Lowercase tokens of the target, for judging whether a domain relates to it."""
    base = (target or "").strip().lower()
    base = base.split("//")[-1].split("/")[0]
    return [t for t in re.split(r"[^a-z0-9]+", base) if len(t) > 2 and t not in _NAME_NOISE]


def _domain_of(url: str) -> str:
    """Bare host, no scheme and no www. — the part worth showing the model."""
    text = str(url or "")
    host = text.split("//")[-1].split("/")[0].split("?")[0].lower()
    return host[4:] if host.startswith("www.") else host


def _url_relates(url: str, tokens: Sequence[str]) -> bool:
    """Host OR path may carry the company: crunchbase.com/organization/vanta counts."""
    if _domain_relates(_domain_of(url), tokens):
        return True
    path = str(url or "").split("//")[-1]
    path = path[path.find("/") + 1 :].lower() if "/" in path else ""
    return any(tok in path for tok in tokens)


def _domain_relates(domain: str, tokens: Sequence[str]) -> bool:
    """Does this host plausibly belong to, or cover, the target company?"""
    if not domain:
        return False
    bare = domain.split(".")[0]
    if any(tok in bare or bare in tok for tok in tokens):
        return True
    # Directories and news outlets are about the company even though the host
    # is not; the path usually carries the name, which the caller checks.
    return domain in _COVERAGE_DOMAINS


#: Hosts that legitimately carry coverage of a company without bearing its name.
_COVERAGE_DOMAINS = frozenset(
    {
        "crunchbase.com", "linkedin.com", "techcrunch.com", "businesswire.com",
        "prnewswire.com", "reuters.com", "bloomberg.com", "cnbc.com", "forbes.com",
        "wikipedia.org", "pitchbook.com", "theinformation.com", "axios.com",
        "sec.gov", "glassdoor.com", "g2.com", "producthunt.com", "ycombinator.com",
    }
)


def _tool_usage(steps: Sequence[Any]) -> dict[str, int]:
    """How many times each tool has actually run."""
    counts = {"web_search": 0, "fetch_page": 0, "detect_tech": 0, "finish": 0}
    for raw in steps:
        s = _as_dict(raw)
        tool = s.get("tool")
        name = str(getattr(tool, "value", tool) or "")
        if name in counts:
            counts[name] += 1
    return counts


def _unread_urls(steps: Sequence[Any]) -> list[str]:
    """Links the agent has been handed but never opened, in discovery order."""
    seen: list[str] = []
    opened: set[str] = set()
    for raw in steps:
        s = _as_dict(raw)
        tool = str(getattr(s.get("tool"), "value", s.get("tool")) or "")
        args = _as_dict(s.get("args"))
        if tool in ("fetch_page", "detect_tech"):
            for value in args.values():
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    opened.add(value.rstrip("/"))
        for url in s.get("sources") or []:
            url = str(url)
            if url.startswith(("http://", "https://")) and url not in seen:
                seen.append(url)
    return [u for u in seen if u.rstrip("/") not in opened]


def _last_tool(steps: Sequence[Any]) -> str:
    if not steps:
        return ""
    tool = _as_dict(steps[-1]).get("tool")
    return str(getattr(tool, "value", tool) or "")


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
    tgt_tokens = _target_tokens(target)

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
    lines.append("STILL UNKNOWN — and where each one actually lives:")
    gaps = [g for g in (known_gaps or []) if g]
    gap_list = [g for g in dict.fromkeys([*gaps, *missing]) if g not in _DERIVED_GAPS]
    if gap_list:
        for gap in gap_list:
            hint = _GAP_GUIDANCE.get(gap)
            lines.append(f"  - {gap}" + (f" — {hint}" if hint else ""))
    else:
        lines.append("  (nothing outstanding)")

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
            srcs = [str(u) for u in (s.get("sources") or [])]
            if srcs and tool == "web_search":
                # Domains, not full URLs: a mismatch between the query and what
                # came back is only obvious when the hosts are readable at a glance.
                doms = list(dict.fromkeys(_domain_of(u) for u in srcs if u))
                shown = ", ".join(doms[:6]) + ("…" if len(doms) > 6 else "")
                related = [d for d in doms if _domain_relates(d, tgt_tokens)]
                if doms and not related:
                    lines.append(
                        f"     results from: {shown}"
                        f"  <- NONE of these relate to {target}; this query lost the brand anchor"
                    )
                else:
                    lines.append(f"     results from: {shown}")
            elif srcs:
                lines.append(f"     sources: {', '.join(srcs[:4])}")
    else:
        lines.append("STEPS ALREADY TAKEN (0): this is your first move.")

    # --- how the effort has actually been spent --------------------------
    used = _tool_usage(steps)
    n_searched, n_read = used["web_search"], used["fetch_page"]
    unread = _unread_urls(steps)

    lines.append("")
    lines.append("EVIDENCE POSITION:")
    steps_left = budget.get("steps", budget.get("max_steps", "?"))
    searches_left = budget.get("searches", budget.get("max_searches", "?"))
    fetches_left = budget.get("fetches", budget.get("max_fetches", "?"))
    lines.append(f"  searches: {n_searched} used / {searches_left} left")
    lines.append(f"  pages read: {n_read} / {fetches_left} left")
    lines.append(f"  steps={steps_left}  searches={searches_left}  fetches={fetches_left}")
    for key, value in budget.items():
        if key not in ("steps", "searches", "fetches", "max_steps", "max_searches", "max_fetches"):
            lines.append(f"  {key}={value}")

    # The single most useful thing to put in front of the model: the links it
    # has collected and never opened.
    related = [u for u in unread if _url_relates(u, tgt_tokens)]
    unrelated = [u for u in unread if u not in set(related)]

    lines.append("")
    if related:
        lines.append(f"UNREAD LINKS THAT LOOK RELATED TO {target} ({len(related)}) — your cheapest source of facts:")
        lines.extend(f"  - {url}" for url in related[:12])
        if len(related) > 12:
            lines.append(f"  … and {len(related) - 12} more")
    elif unread:
        lines.append(f"UNREAD LINKS RELATED TO {target} (0) — nothing you have found looks like this company.")
    else:
        lines.append("URLS FOUND BUT NOT YET READ (0): every link you have found has been opened.")

    if unrelated:
        lines.append("")
        lines.append(
            f"UNREAD LINKS THAT DO NOT LOOK RELATED ({len(unrelated)}) — do NOT open these; "
            "citing one puts an unrelated page in a customer-facing record:"
        )
        lines.extend(f"  - {url}" for url in unrelated[:6])
        if len(unrelated) > 6:
            lines.append(f"  … and {len(unrelated) - 6} more")

    # --- the nudge -------------------------------------------------------
    lines.append("")
    if isinstance(steps_left, int) and steps_left <= 1:
        lines.append("BUDGET IS NEARLY SPENT — call finish unless one step clearly matters more.")
    elif unread and not related:
        # Every link on the table is junk. Opening one is the failure mode that
        # put an unrelated site into a published record; re-anchor instead.
        lines.append(
            f"NOTE: none of the {len(unread)} link(s) you have found look like {target}. Your "
            "queries are matching the words in the name rather than the company. Do NOT open "
            "any of them. Re-query with a tighter anchor: the bare domain, or a site: query "
            "against the company's own domain."
        )
    elif related and n_searched > n_read:
        lines.append(
            f"NOTE: you have run {n_searched} search(es) and opened {n_read} page(s), with "
            f"{len(related)} related link(s) still unread. Searching again will return more "
            "snippets, not more facts. Unless every related link above is useless, open one "
            "now — and if you search instead, say in your thought why."
        )
    elif related and _last_tool(steps) == "web_search":
        lines.append(
            "NOTE: your last step was a search and it returned related links you have not "
            "opened. Reading one is normally the right next move."
        )

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
