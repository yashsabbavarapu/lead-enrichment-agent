"""Outreach hooks — the last mile, and the easiest place to lose credibility.

A hook that references a funding round the company never raised is worse than no
hook at all: it costs the rep the reply. So every generated hook must name a fact
that is literally present in the record, and anything that fails that check is
dropped rather than softened.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from lea.config import Settings
from lea.llm.base import LLMClient
from lea.models import CompanyRecord, ICPFit, OutreachHook, Signal

__all__ = [
    "generate_hooks",
    "MAX_HOOK_WORDS",
    "CLICHES",
    "is_grounded",
    "filter_hooks",
]

MAX_HOOK_WORDS = 45
MAX_HOOKS = 3
MIN_HOOKS = 2

CLICHES: list[str] = [
    "hope this finds you well",
    "i was impressed",
    "reaching out because",
    "touch base",
    "circle back",
    "quick question",
    "saw that you guys",
    # Template scaffolds from our own earlier fallback hooks. They are not
    # classic cold-email cliches, but they read as generated and repeated
    # verbatim across every company, which is worse.
    "usually that means the gtm team",
    "came up as a fit on our side",
    "teams on that stack usually hit",
    "most companies at that stage",
    "worth a look?",
    "i wanted to reach out",
    "i came across your",
]

_FALLBACK_HOOKS_SYSTEM = (
    "You write cold-outreach openers for a B2B sales rep. Produce 3 hooks. Each must be "
    "1-2 sentences, under 40 words, and anchored to ONE specific named fact from the "
    "company record — a funding round, a job posting, a tech choice, a product launch, a "
    "named person. Put that exact fact in `grounded_in` and its URL in `source_url`. "
    "Write like a person: no 'hope this finds you well', no 'I was impressed', no "
    "'reaching out because', no 'touch base', no 'circle back', no 'quick question'. "
    "Lead with the observation, then one line connecting it to a problem they plausibly "
    "have. Never invent a fact that is not in the record."
)


def _hooks_prompt() -> str:
    try:  # pragma: no cover - import shim
        from lea.prompts import HOOKS_SYSTEM

        return HOOKS_SYSTEM
    except Exception:
        return _FALLBACK_HOOKS_SYSTEM


class _HookBundle(BaseModel):
    """Wrapper so the extract() contract (one model) can return a list."""

    hooks: list[OutreachHook] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-\.]*")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "it", "its", "their", "they",
    "this", "that", "these", "those", "as", "new", "recent", "recently", "company",
    "team", "teams", "your", "you", "we", "our", "just", "about", "into", "after",
}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def _record_fact_index(record: CompanyRecord) -> list[tuple[set[str], Optional[str]]]:
    """Every fact the record actually asserts, as a token set plus its source URL."""
    facts: list[tuple[set[str], Optional[str]]] = []

    for s in record.signals or []:
        toks = _tokens(s.headline) | _tokens(s.detail) | _tokens(s.kind)
        if s.date:
            toks |= _tokens(str(s.date))
        if toks:
            facts.append((toks, s.source_url))

    for t in record.tech_stack or []:
        toks = _tokens(t.name) | _tokens(t.category) | _tokens(t.evidence)
        if toks:
            facts.append((toks, None))

    profile_bits = [
        record.company_name,
        record.description,
        record.industry,
        record.sub_industry,
        record.hq_city,
        record.hq_country,
        record.employee_range,
        str(record.founded_year) if record.founded_year else None,
    ]
    profile_tokens = set()
    for b in profile_bits:
        profile_tokens |= _tokens(str(b)) if b else set()
    if record.employee_range:
        profile_tokens |= {"headcount", "employees", "size"}
    if profile_tokens:
        facts.append((profile_tokens, record.sources[0] if record.sources else None))

    return facts


def is_grounded(grounded_in: str, record: CompanyRecord, threshold: float = 0.5) -> bool:
    """Does `grounded_in` actually correspond to something in the record?

    Strict on purpose: we require half of the claim's meaningful tokens to appear
    in one single fact, not scattered across the record. "Series B led by Sequoia"
    only passes if some one signal mentions a Series B led by Sequoia.
    """
    claim = _tokens(grounded_in)
    if not claim:
        return False
    for toks, _url in _record_fact_index(record):
        if not toks:
            continue
        overlap = len(claim & toks) / len(claim)
        if overlap >= threshold:
            return True
    return False


def _source_for(grounded_in: str, record: CompanyRecord) -> Optional[str]:
    """Best source URL for a grounded claim: the URL of the fact it matched."""
    claim = _tokens(grounded_in)
    best: tuple[float, Optional[str]] = (0.0, None)
    for toks, url in _record_fact_index(record):
        if not toks or not claim:
            continue
        overlap = len(claim & toks) / len(claim)
        if overlap > best[0] and url:
            best = (overlap, url)
    if best[1]:
        return best[1]
    return record.sources[0] if record.sources else None


# --------------------------------------------------------------------------
# Mechanical filter
# --------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def filter_hooks(hooks: list[OutreachHook], record: CompanyRecord) -> tuple[list[OutreachHook], list[str]]:
    """Drop the bad ones. Returns (kept, reasons_dropped) — reasons are for the trace."""
    kept: list[OutreachHook] = []
    dropped: list[str] = []
    seen_text: set[str] = set()

    for h in hooks or []:
        if h is None:
            continue
        text = (h.hook or "").strip()
        if not text:
            dropped.append("empty hook")
            continue

        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key in seen_text:
            dropped.append(f"duplicate: {text[:40]}...")
            continue

        wc = _word_count(text)
        if wc > MAX_HOOK_WORDS:
            dropped.append(f"too long ({wc} words): {text[:40]}...")
            continue

        lowered = text.lower()
        hit = next((c for c in CLICHES if c in lowered), None)
        if hit:
            dropped.append(f"cliche '{hit}': {text[:40]}...")
            continue

        grounded = (h.grounded_in or "").strip()
        if not grounded:
            dropped.append(f"no grounded_in: {text[:40]}...")
            continue
        if not is_grounded(grounded, record):
            dropped.append(f"ungrounded claim '{grounded[:40]}': not present in the record")
            continue

        seen_text.add(key)
        kept.append(
            OutreachHook(
                angle=(h.angle or grounded[:60]).strip(),
                hook=text,
                grounded_in=grounded,
                source_url=h.source_url if h.source_url in set(record.sources or []) else _source_for(grounded, record),
            )
        )
        if len(kept) >= MAX_HOOKS:
            break

    return kept, dropped


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------


# --- specifics extraction -------------------------------------------------
#
# A fallback hook must lean on the *details* inside a signal (the amount, the
# round, the investor, which roles, the product name) rather than splicing the
# raw headline into a fixed sentence. Pasting the headline is what produced
# 'Noticed Ramp's Introducing Stack by Ramp.. Usually that means...'.

_AMOUNT_RE = re.compile(r"\$\s?\d[\d,.]*\s*(?:[MBK]\b|million|billion|bn)", re.I)
_ROUND_RE = re.compile(r"\b(pre-seed|seed|series\s+[a-j])\b", re.I)
_INVESTOR_RE = re.compile(
    r"\bled by\s+([A-Z][\w&.'\-]*(?:\s+[A-Z][\w&.'\-]*){0,3})", re.U
)
_VALUATION_RE = re.compile(r"\bat\s+a\s+(\$\s?\d[\d,.]*\s*(?:[MB]\b|billion|million))", re.I)
_ROLE_COUNT_RE = re.compile(r"\b(\d{1,4})\s+(?:new\s+|open\s+)?(?:roles?|positions?|jobs?|hires?)\b", re.I)
_LAUNCH_RE = re.compile(
    r"\b(?:introducing|launch(?:es|ed|ing)?|announc(?:es|ed|ing)|unveil(?:s|ed)?)\s+"
    r"([A-Z][\w][\w\- ]{1,34}?)(?=\s*(?:[,.:;—–]|$|\bfor\b|\bto\b|\bby\b|\bwith\b))"
)
_ROLE_WORDS = [
    "engineering", "sales", "security", "marketing", "product", "design",
    "support", "finance", "operations", "revenue", "customer success", "data",
]
_AE_RE = re.compile(r"\b(account executives?|AEs?|SDRs?|sales engineers?|RevOps)\b", re.I)


def _tidy(text: str) -> str:
    """Trim a fragment so it can be spliced into a sentence without doubling
    punctuation — the '..' bug."""
    return re.sub(r"\s+", " ", (text or "").strip()).rstrip(" .,:;—–-")


def _specifics(s: Signal) -> dict[str, str]:
    """Named details inside one signal. Missing keys simply are not there."""
    blob = f"{s.headline or ''} {s.detail or ''}"
    spec: dict[str, str] = {}
    for key, rx in (
        ("amount", _AMOUNT_RE), ("round", _ROUND_RE), ("investor", _INVESTOR_RE),
        ("valuation", _VALUATION_RE), ("role_count", _ROLE_COUNT_RE), ("product", _LAUNCH_RE),
    ):
        m = rx.search(blob)
        if m:
            spec[key] = _tidy(m.group(1) if m.groups() else m.group(0))
    roles = [w for w in _ROLE_WORDS if re.search(rf"\b{re.escape(w)}\b", blob, re.I)]
    if roles:
        spec["roles"] = _join_human(roles[:3])
    ae = _AE_RE.search(blob)
    if ae:
        spec["ae"] = _tidy(ae.group(1))
    if s.date:
        spec["date"] = _tidy(str(s.date))
        m = re.match(r"(\d{4})-(\d{2})", spec["date"])
        if m:
            spec["month"] = _MONTHS[int(m.group(2)) - 1]
    return spec


_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def _join_human(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + " and " + items[-1]


# --- one builder per sentence shape ---------------------------------------


def _h_funding(name: str, s: Signal) -> Optional[tuple[str, str, str, Optional[str]]]:
    sp = _specifics(s)
    amount, rnd, inv = sp.get("amount"), sp.get("round"), sp.get("investor")
    if not (amount or rnd):
        return None
    raised = " ".join(x for x in [amount, rnd.title() if rnd else None] if x)
    when = f" in {sp['month']}" if sp.get("month") else ""

    if inv:
        text = (
            f"{inv} led your {raised}{when}, and rounds that size tend to be followed by a "
            f"GTM hiring wave the reporting stack was never sized for. How are you tracking "
            f"pipeline attribution through it?"
        )
    elif sp.get("valuation"):
        text = (
            f"{raised} at a {sp['valuation']} valuation is a lot of new expectation to carry. "
            f"The teams I talk to hit the attribution ceiling about two quarters after a round "
            f"like that."
        )
    else:
        text = (
            f"You raised {raised}{when}. The quarter after a raise is usually when GTM reporting "
            f"starts creaking, because headcount arrives faster than the process around it."
        )
    return (f"{raised} -> GTM scale", text, s.headline, s.source_url)


def _h_hiring(name: str, s: Signal) -> Optional[tuple[str, str, str, Optional[str]]]:
    sp = _specifics(s)
    count, roles, ae = sp.get("role_count"), sp.get("roles"), sp.get("ae")
    if count and ae:
        # "account executives" + "reqs" reads wrong; the role becomes an adjective.
        singular = ae[:-1] if ae.lower().endswith("s") and not ae.lower().endswith("ss") else ae
        opener = f"{count} open {singular.lower() if singular.islower() or len(singular) > 3 else singular} reqs at {name}"
    elif count:
        opener = f"{count} open roles at {name}"
    elif roles:
        opener = f"You are hiring across {roles} at once"
    elif ae:
        opener = f"{name} is bringing on {ae.lower()}"
    else:
        return None
    text = (
        f"{opener} — that is the point where 'who owns this account' stops being obvious "
        f"and starts being a meeting. Curious how you are handling it."
    )
    return ("GTM hiring", text, s.headline, s.source_url)


def _h_product(name: str, s: Signal) -> Optional[tuple[str, str, str, Optional[str]]]:
    sp = _specifics(s)
    product = sp.get("product")
    if not product:
        return None
    text = (
        f"{product} looks like a real surface-area change, not a point release. New surface "
        f"usually means new buyer conversations, and the old segmentation stops fitting. "
        f"Is that landing on your team?"
    )
    return (f"launch: {product}", text, s.headline, s.source_url)


def _h_generic_signal(name: str, s: Signal) -> Optional[tuple[str, str, str, Optional[str]]]:
    head = _tidy(s.headline)
    if not head:
        return None
    # Quote the headline rather than splicing it into a clause — inlining it
    # produced "Saw the news about we persuaded Calvin to join...".
    text = (
        f'"{head}" — moves like that usually reshuffle who owns the number, and the '
        f"reporting takes a quarter to catch up. Is that playing out at {name}?"
    )
    return ("recent news", text, s.headline, s.source_url)


def _h_tech(name: str, record: CompanyRecord) -> Optional[tuple[str, str, str, Optional[str]]]:
    names = [t.name for t in (record.tech_stack or [])[:3] if t.name]
    if not names:
        return None
    stack = _join_human(names)
    grounded = f"{record.tech_stack[0].category} {record.tech_stack[0].name}"
    text = (
        f"{stack} is a deliberate stack, not an accident — someone there has opinions about "
        f"GTM data. Those are usually the teams that notice the seam between them first."
    )
    return ("stack choice", text, grounded, record.sources[0] if record.sources else None)


def _h_profile(name: str, record: CompanyRecord) -> Optional[tuple[str, str, str, Optional[str]]]:
    bits = [
        f"{record.employee_range} people" if record.employee_range else None,
        record.industry,
    ]
    detail = ", ".join(b for b in bits if b)
    if not detail:
        return None
    text = (
        f"{name} sits right where we tend to be useful ({detail}). At that size the GTM data "
        f"is usually spread across four tools and one person's head."
    )
    return ("profile fit", text, detail, record.sources[0] if record.sources else None)


def _fallback_hooks(record: CompanyRecord, icp_fit: Optional[ICPFit], need: int) -> list[OutreachHook]:
    """Template hooks built from the strongest facts we actually hold.

    Plain, but true — which beats an empty hooks list in a CSV a rep has to work.
    """
    name = record.company_name or record.input
    out: list[OutreachHook] = []
    used_shapes: set[str] = set()

    def add(hook: Optional[tuple[str, str, str, Optional[str]]], shape: str) -> None:
        """Append a hook unless we already used this sentence shape."""
        if hook is None or len(out) >= need or shape in used_shapes:
            return
        angle, text, grounded, url = hook
        used_shapes.add(shape)
        out.append(
            OutreachHook(
                angle=angle,
                hook=text,
                grounded_in=grounded,
                source_url=url or (record.sources[0] if record.sources else None),
            )
        )

    # One signal per KIND, strongest first. Taking the first N signals meant a
    # company with three product launches got the same sentence three times.
    by_kind: dict[str, Signal] = {}
    for s in record.signals or []:
        by_kind.setdefault(s.kind, s)

    if "funding" in by_kind:
        add(_h_funding(name, by_kind["funding"]), "funding")
    if "hiring" in by_kind:
        add(_h_hiring(name, by_kind["hiring"]), "hiring")
    if "product" in by_kind:
        add(_h_product(name, by_kind["product"]), "product")
    for k in ("leadership", "partnership", "news"):
        if k in by_kind:
            add(_h_generic_signal(name, by_kind[k]), "signal")

    if len(out) < need and record.tech_stack:
        add(_h_tech(name, record), "tech")

    if len(out) < need and (record.industry or record.employee_range):
        add(_h_profile(name, record), "profile")

    # Only lean on the ICP fit when it is a real assessment. A non-assessment
    # (score 0, confidence 0) would otherwise yield "you scored tier D for us",
    # asserting a judgement that was never actually made.
    if (
        len(out) < need
        and icp_fit is not None
        and icp_fit.criteria
        and icp_fit.score > 0
        and icp_fit.confidence >= 0.3
    ):
        top = max(icp_fit.criteria, key=lambda c: c.score * c.weight)
        out.append(
            OutreachHook(
                angle=f"icp: {top.key}",
                hook=(
                    f"{name} scored tier {icp_fit.tier} for us, mostly on "
                    f"{top.key.replace('_', ' ')}. Curious whether that matches how you would "
                    f"describe the team's priorities right now."
                ),
                grounded_in=f"ICP criterion {top.key}",
                source_url=top.evidence_urls[0] if top.evidence_urls else None,
            )
        )

    # Last resort: distinct honest openers. Two of them, because returning one
    # hook is the same failure as returning none for anyone working the CSV.
    generic = [
        OutreachHook(
            angle="no public signal",
            hook=(
                f"We had {name} flagged for research but found little public detail — "
                f"I would rather ask than guess at what your team is actually solving."
            ),
            grounded_in="no public signal found",
            source_url=record.sources[0] if record.sources else None,
        ),
        OutreachHook(
            angle="direct ask",
            hook=(
                f"No pitch yet: I could not tell from the outside how {name} runs its "
                f"go-to-market today. If I am off base, say so and I will stop."
            ),
            grounded_in="no public signal found",
            source_url=record.sources[0] if record.sources else None,
        ),
    ]
    for g in generic:
        if len(out) >= need:
            break
        out.append(g)
    return out[:need]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _build_hooks_context(record: CompanyRecord, icp_fit: Optional[ICPFit], evidence: list[tuple[str, str]]) -> str:
    lines = ["COMPANY RECORD:", record.model_dump_json(indent=2, exclude={"hooks"})]
    if icp_fit is not None:
        strong = sorted(icp_fit.criteria, key=lambda c: c.score * c.weight, reverse=True)[:3]
        lines += [
            "",
            f"ICP FIT: tier {icp_fit.tier}, {icp_fit.score:.0f}/100 (confidence {icp_fit.confidence:.2f}).",
            "Strongest criteria: " + "; ".join(f"{c.key} {c.score:.0f}/10 — {c.reasoning}" for c in strong),
        ]
    if evidence:
        lines += ["", "EVIDENCE (source URLs you may cite):"]
        for url, text in evidence[:6]:
            snippet = (text or "").strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + " ...[truncated]"
            lines.append(f"--- {url} ---")
            lines.append(snippet)
    lines += [
        "",
        f"Write {MAX_HOOKS} hooks. Each under {MAX_HOOK_WORDS - 5} words, each anchored to a "
        "different specific fact above, with that fact quoted in grounded_in.",
    ]
    return "\n".join(lines)


def generate_hooks(
    record: CompanyRecord,
    icp_fit: Optional[ICPFit],
    evidence: list[tuple[str, str]],
    llm: LLMClient,
    settings: Settings,
) -> list[OutreachHook]:
    """2-3 hooks, each traceable to a fact in the record. Never raises, never
    returns an empty list."""
    evidence = list(evidence or [])

    bundle: Optional[_HookBundle] = None
    try:
        bundle, _usage = llm.extract(
            _hooks_prompt(),
            _build_hooks_context(record, icp_fit, evidence),
            _HookBundle,
            hint="Return {'hooks': [...]} with 3 grounded outreach hooks.",
        )
    except Exception:
        bundle = None

    raw = list(bundle.hooks) if isinstance(bundle, _HookBundle) else []
    kept, _dropped = filter_hooks(raw, record)

    if len(kept) < MIN_HOOKS:
        existing = {re.sub(r"\W+", " ", h.hook.lower()).strip() for h in kept}
        for fb in _fallback_hooks(record, icp_fit, MIN_HOOKS - len(kept)):
            if re.sub(r"\W+", " ", fb.hook.lower()).strip() not in existing:
                kept.append(fb)

    return kept[:MAX_HOOKS]
