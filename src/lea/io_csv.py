"""CSV in, CSV out.

Input is forgiving because real lead lists are messy: a header row or not, any of
several column names, stray blanks, a `#` comment someone left in. Output is
deliberately flat — one row per company, no nested JSON in cells — because the
person receiving it opens it in a spreadsheet and sorts by icp_score.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from lea.models import EnrichmentResult

__all__ = ["read_targets", "write_csv", "CSV_COLUMNS", "TARGET_COLUMNS", "flatten_result"]

# Preference order: the leftmost match in the header wins.
TARGET_COLUMNS: list[str] = ["company", "company_name", "domain", "name", "website", "url"]

CSV_COLUMNS: list[str] = [
    "input",
    "company_name",
    "domain",
    "industry",
    "employee_range",
    "hq_city",
    "hq_country",
    "founded_year",
    "icp_score",
    "icp_tier",
    "icp_summary",
    "top_signal",
    "signal_count",
    "tech_stack",
    "hook_1",
    "hook_2",
    "hook_3",
    "unknowns",
    "sources",
    "steps",
    "stop_reason",
]

_MAX_SNIFF = 4096


def _looks_like_header(row: Sequence[str]) -> Optional[int]:
    """Return the index of the target column if this row is a header, else None."""
    lowered = [str(c or "").strip().lower().replace(" ", "_") for c in row]
    for want in TARGET_COLUMNS:
        if want in lowered:
            return lowered.index(want)
    return None


def _clean(value: object) -> str:
    return str(value or "").strip().strip('"').strip()


def read_targets(path: str | os.PathLike[str]) -> list[str]:
    """Read company targets from a CSV-with-header or a bare one-per-line list.

    Blank lines and `#` comments are skipped; values are stripped and deduped
    preserving first-seen order.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")

    # Drop comment lines before parsing so a `#` line can't be read as data.
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(lines)
    if not body.strip():
        return []

    try:
        dialect = csv.Sniffer().sniff(body[:_MAX_SNIFF], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel  # single column, no delimiter to find

    rows = [r for r in csv.reader(io.StringIO(body), dialect) if any(_clean(c) for c in r)]
    if not rows:
        return []

    col = _looks_like_header(rows[0])
    if col is not None:
        data_rows = rows[1:]
    else:
        col = 0
        data_rows = rows

    out: list[str] = []
    seen: set[str] = set()
    for r in data_rows:
        if col >= len(r):
            # ragged row: fall back to the first non-empty cell
            value = next((_clean(c) for c in r if _clean(c)), "")
        else:
            value = _clean(r[col])
        if not value or value.startswith("#"):
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _join(items: Iterable[object], sep: str = "; ") -> str:
    return sep.join(str(i).strip() for i in items if str(i or "").strip())


def flatten_result(result: EnrichmentResult) -> dict[str, object]:
    """One EnrichmentResult -> one spreadsheet row."""
    rec = result.record
    fit = rec.icp_fit
    hooks = list(rec.hooks or [])
    signals = list(rec.signals or [])

    row: dict[str, object] = {c: "" for c in CSV_COLUMNS}
    row.update(
        {
            "input": rec.input or "",
            "company_name": rec.company_name or "",
            "domain": rec.domain or "",
            "industry": rec.industry or "",
            "employee_range": rec.employee_range or "",
            "hq_city": rec.hq_city or "",
            "hq_country": rec.hq_country or "",
            "founded_year": rec.founded_year if rec.founded_year is not None else "",
            "icp_score": f"{fit.score:.1f}" if fit else "",
            "icp_tier": fit.tier if fit else "",
            "icp_summary": (fit.summary or "").replace("\n", " ") if fit else "",
            "top_signal": (signals[0].headline if signals else "").replace("\n", " "),
            "signal_count": len(signals),
            "tech_stack": _join(f"{t.category}:{t.name}" for t in rec.tech_stack or []),
            "unknowns": _join(rec.unknowns or [], sep="; "),
            "sources": _join(rec.sources or []),
            "steps": result.stats.steps if result.stats else 0,
            "stop_reason": (result.stats.stop_reason if result.stats else "") or "",
        }
    )
    for i in range(3):
        row[f"hook_{i + 1}"] = hooks[i].hook.replace("\n", " ") if i < len(hooks) else ""
    return row


def write_csv(results: list[EnrichmentResult], path: str | os.PathLike[str]) -> Path:
    """Write one row per company. Creates parent dirs; always writes the header."""
    p = Path(path)
    if str(p.parent):
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results or []:
            writer.writerow(flatten_result(r))
    return p
