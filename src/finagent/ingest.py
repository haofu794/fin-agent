"""
M1 core: from a company's companyfacts, pick exactly one correct value for each (metric, fiscal year).

Selection rules:
1. Only records from 10-K (annual) filings.
2. Duration metrics (flows, e.g. revenue): end - start must be 350-380 days (drops quarterly data).
   Instant metrics (balances, e.g. total assets): end must be one of the company's fiscal year-end dates.
3. Fiscal year = calendar year of the `end` date. The `fy` field is NOT used: it is the fiscal year
   of the *filing*, and a 10-K reports three years of comparatives under the same `fy`.
4. Candidate tags are tried in priority order: use the first tag that has data for that year.
5. Several records for the same year and tag (original + later restatements) -> take the latest `filed`.
"""
from __future__ import annotations

from datetime import date

from .metrics import METRICS


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _annual_rows(gaap: dict, tag: str, unit: str, kind: str, fy_ends: set[str] | None):
    """All annual records for one tag (rules 1 and 2)."""
    for r in gaap.get(tag, {}).get("units", {}).get(unit, []):
        if r.get("form") not in ("10-K", "10-K/A"):
            continue
        if kind == "duration":
            if not r.get("start") or not 350 <= _days(r["start"], r["end"]) <= 380:
                continue
        else:  # instant
            if fy_ends is not None and r["end"] not in fy_ends:
                continue
        yield r


def fiscal_year_ends(gaap: dict) -> set[str]:
    """Use the end dates of revenue (a duration metric) to find the company's fiscal year-end dates.
    Instant metrics are only accepted on these dates."""
    tags, kind, unit = METRICS["Revenue"]
    return {r["end"] for tag in tags for r in _annual_rows(gaap, tag, unit, kind, None)}


def select_values(ticker: str, facts: dict, years) -> list[dict]:
    gaap = facts["facts"].get("us-gaap", {})
    fy_ends = fiscal_year_ends(gaap)
    out = []
    for metric, (tags, kind, unit) in METRICS.items():
        for year in years:
            for tag in tags:  # rule 4: try tags in priority order
                rows = [r for r in _annual_rows(gaap, tag, unit, kind, fy_ends)
                        if int(r["end"][:4]) == year]                    # rule 3
                if not rows:
                    continue
                best = max(rows, key=lambda r: r["filed"])               # rule 5
                out.append(dict(
                    ticker=ticker, metric=metric, fiscal_year=year,
                    value=best["val"], unit=unit,
                    period_start=best.get("start"), period_end=best["end"],
                    tag=tag, accn=best["accn"], form=best["form"], filed=best["filed"],
                ))
                break  # found: skip the remaining candidates
    return out
