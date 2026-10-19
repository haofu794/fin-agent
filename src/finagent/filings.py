"""
M3 step 1: download 10-K filings (HTML), convert them to plain text, and split them into Items (sections).

Pipeline:
1. submissions API -> find which file is the 10-K for a given company and fiscal year
2. download that HTML file (cached in data/raw/10k/)
3. HTML -> plain text (strip tags and numeric tables)
4. find headings such as "Item 1.", "Item 1A.", "Item 7." and split the text into sections
"""
from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .config import get_settings
from .sec import _get_json, _get_text

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Sections we keep. Item 8 (financial statements) is skipped: its numbers are already in SQLite
# and it is mostly tables.
ITEMS = {
    "1":  "Business",
    "1A": "Risk Factors",
    "7":  "Management's Discussion and Analysis (MD&A)",
}
MIN_SECTION_CHARS = 3000  # anything shorter is almost certainly a bad split (a TOC line or "see page X")
TITLES = {**ITEMS, "FULL": "Full document (standard Items not detected)"}


def find_10k(cik: int, fiscal_year: int) -> dict | None:
    """Among all of the company's filings, find the 10-K whose reportDate (fiscal year end) falls in fiscal_year."""
    raw = get_settings().data_dir / "raw"
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", raw / f"submissions_{cik}.json")
    r = sub["filings"]["recent"]
    for i, form in enumerate(r["form"]):
        if form == "10-K" and r["reportDate"][i].startswith(str(fiscal_year)):
            accn = r["accessionNumber"][i]
            return {"accn": accn, "report_date": r["reportDate"][i], "filed": r["filingDate"][i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{r['primaryDocument'][i]}"}
    return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    # Tables: remove only mostly-numeric ones (financial data tables).
    # Removing all tables is wrong: many filers put section headings in a table
    # (one cell "Item 7.", one cell with the title), so the headings would be lost too.
    for t in soup.find_all("table"):
        chars = [c for c in t.get_text() if not c.isspace()]
        if chars and sum(c.isdigit() or c in "$,.%()-—" for c in chars) / len(chars) > 0.4:
            t.decompose()
    # Also remove hidden inline-XBRL metadata (ix:header); otherwise the text starts with a pile of tag names.
    for t in soup.find_all(style=re.compile(r"display:\s*none")):
        t.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")                 # non-breaking space -> normal space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)         # collapse multiple blank lines
    return text.strip()


# Heading formats: "Item 1A." / "ITEM 1A:" / "Item 1A — Risk Factors"; must be at the start of a line
_HEADING = re.compile(r"^\s*item\s+(\d{1,2}[A-C]?)\s*[.:\-—–]", re.IGNORECASE | re.MULTILINE)


def split_items(text: str) -> dict[str, str]:
    """
    Split the full text into sections. The difficulty: every heading appears twice,
    once in the table of contents and once in the body.
    The TOC "section" is a single line (the next heading follows immediately), while the body
    section runs to thousands of characters.
    So for each Item, among all its occurrences, pick the one with the longest text before the next heading.
    """
    marks = [(m.start(), m.group(1).upper()) for m in _HEADING.finditer(text)]
    best: dict[str, str] = {}
    for i, (pos, item) in enumerate(marks):
        if item not in ITEMS:
            continue
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end].strip()
        if len(body) > len(best.get(item, "")):
            best[item] = body
    return best


def load_10k_sections(ticker: str, cik: int, fiscal_year: int) -> dict | None:
    info = find_10k(cik, fiscal_year)
    if info is None:
        return None
    cache = get_settings().data_dir / "raw" / "10k" / f"{ticker}_{fiscal_year}.htm"
    html = _get_text(info["url"], cache)
    text = html_to_text(html)
    sections = split_items(text)
    if any(len(sections.get(k, "")) < MIN_SECTION_CHARS for k in ITEMS):
        # Splitting failed (a section is missing or too short). Typical causes:
        #   Intel: each Item only says "see pages 31-46"; the body follows its own structure.
        #   IBM: MD&A lives in a separate document (an annual-report exhibit).
        #   McDonald's / Chevron: unusual heading formats produce mismatched sections.
        # Fallback: store the whole document as one "section". Retrieval still works,
        # but citations cannot name the Item. No section label is better than a wrong one.
        sections = {"FULL": text}
    return {"ticker": ticker, "fiscal_year": fiscal_year, **info,
            "sections": {k: {"title": TITLES[k], "text": v} for k, v in sections.items()}}
