"""
M3 step 2: split sections into chunks.

Rules:
- At most MAX_WORDS English words per chunk (~500 tokens; one English word averages ~1.3 tokens).
- Pack whole paragraphs: add paragraphs until the next one would not fit, then start a new chunk,
  so sentences are rarely cut in the middle.
- Consecutive chunks overlap by OVERLAP_WORDS words, so a key sentence on a boundary keeps its context in both.
- Each chunk starts with a header line (company, fiscal year, section), so a chunk retrieved on its own
  still says which company it is about.
"""
from __future__ import annotations

import re

MAX_WORDS = 380
OVERLAP_WORDS = 40


def _split_long(paragraph: str) -> list[str]:
    """A few paragraphs exceed MAX_WORDS on their own: split them by sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > MAX_WORDS:
            out.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    # Even a single sentence may be too long (e.g. a long list without punctuation): hard split.
    final = []
    for piece in out:
        w = piece.split()
        final += [" ".join(w[i:i + MAX_WORDS]) for i in range(0, len(w), MAX_WORDS)]
    return final


def chunk_text(text: str) -> list[str]:
    paragraphs = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())          # collapse line breaks and extra spaces inside the paragraph
        if len(p.split()) < 3:           # one- or two-word "paragraphs" are usually page numbers or headers
            continue
        paragraphs += _split_long(p) if len(p.split()) > MAX_WORDS else [p]

    chunks, cur = [], []
    for p in paragraphs:
        if cur and len(" ".join(cur + [p]).split()) > MAX_WORDS:
            chunks.append(" ".join(cur))
            tail = " ".join(cur).split()[-OVERLAP_WORDS:]   # last 40 words of the previous chunk open the next one
            cur = [" ".join(tail)]
        cur.append(p)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_filing(doc: dict) -> list[dict]:
    """One 10-K (output of filings.py) -> a list of chunks with source metadata."""
    out = []
    for item, sec in doc["sections"].items():
        label = "Full 10-K" if item == "FULL" else f"Item {item} {sec['title']}"
        header = f"[{doc['ticker']} | FY{doc['fiscal_year']} 10-K | {label}]"
        for i, body in enumerate(chunk_text(sec["text"])):
            out.append({
                "chunk_id": f"{doc['ticker']}_{doc['fiscal_year']}_{item}_{i:03d}",
                "ticker": doc["ticker"], "fiscal_year": doc["fiscal_year"],
                "item": item, "section": label, "accn": doc["accn"], "url": doc["url"],
                "text": f"{header}\n{body}",
            })
    return out
