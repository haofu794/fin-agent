"""
M3 第 2 步：把章节切成小块（chunk）。

规则：
- 每块最多 MAX_WORDS 个英文单词（≈ 500 tokens；英文里 1 个词平均约 1.3 个 token）。
- 按段落装箱：一段一段往块里放，放不下了就开新块 → 尽量不把一句话从中间切断。
- 相邻两块重叠 OVERLAP_WORDS 个词 → 一句关键的话正好落在边界上时，两块里都能看到它的上下文。
- 每块开头加一行"标签"：公司、财年、章节 → 块被单独拿出来时，也知道它讲的是谁。
"""
from __future__ import annotations

import re

MAX_WORDS = 380
OVERLAP_WORDS = 40


def _split_long(paragraph: str) -> list[str]:
    """极少数段落本身就超过 MAX_WORDS：按句子再切。"""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > MAX_WORDS:
            out.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    # 连句子都超长（例如没有标点的长列表）：硬切
    final = []
    for piece in out:
        w = piece.split()
        final += [" ".join(w[i:i + MAX_WORDS]) for i in range(0, len(w), MAX_WORDS)]
    return final


def chunk_text(text: str) -> list[str]:
    paragraphs = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())          # 段落内部的换行、多余空格合并
        if len(p.split()) < 3:           # 只有一两个词的"段落"多半是页码、页眉，跳过
            continue
        paragraphs += _split_long(p) if len(p.split()) > MAX_WORDS else [p]

    chunks, cur = [], []
    for p in paragraphs:
        if cur and len(" ".join(cur + [p]).split()) > MAX_WORDS:
            chunks.append(" ".join(cur))
            tail = " ".join(cur).split()[-OVERLAP_WORDS:]   # 上一块的最后 40 个词，作为下一块的开头
            cur = [" ".join(tail)]
        cur.append(p)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_filing(doc: dict) -> list[dict]:
    """一份 10-K（filings.py 的输出）→ 一组带出处信息的小块。"""
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
