"""
M3 第 1 步：下载 10-K 原文（HTML），转成纯文字，按 Item（章节）切开。

流程：
1. submissions 接口 → 找到某家公司某个财年的 10-K 是哪个文件
2. 下载这个 HTML 文件（缓存到 data/raw/10k/）
3. HTML → 纯文字（去掉标签、表格）
4. 找到 "Item 1." "Item 1A." "Item 7." 这些标题，把文字切成章节
"""
from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .config import get_settings
from .sec import _get_json, _get_text

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# 我们要的章节。Item 8（财务报表）不要：里面的数字已经在 SQLite 里了，而且大部分是表格。
ITEMS = {
    "1":  "Business",
    "1A": "Risk Factors",
    "7":  "Management's Discussion and Analysis (MD&A)",
}
MIN_SECTION_CHARS = 3000  # 少于这个长度，基本可以肯定是切错了（只切到目录或一句"见第 X 页"）
TITLES = {**ITEMS, "FULL": "Full document (standard Items not detected)"}


def find_10k(cik: int, fiscal_year: int) -> dict | None:
    """在这家公司提交过的所有文件里，找到 reportDate（财年最后一天）在 fiscal_year 的那份 10-K。"""
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
    # 表格：只删"数字为主"的（财务数据表）。
    # 不能全删：很多公司把章节标题放在表格里（一格 "Item 7."，一格标题），全删会连标题一起丢掉。
    for t in soup.find_all("table"):
        chars = [c for c in t.get_text() if not c.isspace()]
        if chars and sum(c.isdigit() or c in "$,.%()-—" for c in chars) / len(chars) > 0.4:
            t.decompose()
    # 隐藏的 XBRL 元数据块（ix:header）也要去掉，否则文字开头会有一大堆标签名
    for t in soup.find_all(style=re.compile(r"display:\s*none")):
        t.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")                 # 不换行空格 → 普通空格
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)         # 多个空行 → 一个空行
    return text.strip()


# 标题的样子："Item 1A." / "ITEM 1A:" / "Item 1A — Risk Factors"，必须在一行的开头
_HEADING = re.compile(r"^\s*item\s+(\d{1,2}[A-C]?)\s*[.:\-—–]", re.IGNORECASE | re.MULTILINE)


def split_items(text: str) -> dict[str, str]:
    """
    把全文切成章节。难点：每个标题会出现两次——
    一次在开头的目录（Table of Contents）里，一次在正文里。
    目录里的"章节"只有一行字（下一行就是下一个标题），正文里的有几千字。
    所以：对每个 Item，在它出现的所有位置里，选"到下一个标题之间文字最长"的那一个。
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
        # 切分失败（有章节缺失或太短）。常见原因：
        #   Intel：Item 下面只有"见第 31–46 页"，正文按自己的结构编排；
        #   IBM：MD&A 放在另一份文件（年报附件）里；
        #   McDonald's / Chevron：标题格式特殊，切出来的内容对不上。
        # 退而求其次：整篇作为一个"章节"。检索照样能用，只是引用时说不出是哪个 Item。
        # 宁可"没有章节名"，也不要"章节名是错的"。
        sections = {"FULL": text}
    return {"ticker": ticker, "fiscal_year": fiscal_year, **info,
            "sections": {k: {"title": TITLES[k], "text": v} for k, v in sections.items()}}
