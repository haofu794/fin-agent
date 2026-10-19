"""
M3 step 1: download the FY2024 and FY2025 10-Ks of 20 companies, split them into sections,
and save them to data/filings/*.json
Run: python scripts/build_filings.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finagent.config import get_settings
from finagent.filings import ITEMS, load_10k_sections
from finagent.metrics import COMPANIES
from finagent.sec import ticker_to_cik

FILING_YEARS = [2024, 2025]
out_dir = get_settings().data_dir / "filings"
out_dir.mkdir(parents=True, exist_ok=True)
ciks = ticker_to_cik()
problems = []

for ticker in COMPANIES:
    for fy in FILING_YEARS:
        doc = load_10k_sections(ticker, ciks[ticker], fy)
        if doc is None:
            problems.append(f"{ticker} FY{fy}: 找不到 10-K")
            continue
        (out_dir / f"{ticker}_{fy}.json").write_text(json.dumps(doc, ensure_ascii=False))
        if "FULL" in doc["sections"]:
            n = len(doc["sections"]["FULL"]["text"])
            problems.append(f"{ticker} FY{fy}: 没有标准 Item 结构，整篇存为 FULL（{n:,} 字符）")
            print(f"{ticker:5} FY{fy}  FULL: {n:,}")
            continue
        lens = {k: len(doc["sections"].get(k, {}).get("text", "")) for k in ITEMS}
        for k, n in lens.items():
            if n < 3000:   # under 3,000 chars is probably a bad split (only a TOC line, or "see exhibit")
                problems.append(f"{ticker} FY{fy} Item {k}: 只有 {n} 字符")
        print(f"{ticker:5} FY{fy}  " + "  ".join(f"Item {k}: {n:>7,}" for k, n in lens.items()))

print("\n需要检查：" if problems else "\n全部正常")
for p in problems:
    print("  ⚠", p)
