"""
M1: download data for 20 companies -> select values -> store in data/finance.db

Run: python scripts/build_db.py
About 1 minute the first time (downloads); a few seconds afterwards (local cache).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finagent.config import get_settings
from finagent.db import connect, upsert_company, upsert_financial
from finagent.ingest import fiscal_year_ends, select_values
from finagent.metrics import COMPANIES, METRICS, YEARS
from finagent.sec import company_facts, ticker_to_cik

settings = get_settings()
conn = connect(settings.db_path)
ciks = ticker_to_cik()
expected = len(METRICS) * len(YEARS)

for ticker in COMPANIES:
    cik = ciks[ticker]
    facts = company_facts(cik)
    gaap = facts["facts"].get("us-gaap", {})
    last_end = max(fiscal_year_ends(gaap))
    upsert_company(conn, ticker, cik, facts["entityName"], last_end[5:])  # e.g. '09-28'
    rows = select_values(ticker, facts, YEARS)
    for row in rows:
        upsert_financial(conn, row)
    print(f"{ticker:5} {facts['entityName'][:28]:28} 存入 {len(rows):3}/{expected} 个数字")

conn.commit()
total = conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
print(f"\n完成：{settings.db_path}，共 {total} 行")
