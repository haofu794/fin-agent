"""第 2 块练习：手动把 Apple FY2023 / FY2024 营收写进 SQLite，再查出来。"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finagent.db import connect, upsert_company, upsert_financial

db = Path(tempfile.gettempdir()) / "try_finance.db"   # 练习用临时库，不污染 data/
db.unlink(missing_ok=True)
conn = connect(db)
upsert_company(conn, "AAPL", 320193, "Apple Inc.", "09-28")
base = dict(ticker="AAPL", metric="Revenue", unit="USD",
            tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            accn="0000320193-24-000123", form="10-K", filed="2024-11-01")
upsert_financial(conn, {**base, "fiscal_year": 2024, "value": 391035000000,
                        "period_start": "2023-10-01", "period_end": "2024-09-28"})
upsert_financial(conn, {**base, "fiscal_year": 2023, "value": 383285000000,
                        "period_start": "2022-09-25", "period_end": "2023-09-30"})
# 故意再插一次 2024（模拟重述）：主键相同 → 覆盖，不会出现两行
upsert_financial(conn, {**base, "fiscal_year": 2024, "value": 391035000000,
                        "period_start": "2023-10-01", "period_end": "2024-09-28"})
conn.commit()

for r in conn.execute("SELECT ticker, metric, fiscal_year, value, accn FROM financials ORDER BY fiscal_year"):
    print(dict(r))
print("行数:", conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0])
