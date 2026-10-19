"""
M1：我们要存的 14 个指标 + 20 家公司。

每个指标 = (候选 SEC 标签列表, 类型, 单位)
- 候选标签按优先级排列：按顺序试，第一个有数据的就用它（见学习笔记第 3 块）。
- 类型："duration" = 录像（一段时间的累计，如营收）；"instant" = 照片（某一天的余额，如总资产）。
"""

METRICS = {
    # ── 利润表（录像）──
    "Revenue": (["Revenues",  # 总营收优先：Walmart 用另一个标签会少算会员费（第 4 块）
                 "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "SalesRevenueNet"], "duration", "USD"),
    "CostOfRevenue": (["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"], "duration", "USD"),
    "GrossProfit": (["GrossProfit"], "duration", "USD"),
    "OperatingIncome": (["OperatingIncomeLoss"], "duration", "USD"),
    "NetIncome": (["NetIncomeLoss", "ProfitLoss"], "duration", "USD"),
    "EPSDiluted": (["EarningsPerShareDiluted"], "duration", "USD/shares"),
    "DilutedShares": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "duration", "shares"),
    # ── 现金流量表（录像）──
    "OperatingCashFlow": (["NetCashProvidedByUsedInOperatingActivities"], "duration", "USD"),
    "CapEx": (["PaymentsToAcquirePropertyPlantAndEquipment",
               "PaymentsToAcquireProductiveAssets",
               "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"], "duration", "USD"),
    # ── 资产负债表（照片）──
    "TotalAssets": (["Assets"], "instant", "USD"),
    "TotalLiabilities": (["Liabilities"], "instant", "USD"),
    "StockholdersEquity": (["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "instant", "USD"),
    "Cash": (["CashAndCashEquivalentsAtCarryingValue",
              "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "instant", "USD"),
    "LongTermDebt": (["LongTermDebtNoncurrent", "LongTermDebt",
                      "LongTermDebtAndCapitalLeaseObligations"], "instant", "USD"),
}

COMPANIES = [
    # 科技
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "CSCO", "INTC", "ADBE", "IBM",
    # 消费
    "KO", "PEP", "PG", "MCD", "NKE", "WMT", "COST",
    # 工业 / 能源
    "BA", "CVX",
]

YEARS = range(2020, 2026)  # FY2020–FY2025
