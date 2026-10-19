"""
M1: the 14 metrics and 20 companies we store.

Each metric = (candidate SEC XBRL tags, kind, unit)
- Candidate tags are in priority order: the first one with data is used.
- kind: "duration" = a flow accumulated over a period (e.g. revenue);
        "instant"  = a balance on a single date (e.g. total assets).
"""

METRICS = {
    # ── Income statement (duration) ──
    "Revenue": (["Revenues",  # total revenue first: for Walmart the other tag omits membership fees
                 "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "SalesRevenueNet"], "duration", "USD"),
    "CostOfRevenue": (["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"], "duration", "USD"),
    "GrossProfit": (["GrossProfit"], "duration", "USD"),
    "OperatingIncome": (["OperatingIncomeLoss"], "duration", "USD"),
    "NetIncome": (["NetIncomeLoss", "ProfitLoss"], "duration", "USD"),
    "EPSDiluted": (["EarningsPerShareDiluted"], "duration", "USD/shares"),
    "DilutedShares": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "duration", "shares"),
    # ── Cash flow statement (duration) ──
    "OperatingCashFlow": (["NetCashProvidedByUsedInOperatingActivities"], "duration", "USD"),
    "CapEx": (["PaymentsToAcquirePropertyPlantAndEquipment",
               "PaymentsToAcquireProductiveAssets",
               "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"], "duration", "USD"),
    # ── Balance sheet (instant) ──
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
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "CSCO", "INTC", "ADBE", "IBM",
    # Consumer
    "KO", "PEP", "PG", "MCD", "NKE", "WMT", "COST",
    # Industrials / energy
    "BA", "CVX",
]

YEARS = range(2020, 2026)  # FY2020-FY2025
