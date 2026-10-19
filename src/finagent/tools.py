"""
M2 tool layer: the tools the model can call.

Each tool has two parts:
1. A schema: JSON sent to the model describing what the tool does and which arguments it takes.
2. An implementation: the Python function the harness runs when the model calls the tool.

Design principles:
- Tools never raise; on failure they return {"error": ..., "hint": ...}.
  The result is sent back to the model, which can then correct itself (e.g. fix a misspelled metric).
  Raising would crash the run and give the model no chance to recover.
- The database is opened read-only, so even a bug cannot modify or delete data.
- Every tool returns a JSON string, because the model can only read text.
"""
from __future__ import annotations

import ast
import json
import operator
import sqlite3

from .config import get_settings
from .metrics import METRICS, YEARS


# ──────────────────────────── Database (read-only) ────────────────────────────

def _connect_readonly() -> sqlite3.Connection:
    path = get_settings().db_path
    # mode=ro: read-only. Any INSERT / UPDATE / DELETE / DROP fails immediately.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _fmt(value: float, unit: str) -> str:
    """391035000000 -> '391.035B USD'. Display only; the database always stores raw values."""
    if unit == "USD":
        for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= size:
                return f"{value / size:,.3f}{suffix} USD"
        return f"{value:,.0f} USD"
    if unit == "shares":
        return f"{value / 1e9:,.3f}B shares" if abs(value) >= 1e9 else f"{value / 1e6:,.1f}M shares"
    return f"{value} {unit}"  # EPS is USD per share: no scaling


# ──────────────────────────── Tool 1: get_metric ────────────────────────────

def get_metric(ticker: str, metric: str, fiscal_year: int) -> dict:
    ticker = str(ticker).upper().strip()
    # Whitelist check: validate arguments before touching the database.
    if metric not in METRICS:
        return {"error": f"unknown metric '{metric}'",
                "hint": f"valid metrics: {sorted(METRICS)}"}
    try:
        fiscal_year = int(fiscal_year)
    except (TypeError, ValueError):
        return {"error": f"fiscal_year must be an integer, got '{fiscal_year}'"}

    with _connect_readonly() as conn:
        if not conn.execute("SELECT 1 FROM companies WHERE ticker=?", (ticker,)).fetchone():
            return {"error": f"unknown company '{ticker}'",
                    "hint": "call list_companies to see available tickers"}
        row = conn.execute(
            "SELECT * FROM financials WHERE ticker=? AND metric=? AND fiscal_year=?",
            (ticker, metric, fiscal_year),
        ).fetchone()

    if row is None:
        if fiscal_year not in YEARS:
            return {"error": f"fiscal_year {fiscal_year} out of range",
                    "hint": f"available years: {YEARS.start}-{YEARS.stop - 1}"}
        return {"error": "not_reported",
                "hint": f"{ticker} did not report {metric} for FY{fiscal_year} as a separate line item. "
                        "It may be derivable from other metrics "
                        "(e.g. GrossProfit = Revenue - CostOfRevenue)."}

    return {
        "ticker": ticker, "metric": metric, "fiscal_year": fiscal_year,
        "value": row["value"], "unit": row["unit"],
        "display": _fmt(row["value"], row["unit"]),
        "period": f"{row['period_start'] or ''} to {row['period_end']}".strip(),
        "source": {"form": row["form"], "accession": row["accn"],
                   "filed": row["filed"], "xbrl_tag": row["tag"]},
    }


# ──────────────────────────── Tool 2: calculator ────────────────────────────
# Python's built-in eval() is not used: it executes arbitrary code,
# e.g. eval("__import__('os').system('rm -rf ~')") would delete your files.
# Instead the expression is parsed into an AST and only numbers and + - * / ** ( ) are allowed.

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")  # guard against inputs like 9**9**9 that never finish
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported element: {ast.dump(node)[:60]}")


def calculator(expression: str) -> dict:
    expr = str(expression).replace(",", "")  # accept thousands separators such as "391,035"
    try:
        result = _eval_node(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError:
        return {"error": "division by zero"}
    except (SyntaxError, ValueError) as e:
        return {"error": f"invalid expression: {e}",
                "hint": "only numbers and + - * / ** ( ) are allowed, e.g. (391.035-383.285)/383.285*100"}
    return {"expression": expression, "result": result}


# ──────────────────────────── Tool 3: list_companies ────────────────────────────

def list_companies() -> dict:
    with _connect_readonly() as conn:
        rows = conn.execute("SELECT ticker, name, fiscal_year_end FROM companies ORDER BY ticker").fetchall()
    return {
        "companies": [dict(r) for r in rows],
        "metrics": sorted(METRICS),
        "fiscal_years": f"{YEARS.start}-{YEARS.stop - 1}",
        "note": "fiscal_year = the calendar year in which the fiscal year ends "
                "(e.g. Apple FY2024 ends 2024-09-28; NVIDIA FY2024 ends 2024-01-28).",
    }


# ──────────────────────────── Tool 4: search_filings ────────────────────────────

SEARCH_MODE = "hybrid"   # "hybrid" / "bm25" / "vector"; the eval script changes this for ablations (M5)

def search_filings(query: str, ticker: str | None = None, fiscal_year: int | None = None,
                   top_k: int = 5) -> dict:
    from .rag import get_index   # lazy import: loading the index takes a few seconds, numeric-only questions skip it

    ticker = str(ticker).upper().strip() if ticker else None
    try:
        fiscal_year = int(fiscal_year) if fiscal_year else None
        top_k = max(1, min(int(top_k), 8))   # at most 8 passages: more would crowd the model's context
    except (TypeError, ValueError):
        return {"error": "fiscal_year and top_k must be integers"}

    hits = get_index().search(query, ticker=ticker, fiscal_year=fiscal_year, top_k=top_k, mode=SEARCH_MODE)
    if not hits:
        return {"error": "no_results",
                "hint": f"No 10-K text indexed for ticker={ticker}, fiscal_year={fiscal_year}. "
                        "Only FY2024 and FY2025 10-Ks are indexed; call list_companies for tickers."}
    return {"results": [{
        "chunk_id": h["chunk_id"], "ticker": h["ticker"], "fiscal_year": h["fiscal_year"],
        "section": h["section"], "accession": h["accn"], "text": h["text"],
    } for h in hits]}


# ──────────────────────────── Schemas (sent to the model) ────────────────────────────
# OpenAI tool-calling format. The clearer the descriptions, the less often the model misuses a tool.

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_metric",
        "description": "Look up one financial metric for one company and one fiscal year from SEC 10-K filings. "
                       "Returns the exact value, unit, period and source filing.",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
            "metric": {"type": "string", "enum": sorted(METRICS), "description": "Metric name"},
            "fiscal_year": {"type": "integer", "description": "Fiscal year, e.g. 2024"},
        }, "required": ["ticker", "metric", "fiscal_year"]},
    }},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. ALWAYS use this for any arithmetic "
                       "(growth rates, ratios, differences) instead of computing in your head.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. (391.035-383.285)/383.285*100"},
        }, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "search_filings",
        "description": "Search the text of 10-K filings (Item 1 Business, Item 1A Risk Factors, "
                       "Item 7 MD&A) for qualitative questions: risks, strategy, reasons for changes, "
                       "business descriptions. Uses hybrid keyword + semantic search. "
                       "Only FY2024 and FY2025 10-Ks are indexed. Do NOT use it to look up numbers "
                       "(use get_metric). Always pass ticker when the question is about one company, "
                       "and do not repeat the company name inside query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in English, e.g. 'supply chain concentration risk'"},
            "ticker": {"type": "string", "description": "Optional. Restrict to one company, e.g. AAPL"},
            "fiscal_year": {"type": "integer", "description": "Optional. 2024 or 2025"},
            "top_k": {"type": "integer", "description": "Number of passages to return, 1-8 (default 5)"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "list_companies",
        "description": "List available companies, metric names and fiscal years.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

# Name -> function. When the model calls "get_metric", the harness looks up the function here.
TOOL_FUNCTIONS = {"get_metric": get_metric, "calculator": calculator,
                  "search_filings": search_filings, "list_companies": list_companies}


def run_tool(name: str, arguments: str | dict) -> str:
    """Single entry point for the harness: tool name + arguments (JSON string from the model) -> result (JSON string)."""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"unknown tool '{name}'", "hint": f"available: {list(TOOL_FUNCTIONS)}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"arguments are not valid JSON: {e}"})
    try:
        result = TOOL_FUNCTIONS[name](**args)
    except TypeError as e:  # misspelled or missing arguments
        return json.dumps({"error": f"bad arguments: {e}"})
    return json.dumps(result, ensure_ascii=False)
