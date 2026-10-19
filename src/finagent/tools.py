"""
M2 工具层：模型可以调用的工具。

每个工具由两部分组成：
1. schema（说明书）：发给模型看的 JSON，告诉它"有这个工具、干什么用、要传哪些参数"。
2. 实现（Python 函数）：harness 收到模型的调用请求后，真正去执行的代码。

设计原则：
- 工具出错时不抛异常，而是返回 {"error": ..., "hint": ...}。
  原因：这个结果会被发回给模型，模型看到错误提示后可以自己改正（比如把指标名拼对）。
  如果直接抛异常，整个程序就崩了，模型根本没有机会改正。
- 数据库用只读模式打开：即使出 bug，也不可能修改或删除数据。
- 所有返回值都是 JSON 字符串：模型只能读文字。
"""
from __future__ import annotations

import ast
import json
import operator
import sqlite3

from .config import get_settings
from .metrics import METRICS, YEARS


# ──────────────────────────── 数据库（只读） ────────────────────────────

def _connect_readonly() -> sqlite3.Connection:
    path = get_settings().db_path
    # mode=ro：只读。任何 INSERT / UPDATE / DELETE / DROP 都会直接报错。
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _fmt(value: float, unit: str) -> str:
    """391035000000 → '391.035B USD'。只用于显示，数据库里存的始终是原始值。"""
    if unit == "USD":
        for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= size:
                return f"{value / size:,.3f}{suffix} USD"
        return f"{value:,.0f} USD"
    if unit == "shares":
        return f"{value / 1e9:,.3f}B shares" if abs(value) >= 1e9 else f"{value / 1e6:,.1f}M shares"
    return f"{value} {unit}"  # EPS：美元/股，不换算


# ──────────────────────────── 工具 1：get_metric ────────────────────────────

def get_metric(ticker: str, metric: str, fiscal_year: int) -> dict:
    ticker = str(ticker).upper().strip()
    # 白名单检查：在查数据库之前，先确认参数是合法的。
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


# ──────────────────────────── 工具 2：calculator ────────────────────────────
# 不用 Python 自带的 eval()：eval 能执行任意代码，
# 例如 eval("__import__('os').system('rm -rf ~')") 会删掉你的文件。
# 这里先把表达式解析成"语法树"，然后只允许数字和 + - * / ** ( ) 通过。

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
            raise ValueError("exponent too large")  # 防止 9**9**9 这种算到死机的输入
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported element: {ast.dump(node)[:60]}")


def calculator(expression: str) -> dict:
    expr = str(expression).replace(",", "")  # 允许 "391,035" 这种写法
    try:
        result = _eval_node(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError:
        return {"error": "division by zero"}
    except (SyntaxError, ValueError) as e:
        return {"error": f"invalid expression: {e}",
                "hint": "only numbers and + - * / ** ( ) are allowed, e.g. (391.035-383.285)/383.285*100"}
    return {"expression": expression, "result": result}


# ──────────────────────────── 工具 3：list_companies ────────────────────────────

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


# ──────────────────────────── 工具 4：search_filings ────────────────────────────

SEARCH_MODE = "hybrid"   # "hybrid" / "bm25" / "vector"。M5 做 ablation 时由评测脚本修改

def search_filings(query: str, ticker: str | None = None, fiscal_year: int | None = None,
                   top_k: int = 5) -> dict:
    from .rag import get_index   # 用到时才加载索引（几秒钟），只查数字的问题不用等

    ticker = str(ticker).upper().strip() if ticker else None
    try:
        fiscal_year = int(fiscal_year) if fiscal_year else None
        top_k = max(1, min(int(top_k), 8))   # 最多 8 块：再多会挤占模型的上下文
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


# ──────────────────────────── 说明书（发给模型） ────────────────────────────
# OpenAI tool calling 格式。description 写得越清楚，模型越不容易用错。

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

# 名字 → 函数。harness 收到 "get_metric" 这个名字，就从这里找到对应的函数去执行。
TOOL_FUNCTIONS = {"get_metric": get_metric, "calculator": calculator,
                  "search_filings": search_filings, "list_companies": list_companies}


def run_tool(name: str, arguments: str | dict) -> str:
    """harness 唯一需要调用的入口：工具名 + 参数（模型给的 JSON 字符串）→ 结果（JSON 字符串）。"""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"unknown tool '{name}'", "hint": f"available: {list(TOOL_FUNCTIONS)}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"arguments are not valid JSON: {e}"})
    try:
        result = TOOL_FUNCTIONS[name](**args)
    except TypeError as e:  # 参数名写错、少传参数
        return json.dumps({"error": f"bad arguments: {e}"})
    return json.dumps(result, ensure_ascii=False)
