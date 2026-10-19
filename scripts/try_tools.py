"""M2 练习：不经过模型，直接调用工具，看看每种情况返回什么。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finagent.tools import run_tool

cases = [
    ("正常查询",           "get_metric", '{"ticker":"AAPL","metric":"Revenue","fiscal_year":2024}'),
    ("小写 ticker",        "get_metric", '{"ticker":"aapl","metric":"Revenue","fiscal_year":2023}'),
    ("公司没报这个数",      "get_metric", '{"ticker":"WMT","metric":"GrossProfit","fiscal_year":2025}'),
    ("指标名写错",         "get_metric", '{"ticker":"AAPL","metric":"Sales","fiscal_year":2024}'),
    ("没有这家公司",        "get_metric", '{"ticker":"JPM","metric":"Revenue","fiscal_year":2024}'),
    ("年份超出范围",        "get_metric", '{"ticker":"AAPL","metric":"Revenue","fiscal_year":2015}'),
    ("SQL 注入尝试",       "get_metric", '{"ticker":"AAPL; DROP TABLE financials","metric":"Revenue","fiscal_year":2024}'),
    ("增长率",             "calculator", '{"expression":"(391.035-383.285)/383.285*100"}'),
    ("Walmart 毛利",       "calculator", '{"expression":"680985000000 - 511753000000"}'),
    ("除以 0",             "calculator", '{"expression":"1/0"}'),
    ("恶意代码",           "calculator", '{"expression":"__import__(\'os\').system(\'ls\')"}'),
    ("不存在的工具",        "delete_db",  '{}'),
    ("参数不是合法 JSON",   "calculator", '{"expression": '),
]
for label, name, args in cases:
    print(f"── {label}\n   {run_tool(name, args)[:230]}")
print("── list_companies（前 200 字）\n  ", run_tool("list_companies", "{}")[:200])
