"""
Tool-calling smoke test: check that the configured model asks to call a tool instead of making up an answer.
This is a prerequisite for the agent: a model without tool calling cannot drive it.

Usage: python scripts/test_tool_call.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI
from finagent.config import get_settings

# A fake tool that looks up company revenue. We only check whether the model requests it correctly.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_revenue",
        "description": "Get a company's annual revenue (USD) for a fiscal year from the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                "fiscal_year": {"type": "integer", "description": "Fiscal year, e.g. 2024"},
            },
            "required": ["ticker", "fiscal_year"],
        },
    },
}]

s = get_settings()
client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
resp = client.chat.completions.create(
    model=s.llm_model,
    messages=[{"role": "user", "content": "What was Apple's revenue in fiscal year 2024? Use the tool."}],
    tools=TOOLS,
)
msg = resp.choices[0].message

if msg.tool_calls:
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    print("✅ 模型发起了工具调用：")
    print(f"   函数名：{call.function.name}")
    print(f"   参数：  {args}")
    ok = call.function.name == "get_revenue" and str(args.get("ticker", "")).upper() == "AAPL"
    print("✅ 参数正确，这个模型可以用来做 Agent！" if ok else "⚠️ 调用了工具但参数不对，换个更大的模型试试。")
else:
    print("❌ 模型没有调用工具，而是直接回答了：")
    print("  ", (msg.content or "")[:300])
    print("这个模型不适合做 Agent，换一个支持 tool calling 的模型。")
