"""
Act as an MCP client: start scripts/mcp_server.py as a subprocess, list its tools and call a few,
exactly the way Claude Desktop would. Useful to check the server before wiring it into a real client.

Usage: python scripts/try_mcp.py
"""
import json
import sys
from pathlib import Path

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SERVER = Path(__file__).resolve().parent / "mcp_server.py"

CALLS = [
    ("get_metric", {"ticker": "AAPL", "metric": "Revenue", "fiscal_year": 2024}),
    ("calculator", {"expression": "(391.035-383.285)/383.285*100"}),
    ("get_metric", {"ticker": "AAPL", "metric": "Sales", "fiscal_year": 2024}),   # wrong metric on purpose
    ("search_filings", {"query": "outsourcing partners manufacturing countries", "ticker": "AAPL",
                        "fiscal_year": 2024, "top_k": 2}),
]


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"connected to: {init.server_info.name} {init.server_info.version}")
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            for name, args in CALLS:
                res = await session.call_tool(name, args)
                text = res.content[0].text
                if name == "search_filings" and not res.is_error:
                    text = json.dumps([r["chunk_id"] for r in json.loads(text)["results"]])
                print(f"\n{name}({args})\n  is_error={res.is_error}\n  {text[:220]}")


anyio.run(main)
