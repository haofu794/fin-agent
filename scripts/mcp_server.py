"""
MCP server that exposes the fin-agent tools to any MCP client (Claude Desktop, Cursor, ...).

The client starts this script as a subprocess and talks to it over stdin/stdout, so nothing is
exposed on the network. The tool names, descriptions and argument schemas are taken from
TOOL_SCHEMAS and every call goes through run_tool(), so the MCP server and our own harness
always share exactly the same tools.

Run by hand (it will wait for a client on stdin):   python scripts/mcp_server.py
Claude Desktop config: see README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from finagent.agent import DATA_SCOPE
from finagent.tools import TOOL_SCHEMAS, run_tool

# stdout carries the MCP protocol, so this process must never print to it.
# Our tools only return values; anything diagnostic should go to stderr.

TOOLS = [
    types.Tool(name=s["function"]["name"],
               description=s["function"]["description"],
               inputSchema=s["function"]["parameters"])
    for s in TOOL_SCHEMAS
]

INSTRUCTIONS = (
    "Tools for answering questions about US public companies from their SEC 10-K filings. "
    "Use get_metric for numbers and calculator for all arithmetic; use search_filings for "
    "qualitative questions and cite the chunk_id of each passage you use. " + DATA_SCOPE
)


async def list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    # run_tool is blocking (SQLite, and an embedding API call for search_filings),
    # so run it in a worker thread to keep the server's event loop responsive.
    text = await anyio.to_thread.run_sync(run_tool, params.name, params.arguments or {})
    body = json.loads(text)
    # Our tools report problems as {"error": ..., "hint": ...}. Mark those as isError so the
    # client's model knows the call failed and can read the hint and retry.
    is_error = isinstance(body, dict) and "error" in body
    return types.CallToolResult(content=[types.TextContent(text=text)], isError=is_error)


server = Server("fin-agent", version="0.1.0", instructions=INSTRUCTIONS,
                on_list_tools=list_tools, on_call_tool=call_tool)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
