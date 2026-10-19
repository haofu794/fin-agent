"""
M7: web front end for the agent.

    Browser page  --question-->  FastAPI  -->  Agent.run()   (the same harness used by scripts/ask.py)
                  <--events----

Endpoints
  GET  /                  the single-page UI (web/static/index.html)
  GET  /api/health        model name and whether the database / index exist
  POST /api/ask           run the agent and return the full result as one JSON object
  POST /api/ask/stream    run the agent and stream progress as Server-Sent Events (one event per step)

Run locally (binds to 127.0.0.1, so only this computer can reach it):
  python web/app.py            then open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from finagent.agent import Agent
from finagent.config import get_settings

STATIC = Path(__file__).resolve().parent / "static"
MAX_CONCURRENT = 2        # every question costs API money and ~10 s; cap how many run at once
MAX_QUESTION_CHARS = 500  # a question longer than this is almost certainly not a real question

app = FastAPI(title="fin-agent", docs_url="/api/docs")
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_CHUNK_ID = re.compile(r"[A-Z]{1,5}_20\d\d_(?:1A|1|7|FULL)_\d{3}")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


# ──────────────── helpers ────────────────

def _acquire_slot() -> None:
    """Reject instead of queueing when busy, so a burst of requests cannot pile up API costs."""
    if not _slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The agent is busy with other questions. Try again in a few seconds.")


def _public_event(event: dict) -> dict:
    """Turn a harness trace event into what the browser needs (tool results parsed from JSON strings)."""
    out = {k: v for k, v in event.items() if k != "result"}
    if "result" in event:
        try:
            out["result"] = json.loads(event["result"])
        except (TypeError, json.JSONDecodeError):
            out["result"] = event["result"]
    return out


def _final_payload(result, started: float) -> dict:
    """Final answer plus the passages it cites, so the page can show a citation without another request."""
    passages = {}
    for t in result.trace:
        if t.get("tool") == "search_filings":
            body = json.loads(t["result"])
            body = body.get("tool_result", body)          # unwrap a guardrail warning, if any
            for r in body.get("results", []):
                passages[r["chunk_id"]] = {"section": r["section"], "text": r["text"]}
    cited = [c for c in dict.fromkeys(_CHUNK_ID.findall(result.answer or "")) if c in passages]
    return {
        "answer": result.answer,
        "steps": result.steps,
        "stopped_reason": result.stopped_reason,
        "seconds": round(time.time() - started, 1),
        "tool_calls": sum(t["type"] == "tool_call" for t in result.trace),
        "citations": {c: passages[c] for c in cited},
    }


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event: an `event:` line, a `data:` line, then a blank line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ──────────────── routes ────────────────

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    s = get_settings()
    return {"model": s.llm_model,
            "database": s.db_path.exists(),
            "index": (s.data_dir / "index" / "chunks.json").exists()}


@app.post("/api/ask")
async def ask(req: AskRequest):
    _acquire_slot()
    try:
        started = time.time()
        # Agent.run is blocking (network calls to the LLM), so run it in a worker thread
        # to keep the server responsive for other requests.
        result = await run_in_threadpool(Agent().run, req.question, False)
        return _final_payload(result, started)
    finally:
        _slots.release()


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    _acquire_slot()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    started = time.time()

    def worker():
        # Runs in a separate thread. Each harness event is handed to the event loop thread-safely.
        try:
            result = Agent().run(req.question, verbose=False,
                                 on_event=lambda e: loop.call_soon_threadsafe(queue.put_nowait, ("step", e)))
            loop.call_soon_threadsafe(queue.put_nowait, ("final", _final_payload(result, started)))
        except Exception as e:   # report failures to the page instead of silently closing the stream
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            _slots.release()
            loop.call_soon_threadsafe(queue.put_nowait, ("done", {}))

    threading.Thread(target=worker, daemon=True).start()

    async def events():
        while True:
            kind, data = await queue.get()
            if kind == "step":
                yield _sse(data["type"], _public_event(data))
            else:
                yield _sse(kind, data)
            if kind == "done":
                break

    # Note: if the browser disconnects, the worker thread still finishes its current question;
    # the harness has no cancellation point. The step limit bounds how long that can take.
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
