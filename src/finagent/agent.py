"""
M4: hand-written agent harness.

The core is a single loop:
    send the conversation history to the model
    -> does the model want to call a tool?
         yes -> run the tool, append the result to the history, loop again
         no  -> its reply is the final answer; stop
The harness also enforces a MAX_STEPS limit, feeds tool errors back to the model, and records a trace.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .config import PROJECT_ROOT, get_settings
from .metrics import YEARS
from .tools import TOOL_SCHEMAS, run_tool

FILING_YEARS = (2024, 2025)  # keep in sync with scripts/build_filings.py

MAX_STEPS = 8

# The data scope is generated from code constants rather than hard-coded in the prompt,
# so the prompt stays correct when years are added.
DATA_SCOPE = (f"Financial metrics are available for fiscal years {YEARS.start}-{YEARS.stop - 1}. "
              f"10-K text is indexed only for fiscal years {FILING_YEARS[0]} and {FILING_YEARS[-1]}. "
              f"'Latest' or 'most recent fiscal year' means FY{YEARS.stop - 1}.")

SYSTEM_PROMPT = f"""You are a financial analyst assistant answering questions about US public companies \
using data from their SEC 10-K filings.

Rules:
1. Never state a financial number from memory. Every number must come from the get_metric tool.
2. Never do arithmetic in your head. Use the calculator tool for every calculation \
(growth rates, ratios, differences, margins).
3. If a metric is "not_reported", try to derive it from other metrics \
(e.g. GrossProfit = Revenue - CostOfRevenue) and say clearly that it was derived.
4. If the data needed is not available, say so. Do not guess.
4a. get_metric only covers its listed metrics. If the question asks about something that is NOT one of them \
(e.g. membership fees, segment revenue, employee count, store count), do not substitute a similar metric \
such as Revenue; use search_filings to find it in the 10-K text.
4b. For qualitative questions (risks, strategy, reasons), use search_filings and base the answer ONLY on \
the returned passages. Cite each claim with its chunk_id and section (e.g. "AAPL_2024_1A_001, Item 1A Risk Factors"). \
If the passages do not answer the question, search again with different wording, or say the filing does not cover it.
5. fiscal_year means the year in which the company's fiscal year ends \
(Apple FY2024 ended 2024-09-28; NVIDIA FY2024 ended 2024-01-28). \
When the user writes "FY2023" or "fiscal year 2023", use fiscal_year=2023 exactly. \
Never re-map it to a different fiscal year, even if most of that fiscal year fell in another calendar year. \
Only convert dates when the user describes a period by dates (e.g. "the fiscal year ending January 2025").
5b. Data scope: {DATA_SCOPE}
5c. If the user asks about a company, year or filing that is not available, say so clearly at the START of the answer. \
Never silently substitute a different year or company. You may then offer the closest available data, clearly labeled.
6. When reporting a number, copy the "display" field exactly (e.g. "680.985B USD"). \
Do NOT convert units yourself (e.g. do not convert B to 亿 or millions). \
Round percentages to 2 decimal places.
7. Final answer: state the result first, then the key numbers used, then sources. \
Sources for numbers: company, metric, fiscal year, 10-K accession number. \
Sources for text passages: the chunk_id of every passage you used (e.g. AAPL_2024_1A_001) plus its section. \
Answer in the same language as the question."""


# Explicit fiscal-year mentions: "FY2023", "fiscal year 2023", "fiscal 2023", "2023 财年" (Chinese for "fiscal year 2023")
_FY_PATTERNS = [r"\bFY\s?(20\d\d)\b", r"\bfiscal(?:\s+year)?\s+(20\d\d)\b", r"(20\d\d)\s*财年"]


def explicit_fiscal_years(question: str) -> set[int]:
    """
    The guardrail is enabled only when the question contains an explicit fiscal-year mention
    (FY2023 / fiscal year 2023 / 2023 财年). Once enabled, every year in the question counts as
    a year the user asked about.
    Examples:
      "How much did Apple's FY2024 revenue grow vs 2023?" -> {2023, 2024}
          (recognising only {2024} would raise a false alarm when the agent looks up 2023)
      "Which year in 2020-2025 had the highest revenue?" -> no explicit fiscal-year mention -> disabled
          (otherwise looking up 2021-2024 would raise false alarms)
    """
    if not any(re.search(p, question, re.IGNORECASE) for p in _FY_PATTERNS):
        return set()
    return {int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", question)}


def year_guard(question_years: set[int], arguments: str) -> str | None:
    """
    Guardrail: if the user explicitly asked about FY2023 but a tool call used fiscal_year=2024,
    return a warning that is attached to the tool result.
    Only active when the user wrote an explicit fiscal year; range questions such as
    "between 2020 and 2025" are not affected.
    """
    if not question_years:
        return None
    try:
        fy = json.loads(arguments).get("fiscal_year")
    except (json.JSONDecodeError, AttributeError):
        return None
    # Also allow the previous year: growth / year-over-year questions must look it up
    # (e.g. "NVIDIA FY2025 revenue growth" needs FY2024).
    allowed = question_years | {y - 1 for y in question_years}
    if fy is None or int(fy) in allowed:
        return None
    asked = ", ".join(f"FY{y}" for y in sorted(question_years))
    return (f"HARNESS WARNING: the user explicitly asked about {asked}, but this call used fiscal_year={fy}. "
            f"These results are NOT about {asked}. If {asked} data is unavailable, you must say so at the start "
            f"of your answer, and label any FY{fy} information explicitly as FY{fy}.")


@dataclass
class AgentResult:
    answer: str
    steps: int
    trace: list = field(default_factory=list)
    stopped_reason: str = "final_answer"   # or "max_steps"


class Agent:
    def __init__(self, model: str | None = None, max_steps: int = MAX_STEPS,
                 tool_names: list[str] | None = None, log: bool = True, system_prompt: str | None = None):
        s = get_settings()
        self.client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
        self.model = model or s.llm_model
        self.max_steps = max_steps
        self.log = log
        self.system_prompt = system_prompt or SYSTEM_PROMPT   # can be swapped out for ablations
        # tool_names=None -> all tools; ablations can remove one (e.g. no calculator)
        self.tools = [t for t in TOOL_SCHEMAS if tool_names is None or t["function"]["name"] in tool_names]

    def _call_model(self, messages, allow_tools: bool = True):
        kwargs = dict(model=self.model, messages=messages)
        if allow_tools:
            kwargs["tools"] = self.tools
        return self.client.chat.completions.create(**kwargs).choices[0].message

    def run(self, question: str, verbose: bool = True,
            on_event: Callable[[dict], None] | None = None) -> AgentResult:
        """
        Answer one question. `on_event`, if given, is called with every trace event as it happens
        (plus a "model_call" event before each LLM call), so a caller such as the web server
        can stream progress. It never changes the agent's behaviour.
        """
        # Conversation history: the model is stateless, so the full history is resent every turn.
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}]
        trace = []
        question_years = explicit_fiscal_years(question)

        def notify(event: dict) -> None:
            if on_event is not None:
                on_event(event)

        def record(event: dict) -> None:
            trace.append(event)
            notify(event)

        for step in range(1, self.max_steps + 1):
            notify({"step": step, "type": "model_call"})   # progress only; not stored in the trace
            t0 = time.time()
            msg = self._call_model(messages)
            elapsed = round(time.time() - t0, 2)

            # ── Case A: no tool calls -> this is the final answer ──
            if not msg.tool_calls:
                record({"step": step, "type": "final_answer", "content": msg.content, "seconds": elapsed})
                return self._finish(question, msg.content, step, trace, "final_answer")

            # ── Case B: the model wants to call tools ──
            # Append the model's tool-call message to the history first;
            # otherwise next turn it would see results without knowing which calls produced them.
            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls],
            })
            # The model may issue several calls at once (e.g. 2023 and 2024 together); run each one.
            for tc in msg.tool_calls:
                result = run_tool(tc.function.name, tc.function.arguments)   # errors come back as hints, never raise
                warning = year_guard(question_years, tc.function.arguments)
                if warning:
                    result = json.dumps({"harness_warning": warning, "tool_result": json.loads(result)},
                                        ensure_ascii=False)
                    record({"step": step, "type": "guardrail", "warning": warning})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                record({"step": step, "type": "tool_call", "tool": tc.function.name,
                              "arguments": tc.function.arguments, "result": result, "seconds": elapsed})
                if verbose:
                    print(f"  [step {step}] {tc.function.name}({tc.function.arguments})")
                    print(f"           → {result[:150]}")

        # ── MAX_STEPS reached: withhold tools and force an answer from what has been gathered ──
        messages.append({"role": "user", "content":
                         "Step limit reached. Answer now using only the information above, "
                         "and say if anything is missing."})
        msg = self._call_model(messages, allow_tools=False)
        record({"step": self.max_steps + 1, "type": "forced_answer", "content": msg.content})
        return self._finish(question, msg.content, self.max_steps, trace, "max_steps")

    def _finish(self, question, answer, steps, trace, reason) -> AgentResult:
        if not self.log:
            return AgentResult(answer=answer, steps=steps, trace=trace, stopped_reason=reason)
        # Save a trace of every run to logs/ for debugging and evaluation (M5).
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / f"trace_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        path.write_text(json.dumps({"question": question, "model": self.model, "steps": steps,
                                    "stopped_reason": reason, "answer": answer, "trace": trace},
                                   ensure_ascii=False, indent=2))
        return AgentResult(answer=answer, steps=steps, trace=trace, stopped_reason=reason)
