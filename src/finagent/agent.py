"""
M4：手写的 Agent harness。

核心就是一个循环：
    把对话历史发给模型
    → 模型要用工具？ 是 → 执行工具，把结果加进历史，回到开头
                     否 → 它给出的就是最终答案，结束
同时负责：最多 MAX_STEPS 步、错误回传、记录 trace。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .config import PROJECT_ROOT, get_settings
from .metrics import YEARS
from .tools import TOOL_SCHEMAS, run_tool

FILING_YEARS = (2024, 2025)  # 与 scripts/build_filings.py 保持一致

MAX_STEPS = 8

# 数据范围从代码里的常量生成，而不是手写在 prompt 里：以后加年份，prompt 自动跟着变
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


# 用户明确写出的财年："FY2023" "fiscal year 2023" "fiscal 2023" "2023 财年"
_FY_PATTERNS = [r"\bFY\s?(20\d\d)\b", r"\bfiscal(?:\s+year)?\s+(20\d\d)\b", r"(20\d\d)\s*财年"]


def explicit_fiscal_years(question: str) -> set[int]:
    """
    只有问题里出现了明确的财年写法（FY2023 / fiscal year 2023 / 2023 财年），guardrail 才启用；
    启用后，问题里出现的所有年份都算"用户提到的年份"。
    例："Apple 2024 财年营收比 2023 年增长了多少" → {2023, 2024}（只认 {2024} 的话，查 2023 时会误报）
         "2020–2025 年间哪一年营收最高"      → 没有明确财年写法 → 不启用（否则查 2021–2024 时会误报）
    """
    if not any(re.search(p, question, re.IGNORECASE) for p in _FY_PATTERNS):
        return set()
    return {int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", question)}


def year_guard(question_years: set[int], arguments: str) -> str | None:
    """
    Guardrail：用户明确问了 FY2023，工具却用了 fiscal_year=2024 → 返回一段警告，附在工具结果后面。
    只在用户"明确写出财年"时才检查；"2020–2025 年间"这类范围问题不受影响。
    """
    if not question_years:
        return None
    try:
        fy = json.loads(arguments).get("fiscal_year")
    except (json.JSONDecodeError, AttributeError):
        return None
    # 允许"前一年"：增长率、同比这类问题必须查上一年做比较（"NVIDIA FY2025 营收同比增长" 要查 FY2024）
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
    stopped_reason: str = "final_answer"   # 或 "max_steps"


class Agent:
    def __init__(self, model: str | None = None, max_steps: int = MAX_STEPS,
                 tool_names: list[str] | None = None, log: bool = True, system_prompt: str | None = None):
        s = get_settings()
        self.client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
        self.model = model or s.llm_model
        self.max_steps = max_steps
        self.log = log
        self.system_prompt = system_prompt or SYSTEM_PROMPT   # ablation 时可以换一份 prompt
        # tool_names=None → 全部工具；做 ablation 时可以去掉某个工具（例如不给 calculator）
        self.tools = [t for t in TOOL_SCHEMAS if tool_names is None or t["function"]["name"] in tool_names]

    def _call_model(self, messages, allow_tools: bool = True):
        kwargs = dict(model=self.model, messages=messages)
        if allow_tools:
            kwargs["tools"] = self.tools
        return self.client.chat.completions.create(**kwargs).choices[0].message

    def run(self, question: str, verbose: bool = True) -> AgentResult:
        # 对话历史：模型没有记忆，每一轮都要把完整的历史重新发给它
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}]
        trace = []
        question_years = explicit_fiscal_years(question)

        for step in range(1, self.max_steps + 1):
            t0 = time.time()
            msg = self._call_model(messages)
            elapsed = round(time.time() - t0, 2)

            # ── 情况 A：模型没有要用工具 → 这就是最终答案 ──
            if not msg.tool_calls:
                trace.append({"step": step, "type": "final_answer", "content": msg.content, "seconds": elapsed})
                return self._finish(question, msg.content, step, trace, "final_answer")

            # ── 情况 B：模型要用工具 ──
            # 先把模型的"纸条"原样加进历史（否则下一轮它不知道自己调用过什么）
            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls],
            })
            # 模型可能一次写好几张纸条（例如同时查 2023 和 2024），逐张执行
            for tc in msg.tool_calls:
                result = run_tool(tc.function.name, tc.function.arguments)   # 出错也只是返回提示，不会崩
                warning = year_guard(question_years, tc.function.arguments)
                if warning:
                    result = json.dumps({"harness_warning": warning, "tool_result": json.loads(result)},
                                        ensure_ascii=False)
                    trace.append({"step": step, "type": "guardrail", "warning": warning})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                trace.append({"step": step, "type": "tool_call", "tool": tc.function.name,
                              "arguments": tc.function.arguments, "result": result, "seconds": elapsed})
                if verbose:
                    print(f"  [step {step}] {tc.function.name}({tc.function.arguments})")
                    print(f"           → {result[:150]}")

        # ── 走完 MAX_STEPS 还没结束：不再给工具，强制它根据已有信息作答 ──
        messages.append({"role": "user", "content":
                         "Step limit reached. Answer now using only the information above, "
                         "and say if anything is missing."})
        msg = self._call_model(messages, allow_tools=False)
        trace.append({"step": self.max_steps + 1, "type": "forced_answer", "content": msg.content})
        return self._finish(question, msg.content, self.max_steps, trace, "max_steps")

    def _finish(self, question, answer, steps, trace, reason) -> AgentResult:
        if not self.log:
            return AgentResult(answer=answer, steps=steps, trace=trace, stopped_reason=reason)
        # 每次运行都存一份 trace 到 logs/，方便调试和 M5 评测
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / f"trace_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        path.write_text(json.dumps({"question": question, "model": self.model, "steps": steps,
                                    "stopped_reason": reason, "answer": answer, "trace": trace},
                                   ensure_ascii=False, indent=2))
        return AgentResult(answer=answer, steps=steps, trace=trace, stopped_reason=reason)
