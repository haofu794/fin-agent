"""
M5 评测：跑 40 道题 → 自动评分 → 输出汇总。

用法：python evals/run_eval.py [--name baseline] [--no-calculator] [--search-mode hybrid|bm25|vector]

评分规则：
- 数字题（lookup / calc）：从答案里提取所有带单位的数字，只要有一个与标准答案相对误差 < 1% 就算对。
  有 entity 的题（"谁更高"）还要求答案里提到正确的公司。
- 原文题（text）：
  ① 检索命中（retrieval hit）：Agent 检索到的段落里，有没有包含关键句的段落；
  ② 引用有效（citation valid）：答案里写的 chunk_id，是否都真的出现在检索结果里（防止编造引用）；
  ③ LLM-as-judge：用更强的模型，对照要点给 1–5 分。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI

from finagent import tools as tools_mod
from finagent.agent import SYSTEM_PROMPT, Agent

# 公平的 ablation：拿掉 calculator 的同时，也要拿掉"必须用 calculator"这条规则，否则测的是"规则与工具矛盾"
_RULE2 = SYSTEM_PROMPT[SYSTEM_PROMPT.index("2. Never do arithmetic"):SYSTEM_PROMPT.index("3. If a metric")]
NO_CALC_PROMPT = SYSTEM_PROMPT.replace(_RULE2, "2. Do all arithmetic yourself, carefully and step by step.\n")
from finagent.config import get_settings
from finagent.rag import get_index
from finagent.tools import get_metric

JUDGE_MODEL = "gpt-5.4"   # 裁判用比 Agent 更强的模型，减少"自己给自己打分"的偏差


# ──────────────── 标准答案：直接从数据库算 ────────────────
def M(ticker, metric, year):
    r = get_metric(ticker, metric, year)
    if "error" in r:
        raise ValueError(f"gold needs {ticker} {metric} {year}: {r}")
    return r["value"]


# ──────────────── 从答案里提取数字 ────────────────
_SCALE = {"T": 1e12, "trillion": 1e12, "万亿": 1e12, "B": 1e9, "billion": 1e9, "亿": 1e8,
          "M": 1e6, "million": 1e6, "百万": 1e6, "万": 1e4}
# "个百分点" / "percentage points" / "pp" 也是百分比（两个百分比之差），要和 % 一样处理
_NUM = re.compile(r"([-−–]?\$?\s?\d[\d,]*(?:\.\d+)?)\s*(万亿|trillion|billion|million|百万|亿|万|个百分点|百分点|percentage points?|pp\b|T|B|M|%)?", re.I)
_PCT = {"%", "个百分点", "百分点", "pp", "percentage point", "percentage points"}


def extract_numbers(text: str) -> list[tuple[float, str, int]]:
    """返回 [(数值, 类型, 小数位数), ...]。小数位数用来判断"四舍五入后是否一致"。"""
    out = []
    for m in _NUM.finditer(text.replace("**", "")):
        raw, suf = m.group(1), (m.group(2) or "")
        try:
            v = float(raw.replace("$", "").replace(",", "").replace(" ", "").replace("−", "-").replace("–", "-"))
        except ValueError:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        key = suf if suf in _SCALE else suf.lower() if suf.lower() in _SCALE else suf
        if key in _SCALE:
            out.append((v * _SCALE[key], "abs", dec))
        elif suf.lower() in _PCT:
            out.append((v, "%", dec))
        else:
            out.append((v, "plain", dec))
    return out


def number_match(answer: str, gold: float, unit: str, tol: float = 0.01, use_abs: bool = False) -> bool:
    for v, kind, dec in extract_numbers(answer):
        if unit == "USD" and kind not in ("abs",):
            continue
        if unit == "%" and kind != "%":
            continue
        if unit == "count" and kind == "%":
            continue
        # 金额比较绝对值：答案常写成"净亏损 2.2B"而不带负号
        a, g = (abs(v), abs(gold)) if (unit == "USD" or use_abs) else (v, gold)
        if g == 0:
            if abs(a) < 1e-9:
                return True
        elif abs(a - g) / abs(g) < tol:
            return True
        # 比率、百分比：模型按规则保留了 2 位小数（0.1578 → 0.16），四舍五入后一致也算对
        elif unit in ("%", "ratio") and kind != "abs" and round(g, dec) == round(a, dec) and dec >= 1:
            return True
    return False


# ──────────────── 原文题：检索命中、引用有效、LLM 裁判 ────────────────
def retrieved_chunks(trace) -> dict[str, str]:
    got = {}
    for t in trace:
        if t.get("tool") == "search_filings":
            for r in json.loads(t["result"]).get("results", []):
                got[r["chunk_id"]] = r["text"]
    return got


def judge(q, key_points, answer) -> dict:
    s = get_settings()
    client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    prompt = f"""You are grading an answer to a question about a company's 10-K filing.

Question: {q}

Reference key points (from the filing):
{chr(10).join('- ' + k for k in key_points)}

Answer to grade:
{answer}

Score 1-5:
5 = covers all key points accurately, nothing wrong
4 = covers most key points, nothing wrong
3 = covers some key points, or has minor inaccuracies
2 = mostly misses the key points
1 = wrong, or says the information is unavailable
Return JSON only: {{"score": <int>, "missing": [<key points not covered>], "errors": [<factual errors>]}}"""
    r = client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                                       response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)


def judge_unanswerable(q, why, answer) -> dict:
    s = get_settings()
    client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    prompt = f"""A financial assistant was asked a question that CANNOT be answered from its data.
Question: {q}
Why it cannot be answered: {why}

Assistant's answer:
{answer}

PASS if the assistant clearly says the requested information is not available (it may suggest alternatives,
as long as it does not present a made-up or substitute value as the answer to the question).
FAIL if it states a specific value as the answer, or answers using data from a different company/year
without clearly saying so.
Return JSON only: {{"pass": true/false, "reason": "<one sentence>"}}"""
    r = client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                                       response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)


# ──────────────── 跑一道题 ────────────────
def run_one(item, tool_names, system_prompt=None):
    t0 = time.time()
    try:
        res = Agent(tool_names=tool_names, log=False, system_prompt=system_prompt).run(item["q"], verbose=False)
    except Exception as e:  # 一道题出错不能让整个评测停掉
        return {**item, "error": repr(e), "correct": False}
    rec = {"id": item["id"], "type": item["type"], "q": item["q"], "answer": res.answer,
           "steps": res.steps, "stopped_reason": res.stopped_reason, "seconds": round(time.time() - t0, 1),
           "tools_used": [t["tool"] for t in res.trace if t["type"] == "tool_call"], "trace": res.trace}
    if item["type"] in ("lookup", "calc", "hard_num"):
        gold = eval(item["gold"], {"M": M, "abs": abs, "sum": sum, "range": range})  # gold 是我们自己写的，可以 eval
        ok = number_match(res.answer, gold, item["unit"], use_abs=item.get("abs", False))
        if ok and item.get("entity"):
            ok = any(e.lower() in res.answer.lower() for e in item["entity"])
        rec.update(gold=gold, unit=item["unit"], correct=ok)
    elif item["type"] == "unanswerable":
        j = judge_unanswerable(item["q"], item["why"], res.answer)
        rec.update(judge=j, correct=bool(j.get("pass")))
    else:
        got = retrieved_chunks(res.trace)
        hit = any(p.lower() in txt.lower() for txt in got.values() for p in item["gold_phrases"])
        cited = set(re.findall(r"[A-Z]{1,5}_20\d\d_(?:1A|1|7|FULL)_\d{3}", res.answer))
        j = judge(item["q"], item["key_points"], res.answer)
        # 引用分两项看：① 有没有写 chunk_id（是否遵守格式）；② 写了的是否都真的检索到过（是否编造）
        rec.update(retrieval_hit=hit, cited=sorted(cited), has_citation=bool(cited),
                   fabricated=sorted(cited - set(got)), citation_valid=bool(cited) and cited <= set(got), judge=j,
                   correct=j.get("score", 0) >= 4)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--no-calculator", action="store_true", help="拿掉 calculator，但保留'必须用 calculator'的规则（有矛盾）")
    ap.add_argument("--no-calculator-fair", action="store_true", help="拿掉 calculator，同时把规则 2 改成'自己计算'")
    ap.add_argument("--search-mode", default="hybrid", choices=["hybrid", "bm25", "vector"])
    ap.add_argument("--only", default=None, help="只跑某一类：lookup / calc / text")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", default="questions", help="questions（基础 40 题）/ hard（难题 20 题）/ all")
    args = ap.parse_args()

    tool_names, system_prompt = None, None
    if args.no_calculator or args.no_calculator_fair:
        tool_names = ["get_metric", "search_filings", "list_companies"]
    if args.no_calculator_fair:
        system_prompt = NO_CALC_PROMPT
    tools_mod.SEARCH_MODE = args.search_mode
    get_index()   # 先加载索引，避免 8 个线程同时加载

    sets = ["questions", "hard"] if args.set == "all" else [args.set]
    items = [json.loads(l) for name in sets for l in open(ROOT / "evals" / f"{name}.jsonl")]
    if args.only:
        items = [i for i in items if i["type"] == args.only]
    with ThreadPoolExecutor(args.workers) as ex:   # 8 道题同时跑：40 题从约 7 分钟缩短到约 1 分钟
        results = list(ex.map(lambda it: run_one(it, tool_names, system_prompt), items))

    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print(f"\n==== {args.name} ====")
    for typ in ("lookup", "calc", "text", "hard_num", "hard_text", "unanswerable"):
        rs = [r for r in results if r["type"] == typ]
        if not rs:
            continue
        acc = sum(r["correct"] for r in rs)
        line = f"{typ:6} 正确 {acc}/{len(rs)} ({acc / len(rs):.0%})  平均步数 {sum(r.get('steps', 0) for r in rs) / len(rs):.1f}"
        if typ in ("text", "hard_text"):
            line += (f"  检索命中 {sum(r['retrieval_hit'] for r in rs)}/{len(rs)}"
                     f"  写了chunk_id {sum(r['has_citation'] for r in rs)}/{len(rs)}"
                     f"  编造引用 {sum(bool(r['fabricated']) for r in rs)}/{len(rs)}"
                     f"  裁判平均分 {sum(r['judge'].get('score', 0) for r in rs) / len(rs):.2f}")
        print(line)
    total = sum(r["correct"] for r in results)
    print(f"总计   正确 {total}/{len(results)} ({total / len(results):.0%})")
    for r in results:
        if not r["correct"]:
            extra = f"gold={r.get('gold')}" if "gold" in r else f"judge={r.get('judge')}"
            print(f"  ✗ {r['id']} {r['q'][:40]}  {extra}\n      答：{(r.get('answer') or r.get('error',''))[:300]!r}")


if __name__ == "__main__":
    main()
