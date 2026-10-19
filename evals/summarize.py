"""
Summarise repeated runs into a comparison table.
Usage: python evals/summarize.py final abl_bm25 abl_vector abl_nocalc
Each name matches evals/results/<name>_1.json, _2.json, _3.json, ...
"""
import json
import statistics
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
GROUPS = [("数字题 (lookup+calc+hard_num, 40)", ("lookup", "calc", "hard_num")),
          ("其中 calc+hard_num (25)", ("calc", "hard_num")),
          ("原文题 (text+hard_text, 15)", ("text", "hard_text")),
          ("不可回答 (5)", ("unanswerable",))]


def load(prefix):
    return [json.loads(p.read_text()) for p in sorted(RES.glob(f"{prefix}_[0-9].json"))]


def pct(xs):
    m = statistics.mean(xs)
    return f"{m:5.1%}" + (f" ±{statistics.stdev(xs):.1%}" if len(xs) > 1 else "")


rows = []
for prefix in sys.argv[1:]:
    runs = load(prefix)
    if not runs:
        print(f"(没有找到 {prefix}_*.json)"); continue
    row = {"配置": prefix, "次数": len(runs)}
    for label, types in GROUPS:
        row[label] = pct([sum(r["correct"] for r in run if r["type"] in types) /
                          max(1, sum(r["type"] in types for r in run)) for run in runs])
    text = lambda run: [r for r in run if r["type"] in ("text", "hard_text")]
    row["检索命中"] = pct([sum(r["retrieval_hit"] for r in text(run)) / len(text(run)) for run in runs])
    row["裁判均分"] = f"{statistics.mean(statistics.mean(r['judge'].get('score', 0) for r in text(run)) for run in runs):.2f}"
    row["平均步数"] = f"{statistics.mean(statistics.mean(r.get('steps', 0) for r in run) for run in runs):.2f}"
    row["总正确"] = pct([sum(r["correct"] for r in run) / len(run) for run in runs])
    rows.append(row)

cols = list(rows[0])
for c in cols:
    print(f"{c:34}" + "".join(f"{str(r[c]):>18}" for r in rows))
