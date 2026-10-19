"""
Retrieval ablation (no agent involved, so results are deterministic):
replay the search queries the agent actually wrote during evaluation under different retrieval modes
and conditions, and report hit@k.
hit@k = fraction of queries where at least one of the top-k chunks contains the key phrase.
Usage: python evals/retrieval_ablation.py final
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finagent.rag import get_index

prefix = sys.argv[1] if len(sys.argv) > 1 else "final"
items = {json.loads(l)["id"]: json.loads(l) for n in ("questions", "hard")
         for l in open(ROOT / "evals" / f"{n}.jsonl")}

# Collect (query, ticker, year, key phrases), using the agent's first search query for each question
cases = {}
for p in sorted((ROOT / "evals" / "results").glob(f"{prefix}_[0-9].json")):
    for r in json.loads(p.read_text()):
        if r["type"] not in ("text", "hard_text"):
            continue
        for t in r["trace"]:
            if t.get("tool") == "search_filings":
                a = json.loads(t["arguments"])
                cases.setdefault((r["id"], a["query"]), (a.get("ticker"), a.get("fiscal_year"), items[r["id"]]["gold_phrases"]))
                break
print(f"{len(cases)} 条不同的查询（来自 {len({k[0] for k in cases})} 道题）\n")

idx = get_index()
def hit(query, ticker, fy, phrases, mode, k):
    res = idx.search(query, ticker=ticker, fiscal_year=fy, top_k=k, mode=mode)
    return any(p.lower() in r["text"].lower() for r in res for p in phrases)

print(f"{'条件':26}{'k':>3}{'bm25':>9}{'vector':>9}{'hybrid':>9}")
for cond, use_filter in (("按公司+年份筛选（实际用法）", True), ("不筛选（在 3,999 块里找）", False)):
    for k in (1, 3, 5):
        row = []
        for mode in ("bm25", "vector", "hybrid"):
            hits = [hit(q, tk if use_filter else None, fy if use_filter else None, ph, mode, k)
                    for (qid, q), (tk, fy, ph) in cases.items()]
            row.append(sum(hits) / len(hits))
        print(f"{cond:26}{k:>3}" + "".join(f"{x:>9.0%}" for x in row))
