"""
M1 核心：从一家公司的 companyfacts 里，为每个 (指标, 财年) 挑出唯一正确的那一个数字。

挑选规则（对应学习笔记第 1–4 块）：
1. 只要 10-K（年报）里的记录。
2. 录像类：end − start 在 350–380 天之间（排除季度数据）。
   照片类：end 必须是这家公司某个财年的最后一天（排除其他日期的余额）。
3. 财年 = end 日期所在的年份（不用 fy 字段，因为 fy 是"报告"的年份）。
4. 候选标签按顺序试：某一年第 1 个标签有数据就用它，没有再试第 2 个。
5. 同一年同一标签有多条记录（原值 + 后来的重述）→ 取 filed 最新的。
"""
from __future__ import annotations

from datetime import date

from .metrics import METRICS


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _annual_rows(gaap: dict, tag: str, unit: str, kind: str, fy_ends: set[str] | None):
    """某个标签的所有"年度"记录（规则 1、2）。"""
    for r in gaap.get(tag, {}).get("units", {}).get(unit, []):
        if r.get("form") not in ("10-K", "10-K/A"):
            continue
        if kind == "duration":
            if not r.get("start") or not 350 <= _days(r["start"], r["end"]) <= 380:
                continue
        else:  # instant
            if fy_ends is not None and r["end"] not in fy_ends:
                continue
        yield r


def fiscal_year_ends(gaap: dict) -> set[str]:
    """用营收（录像）的 end 日期，找出这家公司每个财年的最后一天。照片类数字只认这些日期。"""
    tags, kind, unit = METRICS["Revenue"]
    return {r["end"] for tag in tags for r in _annual_rows(gaap, tag, unit, kind, None)}


def select_values(ticker: str, facts: dict, years) -> list[dict]:
    gaap = facts["facts"].get("us-gaap", {})
    fy_ends = fiscal_year_ends(gaap)
    out = []
    for metric, (tags, kind, unit) in METRICS.items():
        for year in years:
            for tag in tags:  # 规则 4：按优先级试
                rows = [r for r in _annual_rows(gaap, tag, unit, kind, fy_ends)
                        if int(r["end"][:4]) == year]                    # 规则 3
                if not rows:
                    continue
                best = max(rows, key=lambda r: r["filed"])               # 规则 5
                out.append(dict(
                    ticker=ticker, metric=metric, fiscal_year=year,
                    value=best["val"], unit=unit,
                    period_start=best.get("start"), period_end=best["end"],
                    tag=tag, accn=best["accn"], form=best["form"], filed=best["filed"],
                ))
                break  # 找到了，不再试后面的候选
    return out
