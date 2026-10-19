"""
M1：从 SEC 下载数据，并缓存到 data/raw/。

为什么要缓存：SEC 每个请求都要排队、限速（每秒最多 10 次）。
第一次下载后存成本地文件，以后改解析代码重跑时直接读文件，几秒就跑完，也不会给 SEC 添负担。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from .config import get_settings


def _get_json(url: str, cache_file: Path) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    headers = {"User-Agent": get_settings().sec_user_agent}  # 没有这个，SEC 会直接拒绝
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()  # 403 / 404 等错误时直接报错，而不是悄悄存下一个错误页面
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)
    time.sleep(0.2)  # 礼貌限速：最多每秒 5 次
    return resp.json()


def _get_text(url: str, cache_file: Path) -> str:
    """和 _get_json 一样，只是存的是 HTML 文字。"""
    if cache_file.exists():
        return cache_file.read_text()
    headers = {"User-Agent": get_settings().sec_user_agent}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)
    time.sleep(0.2)
    return resp.text


def ticker_to_cik() -> dict[str, int]:
    """SEC 官方的 ticker → CIK 对照表，例如 {'AAPL': 320193, ...}"""
    raw = get_settings().data_dir / "raw"
    data = _get_json("https://www.sec.gov/files/company_tickers.json", raw / "company_tickers.json")
    return {v["ticker"]: v["cik_str"] for v in data.values()}


def company_facts(cik: int) -> dict:
    raw = get_settings().data_dir / "raw"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"  # CIK 补零到 10 位
    return _get_json(url, raw / f"companyfacts_{cik}.json")
