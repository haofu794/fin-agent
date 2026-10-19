"""
M1: download data from SEC EDGAR and cache it in data/raw/.

Why cache: SEC rate-limits requests (at most 10 per second). After the first download, files are read
from disk, so re-running after a parsing change takes seconds and puts no load on SEC.
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
    headers = {"User-Agent": get_settings().sec_user_agent}  # SEC rejects requests without it
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()  # fail on 403 / 404 instead of silently caching an error page
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)
    time.sleep(0.2)  # polite rate limit: at most 5 requests per second
    return resp.json()


def _get_text(url: str, cache_file: Path) -> str:
    """Same as _get_json, but caches raw HTML text."""
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
    """SEC's official ticker -> CIK map, e.g. {'AAPL': 320193, ...}"""
    raw = get_settings().data_dir / "raw"
    data = _get_json("https://www.sec.gov/files/company_tickers.json", raw / "company_tickers.json")
    return {v["ticker"]: v["cik_str"] for v in data.values()}


def company_facts(cik: int) -> dict:
    raw = get_settings().data_dir / "raw"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"  # CIK zero-padded to 10 digits
    return _get_json(url, raw / f"companyfacts_{cik}.json")
