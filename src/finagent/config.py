"""
Central configuration.

Why it is designed this way:
1. API keys live only in .env, never in code, so the code can be published on GitHub safely.
2. Every module reads settings through get_settings(), so switching models (OpenAI, DeepSeek, local, ...)
   only requires editing .env.
3. A frozen dataclass prevents settings from being modified by accident after loading.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: this file is src/finagent/config.py, so two levels up is fin-agent/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    sec_user_agent: str
    embed_model: str
    data_dir: Path
    db_path: Path


def get_settings() -> Settings:
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_model=os.getenv("LLM_MODEL", ""),
        sec_user_agent=os.getenv("SEC_USER_AGENT", ""),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        data_dir=data_dir,
        db_path=data_dir / "finance.db",
    )
