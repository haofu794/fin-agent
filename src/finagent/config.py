"""
集中管理所有配置。

为什么这样设计：
1. API key 只写在 .env 里，不出现在代码中 → 代码可以放心上传到 GitHub。
2. 所有模块都通过 get_settings() 读取配置 → 想换模型（OpenAI / DeepSeek 等）只改 .env，不改代码。
3. 用 frozen dataclass → 配置读取后不能被意外修改。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：本文件在 src/finagent/config.py，往上两级就是 fin-agent/
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
