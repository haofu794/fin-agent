"""
列出你的 API key 能用的模型 ID，方便填写 .env 里的 LLM_MODEL。
用法：python scripts/list_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI
from finagent.config import get_settings

s = get_settings()
client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
ids = sorted(m.id for m in client.models.list())
print(f"共 {len(ids)} 个模型可用：")
for mid in ids:
    print("  ", mid)
