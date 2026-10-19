"""
Environment self-check: Python version, dependencies, .env settings, SEC connectivity and LLM API, in one go.
Usage (from the fin-agent directory): python scripts/check_setup.py
"""
import sys
from pathlib import Path

# make the package under src/ importable from scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OK, FAIL = "✅", "❌"


def check(name, fn):
    try:
        detail = fn()
        print(f"{OK} {name}" + (f"：{detail}" if detail else ""))
        return True
    except Exception as e:  # catching everything is fine in a self-check: we want to report, not abort
        print(f"{FAIL} {name}：{e}")
        return False


def py_version():
    v = sys.version_info
    assert v >= (3, 10), f"需要 Python ≥ 3.10，当前 {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


def packages():
    import openai, dotenv, requests, pandas  # noqa: F401
    return "openai / python-dotenv / requests / pandas"


def env_file():
    from finagent.config import get_settings, PROJECT_ROOT
    assert (PROJECT_ROOT / ".env").exists(), "找不到 .env，请先 cp .env.example .env"
    s = get_settings()
    assert s.llm_api_key and not s.llm_api_key.startswith("sk-your"), "LLM_API_KEY 还没填"
    assert s.llm_model and s.llm_model != "your-model-id", "LLM_MODEL 还没填（运行 scripts/list_models.py 查看可用 ID）"
    assert "@" in s.sec_user_agent and "example.com" not in s.sec_user_agent, "SEC_USER_AGENT 需要填真实姓名+邮箱"
    return f"model={s.llm_model}"


def sec_api():
    import requests
    from finagent.config import get_settings
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": get_settings().sec_user_agent},
        timeout=20,
    )
    r.raise_for_status()
    return f"可访问，共 {len(r.json())} 家公司"


def llm_api():
    from openai import OpenAI
    from finagent.config import get_settings
    s = get_settings()
    client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    resp = client.chat.completions.create(
        model=s.llm_model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    )
    return f"模型回复 {resp.choices[0].message.content!r}，用了 {resp.usage.total_tokens} tokens"


if __name__ == "__main__":
    results = [
        check("Python 版本", py_version),
        check("依赖包", packages),
        check(".env 配置", env_file),
        check("SEC EDGAR 网络", sec_api),
        check("LLM API", llm_api),
    ]
    print("\n全部通过，可以进入第 2 步！" if all(results) else "\n有未通过的项目，按提示修复后重新运行。")
