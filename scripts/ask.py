"""Ask the agent a question. Usage: python scripts/ask.py "How much did Apple's FY2024 revenue grow vs FY2023?" """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finagent.agent import Agent

question = " ".join(sys.argv[1:]) or "Apple 2024 财年营收比 2023 年增长了多少？"
print(f"问题：{question}\n")
result = Agent().run(question)
print(f"\n用了 {result.steps} 步（{result.stopped_reason}）\n\n{result.answer}")
