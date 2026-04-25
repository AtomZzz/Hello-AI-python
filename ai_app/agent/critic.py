from typing import Any, Dict, Optional

from ai_app.parser.json_parser import JsonParser


def build_critic_prompt(user_input, task, result):
    return f"""
你是一个AI系统的质量评估专家，请判断以下结果是否可靠。

评估标准：
1. 是否回答了用户问题
2. 是否基于已知信息（不能胡编）
3. 推理是否合理
4. 是否存在明显错误

输出JSON：
{{
  "decision": "approve 或 reject",
  "reason": "原因",
  "suggestion": "如果reject，给出改进建议"
}}

用户问题：
{user_input}

当前任务：
{task}

执行结果：
{result}
"""


class Critic:
    def __init__(self, llm_client, model):
        self.llm = llm_client
        self.model = model
        self.parser = JsonParser(required_keys=["decision"])

    def evaluate(
        self,
        user_input: str,
        task: str,
        result: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        use_model = model or self.model
        prompt = build_critic_prompt(user_input, task, result)

        output = self.llm.generate(
            [{"role": "user", "content": prompt.strip()}],
            use_model,
        )
        parsed = self.parser.parse(output or "") or {}

        decision = str(parsed.get("decision") or "reject").strip().lower()
        print(f"Critic decision: {decision}")
        if decision not in ("approve", "reject"):
            decision = "reject"

        return {
            "decision": decision,
            "reason": str(parsed.get("reason") or "").strip(),
            "suggestion": str(parsed.get("suggestion") or "").strip(),
        }

