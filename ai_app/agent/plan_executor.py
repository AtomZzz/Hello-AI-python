import json
from typing import Any, Dict, List, Optional

from ai_app.agent.planner import Planner


class PlanExecutor:
    def __init__(self, planner: Planner, agent_executor):
        self.planner = planner
        self.agent = agent_executor

    @staticmethod
    def _parse_agent_result(raw_result: Any) -> Dict[str, Any]:
        if isinstance(raw_result, dict):
            return raw_result
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {"raw_output": raw_result}
        return {"raw_output": raw_result}

    @staticmethod
    def _extract_final_answer(result: Dict[str, Any]) -> str:
        return str(result.get("final_answer") or "").strip()

    def run(self, user_input: str, model: Optional[str] = None) -> str:
        tasks = self.planner.plan(user_input, model=model)
        execution_trace: List[Dict[str, Any]] = []
        context = ""
        answers: List[str] = []

        for task in tasks:
            input_with_context = (
                f"任务：{task}\n\n"
                f"原始问题：\n{user_input}\n\n"
                f"已知信息：\n{context}".strip()
            )
            raw_result = self.agent.run(input_with_context, model=model)
            parsed_result = self._parse_agent_result(raw_result)
            final_answer = self._extract_final_answer(parsed_result)
            if final_answer:
                answers.append(final_answer)
                context = (context + "\n" + final_answer).strip()

            execution_trace.append(
                {
                    "task": task,
                    "input": input_with_context,
                    "result": parsed_result,
                    "final_answer": final_answer,
                }
            )

        final_answer = answers[-1] if answers else "未产出有效结论。"
        output = {
            "plan": tasks,
            "execution_trace": execution_trace,
            "final_answer": final_answer,
            # Backward-compatible aliases.
            "tasks": tasks,
            "results": [item["result"] for item in execution_trace],
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

