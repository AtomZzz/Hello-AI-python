import json
from typing import Any, Dict, List, Optional

from ai_app.agent.planner import Planner


class PlanExecutor:
    def __init__(self, planner: Planner, agent_executor, critic, max_retry: int = 2):
        self.planner = planner
        self.agent = agent_executor
        self.critic = critic
        self.max_retry = max_retry

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

    @staticmethod
    def _build_retry_input(previous_input: str, critique: Dict[str, Any]) -> str:
        return (
            "原始任务：\n"
            f"{previous_input}\n\n"
            "上一次结果存在问题：\n"
            f"{critique.get('reason', '')}\n\n"
            "改进建议：\n"
            f"{critique.get('suggestion', '')}\n\n"
            "请重新执行任务"
        )

    def run(self, user_input: str, model: Optional[str] = None) -> str:
        tasks = self.planner.plan(user_input, model=model)
        execution_trace: List[Dict[str, Any]] = []
        context = ""
        answers: List[str] = []
        approve_count = 0
        reject_count = 0

        for task in tasks:
            attempt_input = (
                f"任务：{task}\n\n"
                f"原始问题：\n{user_input}\n\n"
                f"已知信息：\n{context}".strip()
            )
            last_result: Dict[str, Any] = {}
            last_final_answer = ""
            approved = False

            for attempt in range(self.max_retry + 1):
                raw_result = self.agent.run(attempt_input, model=model)
                parsed_result = self._parse_agent_result(raw_result)
                final_answer = self._extract_final_answer(parsed_result)
                critique = self.critic.evaluate(user_input, tasks, parsed_result, model=model)

                execution_trace.append(
                    {
                        "task": task,
                        "attempt": attempt,
                        "input": attempt_input,
                        "result": parsed_result,
                        "critique": critique,
                    }
                )

                last_result = parsed_result
                last_final_answer = final_answer
                if critique.get("decision") == "approve":
                    approve_count += 1
                    approved = True
                    break

                reject_count += 1
                attempt_input = self._build_retry_input(attempt_input, critique)

            if last_final_answer:
                answers.append(last_final_answer)
                context = (context + "\n" + last_final_answer).strip()

            if not approved and not last_result:
                reject_count += 1

        final_answer = answers[-1] if answers else "未产出有效结论。"
        critic_summary = {
            "approved": approve_count,
            "rejected": reject_count,
            "max_retry": self.max_retry,
            "total_attempts": len(execution_trace),
        }
        output = {
            "plan": tasks,
            "execution_trace": execution_trace,
            "final_answer": final_answer,
            "critic_summary": critic_summary,
            # Backward-compatible aliases.
            "tasks": tasks,
            "results": [item["result"] for item in execution_trace],
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

