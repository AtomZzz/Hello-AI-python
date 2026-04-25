import json
from typing import Any, Dict, List, Optional

from ai_app.agent.planner import Planner


class PlanExecutor:
    def __init__(
        self,
        planner: Planner,
        agent_executor,
        critic,
        max_retry: int = 2,
        max_replans: int = 1,
    ):
        self.planner = planner
        self.agent = agent_executor
        self.critic = critic
        self.max_retry = max_retry
        self.max_replans = max_replans

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
    def _build_retry_input(previous_input: str, critique: Dict[str, Any], next_attempt: int) -> str:
        extra_strategy = ""
        # attempt=0 是第一次；next_attempt=1 表示第二次尝试，开始要求“换角度/换思路”。
        if next_attempt == 1:
            extra_strategy = "\n\n请尝试不同的分析角度，换一种思路验证结论，避免重复上次路径。"

        return (
            "原始任务：\n"
            f"{previous_input}\n\n"
            "上一次结果存在问题：\n"
            f"{critique.get('reason', '')}\n\n"
            "改进建议：\n"
            f"{critique.get('suggestion', '')}\n\n"
            "请重新执行任务"
            f"{extra_strategy}"
        )

    def run(self, user_input: str, model: Optional[str] = None) -> str:
        original_plan = self.planner.plan(user_input, model=model)
        tasks = list(original_plan)
        execution_trace: List[Dict[str, Any]] = []
        context = ""
        answers: List[str] = []  # 仅记录通过审核的结论
        approve_count = 0
        reject_count = 0
        replans_used = 0
        completed_tasks: List[str] = []

        idx = 0
        while idx < len(tasks):
            task = tasks[idx]
            attempt_input = (
                f"任务：{task}\n\n"
                f"原始问题：\n{user_input}\n\n"
                f"已知信息：\n{context}".strip()
            )
            last_result: Dict[str, Any] = {}
            last_final_answer = ""
            approved = False
            last_critique: Dict[str, Any] = {}

            for attempt in range(self.max_retry + 1):
                raw_result = self.agent.run(attempt_input, model=model)
                parsed_result = self._parse_agent_result(raw_result)
                final_answer = self._extract_final_answer(parsed_result)
                critique = self.critic.evaluate(
                    user_input=user_input,
                    task=task,
                    result=parsed_result,
                    model=model,
                )
                last_critique = critique

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
                attempt_input = self._build_retry_input(
                    attempt_input,
                    critique,
                    next_attempt=attempt + 1,
                )

            # 只允许“通过审核的结果”进入上下文，避免污染后续任务。
            if approved and last_final_answer:
                answers.append(last_final_answer)
                context = (context + "\n" + last_final_answer).strip()
                completed_tasks.append(task)
                idx += 1
                continue

            # 未通过审核：尝试让 Critic 反馈回流到 Planner 形成闭环，重规划“剩余任务”。
            if (not approved) and replans_used < self.max_replans:
                replans_used += 1
                new_remaining = self.planner.replan(
                    user_input=user_input,
                    completed_tasks=completed_tasks,
                    failed_task=task,
                    critique=last_critique,
                    context=context,
                    model=model,
                )
                # 用重规划的任务替换当前剩余任务（不包含已完成任务）
                tasks = list(new_remaining)
                idx = 0
                continue

            if not approved and not last_result:
                reject_count += 1
            idx += 1

        final_answer = answers[-1] if answers else "未通过审核，未产出有效结论。"
        critic_summary = {
            "approved": approve_count,
            "rejected": reject_count,
            "max_retry": self.max_retry,
            "total_attempts": len(execution_trace),
            "replans_used": replans_used,
        }
        output = {
            "plan": original_plan,
            "execution_trace": execution_trace,
            "final_answer": final_answer,
            "critic_summary": critic_summary,
            # Backward-compatible aliases.
            "tasks": original_plan,
            "results": [item["result"] for item in execution_trace],
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

