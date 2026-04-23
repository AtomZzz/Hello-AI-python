from typing import List, Optional

from ai_app.parser.json_parser import JsonParser


ALLOWED_TASK_TYPES = [
    "日志分析",
    "知识检索",
    "总结",
    "建议",
]

_TASK_KEYWORDS = {
    "日志分析": ["日志", "报错", "异常", "error", "traceback", "分析"],
    "知识检索": ["知识", "检索", "查询", "资料", "文档", "搜索", "rag"],
    "总结": ["总结", "归纳", "汇总", "结论"],
    "建议": ["建议", "方案", "修复", "优化", "下一步"],
}


def build_planner_prompt(user_input):
    return f"""
你是一个任务规划器，请将用户问题拆解为多个步骤。

要求：
1. 输出JSON
2. 每个task必须是可执行的
3. 不要超过5步

格式：
{{
  "tasks": ["task1", "task2"]
}}

用户问题：
{user_input}
"""


class Planner:
    def __init__(self, llm_client, model: str):
        self.llm_client = llm_client
        self.model = model
        self.parser = JsonParser(required_keys=["tasks"])

    def plan(self, user_input: str, model: Optional[str] = None) -> List[str]:
        use_model = model or self.model
        prompt = build_planner_prompt(user_input)
        raw_plan = self.llm_client.generate(
            [{"role": "user", "content": prompt.strip()}],
            use_model,
        )
        parsed = self.parser.parse(raw_plan or "") or {}
        tasks = self._coerce_tasks(parsed.get("tasks"))
        return self._filter_tasks(tasks, user_input)

    @staticmethod
    def _coerce_tasks(tasks_value) -> List[str]:
        if isinstance(tasks_value, list):
            cleaned = [str(item).strip() for item in tasks_value if str(item).strip()]
            return cleaned
        return []

    def _infer_task_type(self, task: str) -> Optional[str]:
        lowered = task.lower()
        for task_type in ALLOWED_TASK_TYPES:
            if task_type in task:
                return task_type
        for task_type, keywords in _TASK_KEYWORDS.items():
            if any(keyword in lowered or keyword in task for keyword in keywords):
                return task_type
        return None

    def _build_fallback_tasks(self, user_input: str) -> List[str]:
        text = (user_input or "").lower()
        if any(k in text for k in ("资料", "文档", "知识", "检索", "查询", "rag", "搜索")):
            return ["知识检索：收集与问题相关的事实", "总结：归纳已知信息", "建议：给出可执行建议"]
        return ["日志分析：提取关键信号", "总结：归纳主要原因", "建议：给出下一步行动"]

    def _filter_tasks(self, tasks: List[str], user_input: str) -> List[str]:
        filtered: List[str] = []
        for task in tasks:
            task_type = self._infer_task_type(task)
            if not task_type:
                continue
            normalized = task if task.startswith(task_type) else f"{task_type}：{task}"
            filtered.append(normalized)
            if len(filtered) >= 5:
                break

        if filtered:
            return filtered
        return self._build_fallback_tasks(user_input)
