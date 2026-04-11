# executor.py
import json
import logging
import re

from ai_app.agent.tools import analyze_log, summarize_text
from ai_app.prompt.templates import build_agent_step_messages


logger = logging.getLogger(__name__)


class AgentExecutor:
    def __init__(self, llm_client, model, max_steps=5):
        self.llm_client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.tools = {}
        self.tool_descriptions = {}
        self.register_tool("analyze_log", analyze_log, "分析日志")
        self.register_tool("summarize_text", summarize_text, "文本总结")

    def register_tool(self, name, fn, description):
        if not name or not callable(fn):
            raise ValueError("工具注册失败: name不能为空且fn必须可调用")
        self.tools[name] = fn
        self.tool_descriptions[name] = description or ""

    def list_tool_specs(self):
        specs = []
        for idx, (name, description) in enumerate(self.tool_descriptions.items(), start=1):
            label = chr(ord("a") + idx - 1)
            specs.append(f"{label}. {name}(text) - {description}")
        return specs

    def can_handle(self, user_input):
        text = (user_input or "").lower()
        indicators = [
            "日志", "log", "traceback", "exception", "error", "报错",
            "stack trace", "fatal", "warn", "warning",
        ]
        return any(word in text for word in indicators)

    @staticmethod
    def _parse_action(agent_text):
        action_match = re.search(r"^Action:\s*(.+)$", agent_text, flags=re.MULTILINE)
        input_match = re.search(r"^Action Input:\s*([\s\S]+?)(?:\n(?:Observation:|Final Answer:)|$)", agent_text, flags=re.MULTILINE)
        action = action_match.group(1).strip() if action_match else ""
        action_input = input_match.group(1).strip() if input_match else ""
        return action, action_input

    @staticmethod
    def _parse_final_answer(agent_text):
        match = re.search(r"^Final Answer:\s*([\s\S]+)$", agent_text, flags=re.MULTILINE)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _normalize_list(value):
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if value in (None, ""):
            return []
        return [str(value)]

    @staticmethod
    def _to_observation_text(observation):
        if isinstance(observation, (dict, list)):
            return json.dumps(observation, ensure_ascii=False)
        return str(observation)

    @staticmethod
    def _safe_tool_output(value):
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            return value
        return str(value)

    def run(self, user_input, model=None):
        use_model = model or self.model
        scratchpad = ""
        final_answer_text = ""
        tool_trace = []
        analysis = {}
        summary = None

        for step in range(1, self.max_steps + 1):
            try:
                llm_output = self.llm_client.generate(
                    build_agent_step_messages(
                        user_input,
                        tool_specs=self.list_tool_specs(),
                        scratchpad=scratchpad,
                    ),
                    use_model,
                )
            except Exception as exc:
                logger.warning("Agent step failed at step=%s, fallback to analyze_log: %s", step, exc)
                llm_output = ""

            final_answer_text = self._parse_final_answer(llm_output or "")
            if final_answer_text:
                break

            action, action_input = self._parse_action(llm_output or "")
            tool_input = action_input or user_input

            if action not in self.tools:
                observation = {
                    "error": f"工具不存在: {action or '<empty>'}",
                    "available_tools": list(self.tools.keys()),
                }
                normalized_action = action or "<empty>"
            else:
                normalized_action = action
                try:
                    observation = self._safe_tool_output(self.tools[action](tool_input))
                except Exception as exc:
                    observation = {"error": f"工具执行失败: {exc}"}

            if normalized_action == "analyze_log" and isinstance(observation, dict):
                analysis = observation
            elif not analysis and normalized_action != "summarize_text" and isinstance(observation, dict):
                # Allow custom analysis tools to provide the primary structured analysis.
                analysis = observation
            if normalized_action == "summarize_text":
                summary = observation

            tool_trace.append(
                {
                    "step": step,
                    "action": normalized_action,
                    "action_input": tool_input,
                    "observation": observation,
                }
            )

            observation_text = self._to_observation_text(observation)
            scratchpad += (
                f"\nThought/Action 输出:\n{(llm_output or '').strip()}\n"
                f"Observation: {observation_text}\n"
            )

        if not analysis:
            fallback_analysis = self.tools.get("analyze_log", analyze_log)
            analysis = self._safe_tool_output(fallback_analysis(user_input))
            if not isinstance(analysis, dict):
                analysis = {"raw": analysis}

        if summary is None:
            summary_tool = self.tools.get("summarize_text", summarize_text)
            summary = self._safe_tool_output(summary_tool(analysis))

        if not final_answer_text:
            if isinstance(summary, dict):
                final_answer_text = str(summary.get("overview") or "已完成日志分析，请查看结构化结果。")
            else:
                final_answer_text = str(summary)

        final_answer = {
            "task_type": "日志分析",
            "max_steps": self.max_steps,
            "steps_used": len(tool_trace),
            "tool_trace": tool_trace,
            "analysis": analysis,
            "root_cause": self._normalize_list((summary or {}).get("root_cause") if isinstance(summary, dict) else analysis.get("root_cause", [])),
            "evidence": self._normalize_list((summary or {}).get("key_evidence") if isinstance(summary, dict) else analysis.get("evidence", [])),
            "next_actions": self._normalize_list((summary or {}).get("next_actions") if isinstance(summary, dict) else analysis.get("next_actions", [])),
            "summary": summary,
            "final_answer": final_answer_text,
        }
        return json.dumps(final_answer, ensure_ascii=False, indent=2)

