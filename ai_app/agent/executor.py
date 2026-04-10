# executor.py
import json
import logging
import re

from ai_app.agent.tools import analyze_log, summarize_text
from ai_app.parser.json_parser import JsonParser
from ai_app.prompt.templates import build_agent_messages, build_json_repair_messages, build_log_summary_messages


logger = logging.getLogger(__name__)


class AgentExecutor:
    def __init__(self, llm_client, model):
        self.llm_client = llm_client
        self.model = model
        self.tools = {}
        self.tool_descriptions = {}
        self.summary_parser = JsonParser(
            required_keys=["overview", "severity", "root_cause", "key_evidence", "next_actions", "confidence"]
        )
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
        input_match = re.search(r"^Action Input:\s*([\s\S]+?)\n(?:Observation:|Final Answer:|$)", agent_text, flags=re.MULTILINE)
        action = action_match.group(1).strip() if action_match else ""
        action_input = input_match.group(1).strip() if input_match else ""
        return action, action_input

    @staticmethod
    def _normalize_list(value):
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if value in (None, ""):
            return []
        return [str(value)]

    def _normalize_summary(self, summary_data):
        if not isinstance(summary_data, dict):
            return None
        normalized = dict(summary_data)
        normalized["root_cause"] = self._normalize_list(normalized.get("root_cause"))
        normalized["key_evidence"] = self._normalize_list(normalized.get("key_evidence"))
        normalized["next_actions"] = self._normalize_list(normalized.get("next_actions"))
        normalized["overview"] = str(normalized.get("overview") or "")
        normalized["severity"] = str(normalized.get("severity") or "P3").upper()
        normalized["confidence"] = str(normalized.get("confidence") or "medium").lower()
        return normalized

    def _build_fallback_summary(self, analysis):
        fallback = summarize_text(analysis)
        normalized = self._normalize_summary(fallback) or {
            "overview": "暂未生成可用诊断总结。",
            "severity": "P3",
            "root_cause": analysis.get("root_cause", []),
            "key_evidence": analysis.get("evidence", []),
            "next_actions": analysis.get("next_actions", []),
            "confidence": "low",
        }
        normalized["fallback_used"] = True
        return normalized

    def _generate_ai_summary(self, user_input, analysis, use_model):
        try:
            raw_summary = self.llm_client.generate(build_log_summary_messages(user_input, analysis), use_model)
            parsed = self.summary_parser.parse(raw_summary)
            if not parsed:
                repair_messages = build_json_repair_messages(raw_summary, self.summary_parser.required_keys)
                repaired = self.llm_client.generate(repair_messages, use_model)
                parsed = self.summary_parser.parse(repaired)
            normalized = self._normalize_summary(parsed)
            if normalized:
                normalized["fallback_used"] = False
                return normalized
        except Exception as exc:
            logger.warning("AI summary generation failed, fallback to rule summary: %s", exc)
        return self._build_fallback_summary(analysis)

    def run(self, user_input, model=None):
        use_model = model or self.model
        llm_output = ""
        try:
            llm_output = self.llm_client.generate(
                build_agent_messages(user_input, tool_specs=self.list_tool_specs()),
                use_model,
            )
        except Exception as exc:
            logger.warning("Agent planning failed, fallback to direct tool chain: %s", exc)

        action, action_input = self._parse_action(llm_output or "")
        tool_input = action_input or user_input

        # Safety fallback: keep tool use controlled even when LLM action parsing fails.
        if action not in self.tools:
            action = "analyze_log"

        analysis = self.tools[action](tool_input)
        summary = self._generate_ai_summary(tool_input, analysis, use_model)

        final_answer = {
            "task_type": "日志分析",
            "action": action,
            "analysis": analysis,
            "root_cause": summary.get("root_cause", analysis.get("root_cause", [])),
            "evidence": summary.get("key_evidence", analysis.get("evidence", [])),
            "next_actions": summary.get("next_actions", analysis.get("next_actions", [])),
            "summary": summary,
        }
        return json.dumps(final_answer, ensure_ascii=False, indent=2)

