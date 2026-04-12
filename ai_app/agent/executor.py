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

    @staticmethod
    def _contains_unseen_facts(answer_text, tool_trace):
        observation_text = json.dumps([item.get("observation", {}) for item in tool_trace], ensure_ascii=False).lower()
        # Reject unseen English-like technical tokens (e.g., H2, Redis, Kafka) in final answer.
        token_candidates = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", answer_text or "")
        generic_tokens = {"final", "answer", "error", "warning", "timeout", "traceback", "log"}
        for token in token_candidates:
            token_l = token.lower()
            if token_l in generic_tokens:
                continue
            if token_l not in observation_text:
                return True

        # Reject unseen numeric facts in final answer.
        number_tokens = re.findall(r"\b\d+\b", answer_text or "")
        for number in number_tokens:
            if number not in observation_text:
                return True
        return False

    @staticmethod
    def _extract_anchor_items(observation_data):
        if not isinstance(observation_data, dict):
            return []
        anchors = []
        for key in ("root_cause", "key_evidence", "evidence"):
            value = observation_data.get(key)
            if isinstance(value, list):
                anchors.extend(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                anchors.append(value.strip())
        return anchors

    def _is_final_answer_grounded(self, answer_text, tool_trace):
        if not answer_text or not tool_trace:
            return False
        if self._contains_unseen_facts(answer_text, tool_trace):
            return False

        latest_summary = None
        latest_analysis = None
        for item in reversed(tool_trace):
            obs = item.get("observation") or {}
            obs_type = obs.get("type")
            obs_data = obs.get("data")
            if latest_summary is None and obs_type == "summary_result" and isinstance(obs_data, dict):
                latest_summary = obs_data
            if latest_analysis is None and obs_type == "analysis_result" and isinstance(obs_data, dict):
                latest_analysis = obs_data

        answer_text_lower = answer_text.lower()
        anchor_source = latest_summary or latest_analysis or {}
        anchors = self._extract_anchor_items(anchor_source)
        if not anchors:
            return True

        return any(anchor.lower() in answer_text_lower for anchor in anchors)

    @staticmethod
    def _parse_action_input_payload(action_input):
        text = (action_input or "").strip()
        if not text:
            return None
        if not (text.startswith("{") or text.startswith("[")):
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def _build_observation(self, action_name, result):
        if action_name == "analyze_log":
            observation_type = "analysis_result"
        elif action_name == "summarize_text":
            observation_type = "summary_result"
        elif action_name in ("<empty>", "<none>"):
            observation_type = "tool_error"
        else:
            observation_type = "tool_result"
        return {
            "type": observation_type,
            "data": result,
        }

    def run(self, user_input, model=None):
        use_model = model or self.model
        scratchpad = ""
        final_answer_text = ""
        tool_trace = []
        analysis = {}
        summary = None
        final_answer_grounded = False

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
                if not tool_trace:
                    # Final answer must be based on observations.
                    scratchpad += "\n系统提示: 你还没有任何 Observation，请先调用工具再给 Final Answer。\n"
                    final_answer_text = ""
                elif not self._is_final_answer_grounded(final_answer_text, tool_trace):
                    scratchpad += "\n系统提示: 你的 Final Answer 未充分引用 Observation 的 root_cause/evidence，或含未出现信息，请重试。\n"
                    final_answer_text = ""
                else:
                    final_answer_grounded = True
                    break

            action, action_input = self._parse_action(llm_output or "")
            requested_input = action_input or user_input

            # Ensure log analysis always consumes the full original user input.
            if action == "analyze_log":
                tool_input = user_input
            elif action == "summarize_text":
                payload = self._parse_action_input_payload(action_input)
                if not analysis:
                    fallback_analysis = self.tools.get("analyze_log", analyze_log)
                    analysis = self._safe_tool_output(fallback_analysis(user_input))
                    if not isinstance(analysis, dict):
                        analysis = {"raw": analysis}
                # Prefer model-provided structured payload when valid; fallback to analysis.
                tool_input = payload if isinstance(payload, dict) else analysis
            else:
                tool_input = requested_input

            if action not in self.tools:
                tool_result = {
                    "error": f"工具不存在: {action or '<empty>'}",
                    "available_tools": list(self.tools.keys()),
                }
                normalized_action = action or "<empty>"
            else:
                normalized_action = action
                try:
                    tool_result = self._safe_tool_output(self.tools[action](tool_input))
                except Exception as exc:
                    tool_result = {"error": f"工具执行失败: {exc}"}

            observation = self._build_observation(normalized_action, tool_result)

            if normalized_action == "analyze_log" and isinstance(tool_result, dict):
                analysis = tool_result
            elif not analysis and normalized_action != "summarize_text" and isinstance(tool_result, dict):
                # Allow custom analysis tools to provide the primary structured analysis.
                analysis = tool_result
            if normalized_action == "summarize_text":
                summary = tool_result

            tool_trace.append(
                {
                    "step": step,
                    "action": normalized_action,
                    "action_input": requested_input,
                    "action_input_effective": tool_input,
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

        if not final_answer_text:
            fallback_overview = "已完成日志分析，请查看结构化结果。"
            if isinstance(summary, dict):
                fallback_overview = str(summary.get("overview") or fallback_overview)
            elif summary is not None:
                fallback_overview = str(summary)
            final_answer_text = fallback_overview
            final_answer_grounded = True

        if summary is None:
            summary = {
                "overview": final_answer_text,
                "severity": "INFO",
                "root_cause": analysis.get("root_cause", []) if isinstance(analysis, dict) else [],
                "key_evidence": analysis.get("evidence", []) if isinstance(analysis, dict) else [],
                "next_actions": analysis.get("next_actions", []) if isinstance(analysis, dict) else [],
                "confidence": "low",
                "auto_generated": True,
            }

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
            "final_answer_grounded": final_answer_grounded,
        }
        return json.dumps(final_answer, ensure_ascii=False, indent=2)

