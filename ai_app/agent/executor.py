# executor.py
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ai_app.agent.tools import FunctionTool, Tool, build_default_tools
from ai_app.parser.json_parser import JsonParser
from ai_app.prompt.templates import build_agent_step_messages


logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 5
    finished: bool = False


class AgentExecutor:
    def __init__(self, llm_client, model, max_steps=5):
        self.llm_client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.tools: Dict[str, Tool] = {}
        self.step_parser = JsonParser(required_keys=["thought", "action", "action_input"])
        for tool in build_default_tools():
            self.register_tool(tool)

    def register_tool(self, tool_or_name, fn=None, description="", input_schema=None):
        if isinstance(tool_or_name, Tool):
            tool = tool_or_name
        else:
            name = str(tool_or_name or "").strip()
            if not name or not callable(fn):
                raise ValueError("工具注册失败: name不能为空且fn必须可调用")

            schema = input_schema or {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
            }

            def _legacy_handler(input_data):
                payload = input_data or {}
                if isinstance(payload, dict) and "text" in payload and len(payload) == 1:
                    return fn(payload.get("text"))
                return fn(payload)

            tool = FunctionTool(
                name=name,
                description=description or "",
                input_schema=schema,
                handler=_legacy_handler,
            )

        self.tools[tool.name] = tool

    def list_tool_specs(self):
        specs = []
        for idx, tool in enumerate(self.tools.values(), start=1):
            label = chr(ord("a") + idx - 1)
            specs.append(
                f"{label}. {tool.name} - {tool.description} | schema={json.dumps(tool.input_schema, ensure_ascii=False)}"
            )
        return specs

    def can_handle(self, user_input):
        text = (user_input or "").lower()
        indicators = [
            "日志", "log", "traceback", "exception", "error", "报错",
            "stack trace", "fatal", "warn", "warning",
        ]
        return any(word in text for word in indicators)

    @staticmethod
    def _build_observation(action_name, result):
        if action_name == "analyze_log":
            observation_type = "analysis_result"
        elif action_name == "summarize_text":
            observation_type = "summary_result"
        else:
            observation_type = "tool_result"
        return {
            "type": observation_type,
            "data": result,
        }

    @staticmethod
    def _normalize_action_input(action_input: Any) -> Dict[str, Any]:
        if isinstance(action_input, dict):
            return action_input
        if isinstance(action_input, str):
            return {"text": action_input}
        if action_input is None:
            return {}
        return {"value": action_input}

    def _validate_input(self, payload: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
        if schema.get("type") == "object" and not isinstance(payload, dict):
            return False, "action_input 必须是对象"

        required = schema.get("required") or []
        for key in required:
            if key not in payload:
                return False, f"缺少必填字段: {key}"

        type_map = {
            "string": str,
            "object": dict,
            "array": list,
            "number": (int, float),
            "boolean": bool,
        }
        properties = schema.get("properties") or {}
        for key, rule in properties.items():
            if key not in payload:
                continue
            expected_type = rule.get("type")
            if expected_type in type_map and not isinstance(payload[key], type_map[expected_type]):
                return False, f"字段 {key} 类型错误，期望 {expected_type}"
        return True, ""

    def safe_tool_call(self, action, action_input):
        if action not in self.tools:
            return {
                "error": f"工具不存在: {action}",
                "available_tools": list(self.tools.keys()),
            }

        tool = self.tools[action]
        payload = self._normalize_action_input(action_input)
        ok, reason = self._validate_input(payload, tool.input_schema)
        if not ok:
            return {
                "error": f"参数校验失败: {reason}",
                "schema": tool.input_schema,
            }

        try:
            return tool.run(payload)
        except Exception as exc:
            return {"error": f"工具执行失败: {exc}"}

    def _parse_step_json(self, llm_output: str) -> Dict[str, Any]:
        parsed = self.step_parser.parse(llm_output or "")
        if parsed:
            return parsed

        # Backward-compatible fallback for old non-JSON ReAct format.
        action_match = re.search(r"^Action:\s*(.+)$", llm_output or "", flags=re.MULTILINE)
        input_match = re.search(
            r"^Action Input:\s*([\s\S]+?)(?:\n(?:Observation:|Final Answer:)|$)",
            llm_output or "",
            flags=re.MULTILINE,
        )
        final_match = re.search(r"^Final Answer:\s*([\s\S]+)$", llm_output or "", flags=re.MULTILINE)
        thought_match = re.search(r"^Thought:\s*([\s\S]+?)(?:\nAction:|\nFinal Answer:|$)", llm_output or "", flags=re.MULTILINE)

        if final_match:
            return {
                "thought": (thought_match.group(1).strip() if thought_match else ""),
                "action": "final_answer",
                "action_input": {"answer": final_match.group(1).strip()},
            }

        return {
            "thought": (thought_match.group(1).strip() if thought_match else ""),
            "action": (action_match.group(1).strip() if action_match else ""),
            "action_input": (input_match.group(1).strip() if input_match else {}),
        }

    @staticmethod
    def _extract_summary(result: Any) -> Dict[str, Any]:
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _extract_analysis(result: Any) -> Dict[str, Any]:
        return result if isinstance(result, dict) else {}

    def run(self, user_input, model=None):
        use_model = model or self.model
        state = AgentState(max_iterations=self.max_steps)
        scratchpad = ""
        analysis = {}
        summary = {}
        final_answer_text = ""

        while not state.finished:
            state.iteration += 1
            if state.iteration > state.max_iterations:
                break

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
                logger.warning("Agent step failed at iteration=%s: %s", state.iteration, exc)
                llm_output = ""

            parsed = self._parse_step_json(llm_output)
            thought = str(parsed.get("thought") or "").strip()
            action = str(parsed.get("action") or "").strip()
            action_input = self._normalize_action_input(parsed.get("action_input"))

            if action == "final_answer":
                final_answer_text = str(
                    action_input.get("answer")
                    or action_input.get("final_answer")
                    or action_input.get("text")
                    or ""
                ).strip()
                if not final_answer_text:
                    final_answer_text = "已完成分析，请查看 steps 里的工具输出。"
                state.finished = True
                state.steps.append(
                    {
                        "thought": thought,
                        "action": action,
                        "input": action_input,
                        "observation": {"type": "final_answer", "data": final_answer_text},
                    }
                )
                break

            # Enterprise guardrails for deterministic routing.
            if action == "analyze_log":
                effective_input = {"text": user_input}
            elif action == "summarize_text":
                if isinstance(action_input.get("analysis"), dict):
                    effective_input = {"analysis": action_input.get("analysis")}
                elif analysis:
                    effective_input = {"analysis": analysis}
                else:
                    effective_input = {"text": user_input}
            else:
                effective_input = action_input

            tool_result = self.safe_tool_call(action, effective_input)
            observation = self._build_observation(action, tool_result)

            state.steps.append(
                {
                    "thought": thought,
                    "action": action,
                    "input": effective_input,
                    "observation": observation,
                }
            )

            if action == "analyze_log" and isinstance(tool_result, dict) and "error" not in tool_result:
                analysis = self._extract_analysis(tool_result)
            if action == "summarize_text" and isinstance(tool_result, dict) and "error" not in tool_result:
                summary = self._extract_summary(tool_result)

            scratchpad += (
                f"\nStepJSON: {json.dumps(parsed, ensure_ascii=False)}\n"
                f"Observation: {json.dumps(observation, ensure_ascii=False)}\n"
            )

        if not analysis:
            analysis_result = self.safe_tool_call("analyze_log", {"text": user_input})
            if isinstance(analysis_result, dict) and "error" not in analysis_result:
                analysis = analysis_result

        if not summary and analysis:
            summary_result = self.safe_tool_call("summarize_text", {"analysis": analysis})
            if isinstance(summary_result, dict) and "error" not in summary_result:
                summary = summary_result

        if not final_answer_text:
            final_answer_text = (
                str(summary.get("overview"))
                if isinstance(summary, dict) and summary.get("overview")
                else "已完成日志分析，请查看结构化结果。"
            )

        result = {
            "task_type": "日志分析",
            "final_answer": final_answer_text,
            "steps": state.steps,
            "max_iterations": state.max_iterations,
            "iterations": state.iteration,
            "finished": state.finished,
            "analysis": analysis,
            "summary": summary,
            # Backward-compatible fields.
            "steps_used": len(state.steps),
            "tool_trace": state.steps,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

