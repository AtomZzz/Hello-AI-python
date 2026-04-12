import json
import unittest

from ai_app.agent.executor import AgentExecutor


class FakeLLM:
    def generate(self, messages, model):
        prompt = messages[0].get("content", "") if messages else ""
        if "Observation:" not in prompt:
            return (
                "Thought: 先执行日志分析获取结构化信息\n"
                "Action: analyze_log\n"
                "Action Input: 2024-04-09 10:00:10 [ERROR] Deadlock found when trying to get lock; try restarting transaction"
            )
        if '"type": "summary_result"' in prompt:
            return (
                "Thought: 已经完成分析\n"
                "Final Answer: 根因为高频异常模式，建议优先处理高频异常模式并执行修复。"
            )
        if '"type": "analysis_result"' in prompt:
            return (
                "Thought: 已有分析结果，继续调用总结工具\n"
                "Action: summarize_text\n"
                "Action Input: 使用上一步日志分析结果生成总结"
            )
        return "Final Answer: 完成"


class FakeInvalidSummaryLLM:
    def generate(self, messages, model):
        prompt = messages[0].get("content", "") if messages else ""
        if "Observation:" not in prompt:
            return (
                "Thought: 先分析日志\n"
                "Action: analyze_log\n"
                "Action Input: 2024-04-09 10:00:15 [ERROR] Connection timeout: Too many connections (max_connections: 151)"
            )
        return "Final Answer: 无法继续工具调用，使用当前结果。"


class FakeCustomToolLLM:
    def generate(self, messages, model):
        prompt = messages[0].get("content", "") if messages else ""
        if "Observation:" not in prompt:
            return (
                "Thought: 使用自定义工具\n"
                "Action: custom_log_tool\n"
                "Action Input: custom payload"
            )
        return "Final Answer: done"


class FakeHallucinationLLM:
    def generate(self, messages, model):
        prompt = messages[0].get("content", "") if messages else ""
        if "Observation:" not in prompt:
            return "Thought: 先分析日志\nAction: analyze_log\nAction Input: [2026-04-10T09:15:30.456Z] ERROR one-line"
        return "Final Answer: H2 未启动导致连接失败"


class FakeSummarizePayloadLLM:
    def generate(self, messages, model):
        prompt = messages[0].get("content", "") if messages else ""
        if "Observation:" not in prompt:
            return "Thought: 先分析\nAction: analyze_log\nAction Input: ignored"
        if '"type": "analysis_result"' in prompt:
            return (
                "Thought: 使用结构化payload做总结\n"
                "Action: summarize_text\n"
                "Action Input: {\"root_cause\": [\"结构化根因\"], \"evidence\": [\"结构化证据\"], \"next_actions\": [\"结构化动作\"], \"error_count\": 1}"
            )
        return "Final Answer: 结构化根因已确认。"


class AgentExecutorTest(unittest.TestCase):
    def test_can_handle_log_request(self):
        executor = AgentExecutor(FakeLLM(), "fake-model")
        self.assertTrue(executor.can_handle("帮我分析这个日志报错"))
        self.assertFalse(executor.can_handle("写个hello world"))

    def test_run_returns_structured_json(self):
        executor = AgentExecutor(FakeLLM(), "fake-model")
        output = executor.run("日志里有error", model="fake-model")
        data = json.loads(output)
        self.assertEqual(data["task_type"], "日志分析")
        self.assertIn("analysis", data)
        self.assertIn("summary", data)
        self.assertIn("root_cause", data)
        self.assertGreaterEqual(data["steps_used"], 2)
        self.assertEqual(data["tool_trace"][0]["action"], "analyze_log")
        self.assertEqual(data["tool_trace"][1]["action"], "summarize_text")
        self.assertEqual(data["tool_trace"][0]["observation"]["type"], "analysis_result")
        self.assertEqual(data["tool_trace"][1]["observation"]["type"], "summary_result")
        self.assertIn("高频异常模式", data["final_answer"])
        self.assertTrue(data["final_answer_grounded"])

    def test_run_falls_back_when_ai_summary_invalid(self):
        executor = AgentExecutor(FakeInvalidSummaryLLM(), "fake-model")
        output = executor.run("日志里有error", model="fake-model")
        data = json.loads(output)
        self.assertGreaterEqual(data["steps_used"], 1)
        self.assertEqual(data["tool_trace"][0]["action_input_effective"], "日志里有error")
        self.assertIn("overview", data["summary"])
        self.assertTrue(data["final_answer"])

    def test_analyze_log_forces_full_user_input(self):
        executor = AgentExecutor(FakeInvalidSummaryLLM(), "fake-model")
        multiline_log = "line-1 ERROR timeout\nline-2 ERROR refused"
        output = executor.run(multiline_log, model="fake-model")
        data = json.loads(output)
        first_step = data["tool_trace"][0]
        self.assertEqual(first_step["action"], "analyze_log")
        self.assertEqual(first_step["action_input_effective"], multiline_log)

    def test_rejects_ungrounded_final_answer(self):
        executor = AgentExecutor(FakeHallucinationLLM(), "fake-model", max_steps=2)
        output = executor.run("2026-04-10 ERROR deadlock", model="fake-model")
        data = json.loads(output)
        self.assertNotIn("H2", data["final_answer"])
        self.assertTrue(data["final_answer_grounded"])

    def test_summarize_uses_structured_action_input_when_valid(self):
        executor = AgentExecutor(FakeSummarizePayloadLLM(), "fake-model", max_steps=4)
        output = executor.run("2026-04-10 ERROR connection timeout", model="fake-model")
        data = json.loads(output)
        self.assertEqual(data["tool_trace"][1]["action"], "summarize_text")
        self.assertIsInstance(data["tool_trace"][1]["action_input_effective"], dict)
        self.assertIn("结构化根因", data["summary"]["root_cause"])

    def test_register_tool_and_execute(self):
        executor = AgentExecutor(FakeCustomToolLLM(), "fake-model")

        def custom_log_tool(text):
            return {"line_count": 1, "levels": {"ERROR": 1}, "error_count": 1, "error_samples": [text]}

        executor.register_tool("custom_log_tool", custom_log_tool, "自定义日志分析")
        output = executor.run("ignored", model="fake-model")
        data = json.loads(output)
        self.assertEqual(data["tool_trace"][0]["action"], "custom_log_tool")
        self.assertEqual(data["analysis"]["error_count"], 1)


if __name__ == "__main__":
    unittest.main()

