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
        if "Action: summarize_text" in prompt:
            return (
                "Thought: 已经完成分析\n"
                "Final Answer: 已定位异常并给出处理建议"
            )
        if "\"root_cause\"" in prompt:
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
        self.assertEqual(data["final_answer"], "已定位异常并给出处理建议")

    def test_run_falls_back_when_ai_summary_invalid(self):
        executor = AgentExecutor(FakeInvalidSummaryLLM(), "fake-model")
        output = executor.run("日志里有error", model="fake-model")
        data = json.loads(output)
        self.assertGreaterEqual(data["steps_used"], 1)
        self.assertTrue(data["root_cause"])
        self.assertIn("高频异常模式", data["root_cause"][0])
        self.assertIn("overview", data["summary"])
        self.assertTrue(data["final_answer"])

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

