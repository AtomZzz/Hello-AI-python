import json
import unittest

from ai_app.agent.executor import AgentExecutor


class FakeLLM:
    def generate(self, messages, model):
        return (
            "Thought: 我先分析日志\n"
            "Action: analyze_log\n"
            "Action Input: 2026-04-09 ERROR timeout\n"
            "Observation: ...\n"
            "Final Answer: done"
        )


class FakeCustomToolLLM:
    def generate(self, messages, model):
        return (
            "Thought: 使用自定义工具\n"
            "Action: custom_log_tool\n"
            "Action Input: custom payload\n"
            "Observation: ...\n"
            "Final Answer: done"
        )


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
        self.assertNotIn('{"line_count"', data["summary"])

    def test_register_tool_and_execute(self):
        executor = AgentExecutor(FakeCustomToolLLM(), "fake-model")

        def custom_log_tool(text):
            return {"line_count": 1, "levels": {"ERROR": 1}, "error_count": 1, "error_samples": [text]}

        executor.register_tool("custom_log_tool", custom_log_tool, "自定义日志分析")
        output = executor.run("ignored", model="fake-model")
        data = json.loads(output)
        self.assertEqual(data["action"], "custom_log_tool")
        self.assertEqual(data["analysis"]["error_count"], 1)


if __name__ == "__main__":
    unittest.main()

