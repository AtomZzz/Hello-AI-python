import json
import unittest

from ai_app.agent.executor import AgentExecutor


class FakeLLM:
    def generate(self, messages, model):
        system_text = ""
        if messages and isinstance(messages, list):
            system_text = messages[0].get("content", "")

        if "你是企业级日志诊断专家" in system_text:
            return json.dumps(
                {
                    "overview": "检测到数据库事务与锁竞争异常，属于高优先级故障。",
                    "severity": "P2",
                    "root_cause": ["存在死锁竞争"],
                    "key_evidence": ["Deadlock found when trying to get lock"],
                    "next_actions": ["缩短事务并增加重试。"],
                    "confidence": "high",
                },
                ensure_ascii=False,
            )

        return (
            "Thought: 我先分析日志\n"
            "Action: analyze_log\n"
            "Action Input: 2024-04-09 10:00:10 [ERROR] Deadlock found when trying to get lock; try restarting transaction\n"
            "Observation: ...\n"
            "Final Answer: done"
        )


class FakeInvalidSummaryLLM:
    def generate(self, messages, model):
        system_text = ""
        if messages and isinstance(messages, list):
            system_text = messages[0].get("content", "")

        if "你是企业级日志诊断专家" in system_text:
            return "not-json"
        if "你是JSON修复器" in system_text:
            return "still-not-json"
        return (
            "Thought: 我先分析日志\n"
            "Action: analyze_log\n"
            "Action Input: 2024-04-09 10:00:15 [ERROR] Connection timeout: Too many connections (max_connections: 151)\n"
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
        self.assertIn("root_cause", data)
        self.assertIn("存在死锁竞争", data["root_cause"])
        self.assertEqual(data["summary"]["severity"], "P2")
        self.assertFalse(data["summary"]["fallback_used"])

    def test_run_falls_back_when_ai_summary_invalid(self):
        executor = AgentExecutor(FakeInvalidSummaryLLM(), "fake-model")
        output = executor.run("日志里有error", model="fake-model")
        data = json.loads(output)
        self.assertTrue(data["summary"]["fallback_used"])
        self.assertTrue(data["root_cause"])
        self.assertIn("高频异常模式", data["root_cause"][0])
        self.assertIn("overview", data["summary"])

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

