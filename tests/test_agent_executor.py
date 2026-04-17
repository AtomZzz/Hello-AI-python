import json
import unittest

from ai_app.agent.executor import AgentExecutor


class SequenceLLM:
    def __init__(self, responses):
        self.responses = responses
        self.idx = 0

    def generate(self, messages, model):
        if self.idx >= len(self.responses):
            return self.responses[-1]
        value = self.responses[self.idx]
        self.idx += 1
        return value


class LoopLLM:
    def generate(self, messages, model):
        return json.dumps(
            {
                "thought": "继续分析",
                "action": "analyze_log",
                "action_input": {"text": "截断输入"},
            },
            ensure_ascii=False,
        )


class AgentExecutorTest(unittest.TestCase):
    def test_can_handle_log_request(self):
        executor = AgentExecutor(SequenceLLM(["{}"]), "fake-model")
        self.assertTrue(executor.can_handle("帮我分析这个日志报错"))
        self.assertFalse(executor.can_handle("写个hello world"))

    def test_run_uses_state_machine_and_records_steps(self):
        llm = SequenceLLM(
            [
                json.dumps(
                    {
                        "thought": "先做日志分析",
                        "action": "analyze_log",
                        "action_input": {"text": "只是一行"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "thought": "基于分析做总结",
                        "action": "summarize_text",
                        "action_input": {},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "thought": "结束",
                        "action": "final_answer",
                        "action_input": {"answer": "已完成日志诊断"},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        executor = AgentExecutor(llm, "fake-model")
        output = executor.run("line-1 ERROR timeout\nline-2 ERROR refused", model="fake-model")
        data = json.loads(output)

        self.assertEqual(data["task_type"], "日志分析")
        self.assertEqual(data["final_answer"], "已完成日志诊断")
        self.assertIn("analysis", data)
        self.assertIn("summary", data)
        self.assertGreaterEqual(data["steps_used"], 3)
        first_step = data["steps"][0]
        self.assertEqual(first_step["action"], "analyze_log")
        self.assertEqual(first_step["input"]["text"], "line-1 ERROR timeout\nline-2 ERROR refused")
        self.assertEqual(first_step["observation"]["type"], "analysis_result")
        self.assertEqual(data["steps"][1]["action"], "summarize_text")
        self.assertEqual(data["steps"][1]["observation"]["type"], "summary_result")

    def test_safe_tool_call_blocks_unknown_tool(self):
        llm = SequenceLLM(
            [
                json.dumps(
                    {
                        "thought": "调用不存在的工具",
                        "action": "unknown_tool",
                        "action_input": {},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "thought": "结束",
                        "action": "final_answer",
                        "action_input": {"answer": "done"},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        executor = AgentExecutor(llm, "fake-model")
        output = executor.run("日志有异常", model="fake-model")
        data = json.loads(output)
        self.assertIn("工具不存在", data["steps"][0]["observation"]["data"]["error"])

    def test_safe_tool_call_validates_schema(self):
        executor = AgentExecutor(SequenceLLM(["{}"]), "fake-model")
        result = executor.safe_tool_call("analyze_log", {"text": 123})
        self.assertIn("参数校验失败", result["error"])

    def test_state_machine_stops_on_max_iterations(self):
        executor = AgentExecutor(LoopLLM(), "fake-model", max_steps=2)
        output = executor.run("line-1 ERROR timeout", model="fake-model")
        data = json.loads(output)
        self.assertFalse(data["finished"])
        self.assertLessEqual(data["steps_used"], 2)


if __name__ == "__main__":
    unittest.main()

