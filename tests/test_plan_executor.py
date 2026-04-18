import json
import unittest

from ai_app.agent.plan_executor import PlanExecutor
from ai_app.agent.planner import Planner


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.idx = 0

    def generate(self, messages, model):
        if self.idx >= len(self.responses):
            return self.responses[-1]
        value = self.responses[self.idx]
        self.idx += 1
        return value


class RecordingAgent:
    def __init__(self):
        self.inputs = []

    def run(self, user_input, model=None):
        self.inputs.append(user_input)
        return json.dumps(
            {
                "final_answer": f"处理完成#{len(self.inputs)}",
                "steps": [],
            },
            ensure_ascii=False,
        )


class PlannerAndPlanExecutorTest(unittest.TestCase):
    def test_planner_filters_disallowed_tasks_and_keeps_max_5(self):
        llm = SequenceLLM(
            [
                json.dumps(
                    {
                        "tasks": [
                            "先随便聊聊",
                            "日志分析：定位报错",
                            "知识检索：查相关背景",
                            "总结：归纳原因",
                            "建议：给修复方案",
                            "建议：补充监控项",
                            "建议：扩展预案",
                        ]
                    },
                    ensure_ascii=False,
                )
            ]
        )
        planner = Planner(llm, "fake-model")

        tasks = planner.plan("请帮我分析日志并给建议", model="fake-model")

        self.assertEqual(len(tasks), 5)
        self.assertTrue(all("随便聊聊" not in t for t in tasks))
        self.assertTrue(tasks[0].startswith("日志分析"))

    def test_plan_executor_passes_context_between_tasks(self):
        llm = SequenceLLM(
            [
                json.dumps(
                    {
                        "tasks": [
                            "日志分析：先做排查",
                            "总结：汇总结论",
                        ]
                    },
                    ensure_ascii=False,
                )
            ]
        )
        planner = Planner(llm, "fake-model")
        agent = RecordingAgent()
        executor = PlanExecutor(planner, agent)

        output = executor.run("line-1 ERROR timeout", model="fake-model")
        data = json.loads(output)

        self.assertEqual(data["plan"], ["日志分析：先做排查", "总结：汇总结论"])
        self.assertEqual(len(data["execution_trace"]), 2)
        self.assertEqual(data["final_answer"], "处理完成#2")
        self.assertIn("处理完成#1", agent.inputs[1])


if __name__ == "__main__":
    unittest.main()

