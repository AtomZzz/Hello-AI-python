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


class SequenceCritic:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.idx = 0

    def evaluate(self, user_input, task, result, model=None):
        if self.idx >= len(self.decisions):
            decision = self.decisions[-1]
        else:
            decision = self.decisions[self.idx]
            self.idx += 1
        return {
            "decision": decision,
            "reason": "result is incomplete" if decision == "reject" else "",
            "suggestion": "add more evidence" if decision == "reject" else "",
        }


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
        critic = SequenceCritic(["approve", "approve"])
        executor = PlanExecutor(planner, agent, critic)

        output = executor.run("line-1 ERROR timeout", model="fake-model")
        data = json.loads(output)

        self.assertEqual(data["plan"], ["日志分析：先做排查", "总结：汇总结论"])
        self.assertEqual(len(data["execution_trace"]), 2)
        self.assertEqual(data["final_answer"], "处理完成#2")
        self.assertIn("处理完成#1", agent.inputs[1])
        self.assertIn("critic_summary", data)
        self.assertEqual(data["critic_summary"]["approved"], 2)

    def test_plan_executor_retries_when_critic_rejects(self):
        llm = SequenceLLM([json.dumps({"tasks": ["日志分析：定位错误"]}, ensure_ascii=False)])
        planner = Planner(llm, "fake-model")
        agent = RecordingAgent()
        critic = SequenceCritic(["reject", "approve"])
        executor = PlanExecutor(planner, agent, critic, max_retry=2)

        output = executor.run("line-1 ERROR timeout", model="fake-model")
        data = json.loads(output)

        self.assertEqual(len(data["execution_trace"]), 2)
        self.assertEqual(data["execution_trace"][0]["attempt"], 0)
        self.assertEqual(data["execution_trace"][1]["attempt"], 1)
        self.assertEqual(data["execution_trace"][0]["critique"]["decision"], "reject")
        self.assertEqual(data["execution_trace"][1]["critique"]["decision"], "approve")
        self.assertIn("原始任务", agent.inputs[1])
        self.assertEqual(data["critic_summary"]["rejected"], 1)
        self.assertEqual(data["critic_summary"]["approved"], 1)


if __name__ == "__main__":
    unittest.main()

