import json
import unittest

from ai_app.agent.critic import Critic


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


class CriticTest(unittest.TestCase):
    def test_evaluate_returns_approve(self):
        llm = SequenceLLM(
            [json.dumps({"decision": "approve", "reason": "ok", "suggestion": ""}, ensure_ascii=False)]
        )
        critic = Critic(llm, "fake-model")
        data = critic.evaluate("q", "t1", {"final_answer": "done"})
        self.assertEqual(data["decision"], "approve")
        self.assertEqual(data["reason"], "ok")

    def test_evaluate_invalid_decision_defaults_to_reject(self):
        llm = SequenceLLM([json.dumps({"decision": "maybe"}, ensure_ascii=False)])
        critic = Critic(llm, "fake-model")
        data = critic.evaluate("q", "t1", {"final_answer": "done"})
        self.assertEqual(data["decision"], "reject")


if __name__ == "__main__":
    unittest.main()

