import json
import unittest

from ai_app.service.chat_service import ChatService


class DummyLLM:
    def __init__(self):
        self.calls = 0

    def list_models(self):
        return ["dummy-model"]

    def generate(self, messages, model):
        system_text = ""
        user_text = ""
        if messages and isinstance(messages, list):
            system_text = messages[0].get("content", "")
            user_text = messages[-1].get("content", "")

        if "请求路由器" in system_text:
            return '{"use_agent": true, "use_rag": false, "require_json": true, "reason": "日志分析走Agent"}'

        if "任务规划器" in user_text:
            return json.dumps(
                {
                    "tasks": [
                        "日志分析：提取关键异常",
                        "总结：归纳原因",
                    ]
                },
                ensure_ascii=False,
            )

        self.calls += 1
        if self.calls % 2 == 1:
            return json.dumps(
                {
                    "thought": "调用日志分析工具",
                    "action": "analyze_log",
                    "action_input": {"text": "2026-04-09 ERROR connect timeout"},
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "thought": "结束",
                "action": "final_answer",
                "action_input": {"answer": "已完成日志诊断"},
            },
            ensure_ascii=False,
        )


class TestableChatService(ChatService):
    def _get_llm_client(self):
        return DummyLLM()

    def _get_default_model(self):
        return "dummy-model"


class ChatServiceAgentRouteTest(unittest.TestCase):
    def test_agent_route_is_used_when_ai_router_selects_agent(self):
        service = TestableChatService(rag_enabled=False, agent_enabled=True)
        output = service.chat("请分析这个日志 ERROR connect timeout")
        data = json.loads(output)

        self.assertIn("plan", data)
        self.assertIn("execution_trace", data)
        self.assertIn("final_answer", data)
        self.assertTrue(data["plan"])
        self.assertEqual(service.last_route.get("source"), "ai")
        self.assertTrue(service.last_route.get("use_agent"))


if __name__ == "__main__":
    unittest.main()

