import unittest

from ai_app.service.chat_service import ChatService


class DummyLLM:
    def list_models(self):
        return ["dummy-model"]

    def generate(self, messages, model):
        system_text = ""
        if messages and isinstance(messages, list):
            system_text = messages[0].get("content", "")

        if "你是请求路由器" in system_text:
            return '{"use_agent": true, "use_rag": false, "require_json": true, "reason": "日志分析走Agent"}'

        # For agent planning calls.
        return (
            "Thought: 调用日志分析工具\n"
            "Action: analyze_log\n"
            "Action Input: 2026-04-09 ERROR bad request\n"
            "Observation: ...\n"
            "Final Answer: done"
        )


class TestableChatService(ChatService):
    def _get_llm_client(self):
        return DummyLLM()

    def _get_default_model(self):
        return "dummy-model"


class ChatServiceAgentRouteTest(unittest.TestCase):
    def test_agent_route_is_used_when_ai_router_selects_agent(self):
        service = TestableChatService(rag_enabled=False, agent_enabled=True)
        output = service.chat("请分析这个日志: ERROR connect timeout")
        self.assertIn("日志分析", output)
        self.assertEqual(service.last_route.get("source"), "ai")
        self.assertTrue(service.last_route.get("use_agent"))


if __name__ == "__main__":
    unittest.main()

