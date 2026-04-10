import unittest

try:
    from ai_app.web.app import create_app
except Exception as import_error:
    create_app = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


class FakeChatService:
    def __init__(self):
        self.last_route = {
            "source": "test",
            "use_agent": False,
            "use_rag": False,
            "require_json": False,
        }

    def chat(self, message):
        self.last_route = {
            "source": "test",
            "use_agent": "日志" in message,
            "use_rag": False,
            "require_json": False,
        }
        return f"echo:{message}"


class WebChatApiTest(unittest.TestCase):
    def setUp(self):
        if create_app is None:
            self.skipTest(f"flask/web app import unavailable: {_IMPORT_ERROR}")
        app = create_app(chat_service=FakeChatService())
        app.testing = True
        self.client = app.test_client()

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("status"), "ok")

    def test_chat_success(self):
        resp = self.client.post("/api/chat", json={"message": "你好"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("reply"), "echo:你好")
        self.assertIn("route", data)

    def test_chat_validation(self):
        resp = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()

