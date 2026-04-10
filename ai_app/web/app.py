# app.py
import logging
import os
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from ai_app.service.chat_service import ChatService


logger = logging.getLogger(__name__)


def parse_bool(value, default=True):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def build_chat_service_from_env():
    load_dotenv()

    llm_type = os.getenv("LLM_TYPE", "ollama")
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))
    rag_enabled = parse_bool(os.getenv("RAG_ENABLED", "1"), default=True)
    routing_mode = os.getenv("ROUTING_MODE", "hybrid").strip().lower()

    model_name = None
    router_model = os.getenv("ROUTER_MODEL")
    online_conf = None

    if llm_type == "qwen":
        model_name = os.getenv("QWEN_MODEL", "qwen-plus")
        online_conf = {
            "api_key": os.getenv("QWEN_API_KEY"),
            "api_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model_name": model_name,
            "timeout": timeout,
        }
    elif llm_type == "online":
        model_name = os.getenv("ONLINE_MODEL", "online-model")
        online_conf = {
            "api_key": os.getenv("ONLINE_API_KEY"),
            "api_url": os.getenv("ONLINE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model_name": model_name,
            "timeout": timeout,
        }

    if not router_model:
        router_model = model_name

    return ChatService(
        model=model_name,
        router_model=router_model,
        llm_type=llm_type,
        online_conf=online_conf,
        rag_enabled=rag_enabled,
        routing_mode=routing_mode,
    )


class ChatEngine:
    """Thread-safe wrapper around ChatService for web requests."""

    def __init__(self, chat_service):
        self.chat_service = chat_service
        self._lock = threading.Lock()

    def chat(self, message):
        with self._lock:
            reply = self.chat_service.chat(message)
            route = dict(self.chat_service.last_route or {})
        return reply, route


def create_app(chat_service=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    engine = ChatEngine(chat_service or build_chat_service_from_env())
    app.config["CHAT_ENGINE"] = engine

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/chat")
    def chat_api():
        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message 不能为空"}), 400

        try:
            reply, route = app.config["CHAT_ENGINE"].chat(message)
            return jsonify({"reply": reply, "route": route})
        except Exception as exc:
            logger.exception("Web chat failed: %s", exc)
            return jsonify({"error": f"请求失败: {exc}"}), 500

    return app

