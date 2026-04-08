# main.py
from ai_app.service.chat_service import ChatService
from dotenv import load_dotenv
import importlib.util
import logging
import os
import sys


def check_rag_dependencies():
    checks = {
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
        "faiss": importlib.util.find_spec("faiss") is not None,
    }
    missing = [name for name, ok in checks.items() if not ok]
    return len(missing) == 0, missing


def parse_bool(value, default=True):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")

if __name__ == '__main__':
    load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print(f"[ENV] Python executable: {sys.executable}")
    rag_enabled = parse_bool(os.getenv("RAG_ENABLED", "1"), default=True)
    routing_mode = os.getenv("ROUTING_MODE", "hybrid").strip().lower()
    print(f"[ROUTER] ROUTING_MODE={routing_mode}")

    if rag_enabled:
        rag_ok, rag_missing = check_rag_dependencies()
        if rag_ok:
            print("[RAG-CHECK] 依赖检查通过: sentence_transformers, faiss")
        else:
            print(f"[RAG-CHECK] 缺少依赖: {', '.join(rag_missing)}")
            print("[RAG-CHECK] 将自动降级为普通LLM流程(不使用RAG检索)。")
    else:
        print("[RAG-CHECK] RAG_ENABLED=0，已禁用RAG初始化。")

    llm_type = os.getenv("LLM_TYPE", "ollama")
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))
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
    if router_model:
        print(f"[ROUTER] ROUTER_MODEL={router_model}")

    try:
        chat_service = ChatService(
            model=model_name,
            router_model=router_model,
            llm_type=llm_type,
            online_conf=online_conf,
            rag_enabled=rag_enabled,
            routing_mode=routing_mode,
        )
    except RuntimeError as e:
        print(str(e))
        exit(1)
    print("欢迎使用本地/线上 AI 交互程序！输入 exit/quit/q 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ("exit", "quit", "q"):
            print("退出程序。"); break
        reply = chat_service.chat(user_input)
        route = chat_service.last_route or {}
        print(
            "[ROUTER] "
            f"source={route.get('source', 'unknown')} "
            f"router_model={route.get('router_model', chat_service.last_router_model)} "
            f"use_rag={route.get('use_rag')} "
            f"require_json={route.get('require_json')} "
            f"reason={route.get('reason', '')}"
        )
        print(f"AI: {reply}")
