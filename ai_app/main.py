# main.py
from ai_app.service.chat_service import ChatService
from dotenv import load_dotenv
import importlib.util
import logging
import os
import sys


EXIT_COMMANDS = {"exit", "quit", "q"}
PASTE_START_COMMANDS = {":paste", ":p"}
PASTE_END_COMMANDS = {":end", ":done", ":eof"}
PASTE_CANCEL_COMMANDS = {":cancel", ":abort"}


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


def read_user_input(reader=input, writer=print):
    """Read one chat request, supporting optional multiline paste mode."""
    try:
        first_line = reader("你: ")
    except EOFError:
        return "__exit__"

    command = first_line.strip().lower()
    if command in EXIT_COMMANDS:
        return "__exit__"

    if command not in PASTE_START_COMMANDS:
        return first_line

    writer("[INPUT] 已进入粘贴模式。输入 :end 提交，输入 :cancel 取消。")
    lines = []
    while True:
        try:
            line = reader("?> ")
        except EOFError:
            return "\n".join(lines).strip()

        line_command = line.strip().lower()
        if line_command in PASTE_END_COMMANDS:
            return "\n".join(lines).strip()
        if line_command in PASTE_CANCEL_COMMANDS:
            writer("[INPUT] 已取消本次粘贴输入。")
            return ""
        lines.append(line)

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
    print("提示: 输入 :paste 进入多行粘贴模式，可一次提交整段日志。")
    while True:
        user_input = read_user_input()
        if user_input == "__exit__":
            print("退出程序。")
            break
        if not user_input.strip():
            continue
        reply = chat_service.chat(user_input)
        route = chat_service.last_route or {}
        print(
            "[ROUTER] "
            f"source={route.get('source', 'unknown')} "
            f"router_model={route.get('router_model', chat_service.last_router_model)} "
            f"use_agent={route.get('use_agent')} "
            f"use_rag={route.get('use_rag')} "
            f"require_json={route.get('require_json')} "
            f"reason={route.get('reason', '')}"
        )
        print(f"AI: {reply}")
