# main.py
from ai_app.service.chat_service import ChatService
from dotenv import load_dotenv
import logging
import os

if __name__ == '__main__':
    load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    # logging.basicConfig(
    #     level=getattr(logging, log_level, logging.INFO),
    #     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    # )

    llm_type = os.getenv("LLM_TYPE", "ollama")
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))
    online_conf = None
    if llm_type == "qwen":
        online_conf = {
            "api_key": os.getenv("QWEN_API_KEY"),
            "api_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model_name": os.getenv("QWEN_MODEL", "qwen-plus"),
            "timeout": timeout,
        }
    elif llm_type == "online":
        online_conf = {
            "api_key": os.getenv("ONLINE_API_KEY"),
            "api_url": os.getenv("ONLINE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "timeout": timeout,
        }

    try:
        chat_service = ChatService(llm_type=llm_type, online_conf=online_conf)
    except RuntimeError as e:
        print(str(e))
        exit(1)
    print("欢迎使用本地/线上 AI 交互程序！输入 exit/quit/q 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ("exit", "quit", "q"):
            print("退出程序。"); break
        reply = chat_service.chat(user_input)
        print(f"AI: {reply}")
