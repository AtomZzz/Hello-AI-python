# main.py
from ai_app.service.chat_service import ChatService
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    load_dotenv()
    llm_type = os.getenv("LLM_TYPE", "ollama")
    online_conf = {
        "api_key": os.getenv("QWEN_API_KEY")
    } if llm_type == "qwen" else None
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
