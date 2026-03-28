# main.py
from ai_app.service.chat_service import ChatService

if __name__ == '__main__':
    try:
        chat_service = ChatService()
    except RuntimeError as e:
        print(str(e))
        exit(1)
    print("欢迎使用本地 Ollama AI 交互程序！输入 exit/quit/q 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ("exit", "quit", "q"):
            print("退出程序。"); break
        reply = chat_service.chat(user_input)
        print(f"AI: {reply}")

