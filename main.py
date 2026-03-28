import requests

# 检查 Ollama 根路径

def check_ollama_root():
    try:
        resp = requests.get("http://localhost:11434/")
        print(f"[DEBUG] / 返回: {resp.status_code} {resp.text[:100]}")
        return resp.status_code == 200
    except Exception as e:
        print(f"[DEBUG] / 请求异常: {e}")
        return False

# 检查 /api/tags 路径

def check_ollama_tags():
    try:
        resp = requests.get("http://localhost:11434/api/tags")
        print(f"[DEBUG] /api/tags 返回: {resp.status_code} {resp.text[:100]}")
        return resp.status_code == 200
    except Exception as e:
        print(f"[DEBUG] /api/tags 请求异常: {e}")
        return False

# Ollama 健康检查（仅用 /api/tags）
def check_ollama_health():
    try:
        resp = requests.get("http://localhost:11434/api/tags")
        return resp.status_code == 200
    except Exception as e:
        return False

# 获取本地可用模型列表
def list_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags")
        if resp.status_code == 200:
            data = resp.json()
            models = [m['name'] for m in data.get('models', [])]
            return models
        else:
            return []
    except Exception as e:
        return []

def query_ollama(prompt, model="llama2"):
    """
    向本地 Ollama 模型发送 prompt 并返回回复。
    :param prompt: 用户输入的内容
    :param model: 使用的模型名称，默认 llama2
    :return: 模型回复内容
    """
    url = f"http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "无回复")
    except Exception as e:
        return f"请求失败: {e}"


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("[诊断] 检查 Ollama 服务根路径:")
    check_ollama_root()
    print("[诊断] 检查 /api/tags 路径:")
    check_ollama_tags()
    if not check_ollama_health():
        print("Ollama 服务未启动或端口错误，请检查 Ollama 是否运行在 http://localhost:11434")
        exit(1)
    models = list_ollama_models()
    if not models:
        print("未检测到本地模型，请先用 ollama pull 下载模型。")
        exit(1)
    print(f"检测到本地模型: {models}")
    model = models[0]
    print(f"默认使用模型: {model}")
    print("欢迎使用本地 Ollama AI 交互程序！输入 exit/quit/q 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ("exit", "quit", "q"):
            print("退出程序。")
            break
        reply = query_ollama(user_input, model=model)
        print(f"AI: {reply}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
