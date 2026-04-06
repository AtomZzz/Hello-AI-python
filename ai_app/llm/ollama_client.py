# ollama_client.py
import requests
from ai_app.llm.base_llm_client import BaseLLMClient

class OllamaClient(BaseLLMClient):
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url.rstrip('/')

    def check_root(self):
        try:
            resp = requests.get(f"{self.base_url}/")
            return resp.status_code == 200
        except Exception:
            return False

    def check_tags(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def health(self):
        return self.check_tags()

    def list_models(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m['name'] for m in data.get('models', [])]
            return []
        except Exception:
            return []

    def generate(self, messages, model):
        try:
            # Backward compatibility for legacy string prompt call sites.
            if isinstance(messages, str):
                url = f"{self.base_url}/api/generate"
                payload = {"model": model, "prompt": messages, "stream": False}
                response = requests.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "无回复")

            url = f"{self.base_url}/api/chat"
            payload = {"model": model, "messages": messages, "stream": False}
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "无回复")
        except Exception as e:
            return f"请求失败: {e}"
