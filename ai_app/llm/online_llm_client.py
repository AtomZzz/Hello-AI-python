# online_llm_client.py
import requests
from ai_app.llm.base_llm_client import BaseLLMClient

class OnlineLLMClient(BaseLLMClient):
    def __init__(self, api_url, api_key=None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key

    def list_models(self):
        # 假设线上API有模型列表接口，否则可返回默认模型
        # return ["online-model-1", "online-model-2"]
        return ["online-model"]

    def generate(self, prompt, model):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": model,
            "prompt": prompt
        }
        try:
            resp = requests.post(f"{self.api_url}/generate", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # 假设返回格式为 {"response": "..."}
            return data.get("response", "无回复")
        except Exception as e:
            return f"线上模型请求失败: {e}"

