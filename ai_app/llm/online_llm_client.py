# online_llm_client.py
import logging

import requests
from ai_app.llm.base_llm_client import BaseLLMClient


logger = logging.getLogger(__name__)


class OnlineLLMClient(BaseLLMClient):
    def __init__(self, api_url, api_key=None, timeout=60):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def list_models(self):
        # 假设线上API有模型列表接口，否则可返回默认模型
        # return ["online-model-1", "online-model-2"]
        return ["online-model"]

    def generate(self, messages, model):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            if isinstance(messages, str):
                payload = {"model": model, "prompt": messages}
                logger.info("Online generate request: model=%s timeout=%ss", model, self.timeout)
                resp = requests.post(
                    f"{self.api_url}/generate",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "无回复")

            # Compatible with OpenAI-style chat completion endpoints.
            payload = {"model": model, "messages": messages}
            logger.info("Online chat request: model=%s timeout=%ss", model, self.timeout)
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "无回复")
            return data.get("response", "无回复")
        except requests.Timeout:
            logger.warning("Online request timed out: model=%s timeout=%ss", model, self.timeout)
            return f"线上模型请求超时: 超过{self.timeout}秒未返回"
        except Exception as e:
            logger.exception("Online generate failed: %s", e)
            return f"线上模型请求失败: {e}"

