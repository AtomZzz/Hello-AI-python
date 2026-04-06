# qwen_client.py
import logging

import requests

from ai_app.llm.base_llm_client import BaseLLMClient


logger = logging.getLogger(__name__)

class QwenLLMClient(BaseLLMClient):
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus", timeout=60):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout

    def list_models(self):
        return [self.model_name]

    def generate(self, messages, model=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": model or self.model_name,
            "messages": messages,
        }
        try:
            logger.info("Qwen chat request: model=%s timeout=%ss", payload["model"], self.timeout)
            resp = requests.post(
                f"{self.base_url}/chat/completions",
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
            logger.warning("Qwen request timed out: model=%s timeout=%ss", payload["model"], self.timeout)
            return f"通义千问请求超时: 超过{self.timeout}秒未返回"
        except requests.HTTPError as e:
            detail = ""
            if e.response is not None:
                try:
                    detail = e.response.text
                except Exception:
                    detail = str(e)
            logger.exception("Qwen API HTTP error: %s", detail)
            return f"通义千问API错误: {detail or e}"
        except Exception as e:
            logger.exception("Qwen request failed: %s", e)
            return f"通义千问请求失败: {e}"

