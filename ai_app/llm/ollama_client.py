# ollama_client.py
import logging

import requests
from ai_app.llm.base_llm_client import BaseLLMClient


logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    def __init__(self, base_url="http://localhost:11434", timeout=60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def check_root(self):
        try:
            resp = requests.get(f"{self.base_url}/", timeout=self.timeout)
            return resp.status_code == 200
        except Exception as e:
            logger.warning("Ollama root health check failed: %s", e)
            return False

    def check_tags(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            return resp.status_code == 200
        except Exception as e:
            logger.warning("Ollama tags health check failed: %s", e)
            return False

    def health(self):
        return self.check_tags()

    def list_models(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                models = [m['name'] for m in data.get('models', [])]
                logger.debug("Ollama models loaded: %s", models)
                return models
            return []
        except Exception as e:
            logger.exception("Ollama list_models failed: %s", e)
            return []

    def generate(self, messages, model):
        try:
            # Backward compatibility for legacy string prompt call sites.
            if isinstance(messages, str):
                url = f"{self.base_url}/api/generate"
                payload = {"model": model, "prompt": messages, "stream": False}
                logger.info("Ollama generate request: model=%s timeout=%ss", model, self.timeout)
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "无回复")

            url = f"{self.base_url}/api/chat"
            payload = {"model": model, "messages": messages, "stream": False}
            logger.info("Ollama chat request: model=%s timeout=%ss", model, self.timeout)
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "无回复")
        except requests.Timeout:
            logger.warning("Ollama request timed out: model=%s timeout=%ss", model, self.timeout)
            return f"请求超时: 超过{self.timeout}秒未返回"
        except Exception as e:
            logger.exception("Ollama generate failed: %s", e)
            return f"请求失败: {e}"
