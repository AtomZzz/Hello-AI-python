# chat_service.py
from dashscope import Generation

from ai_app.llm.base_llm_client import BaseLLMClient
from ai_app.llm.ollama_client import OllamaClient
from ai_app.llm.online_llm_client import OnlineLLMClient
from ai_app.prompt.templates import build_prompt, DEFAULT_SYSTEM_PROMPT
import dashscope

# 通义千问专用 LLM Client
class QwenLLMClient(BaseLLMClient):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model_name = "qwen-plus"

    def list_models(self):
        return [self.model_name]

    def generate(self, prompt, model=None):
        from http import HTTPStatus
        import dashscope

        dashscope.api_key = self.api_key

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            response = Generation.call(
                model=self.model_name,
                messages=messages,
                result_format='message'  # ⚠️ 必须加
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                return f"通义千问API错误: {response.status_code} {getattr(response, 'message', '')}"

        except Exception as e:
            return f"通义千问请求失败: {e}"

class ChatService:
    def __init__(self, model=None, system_prompt=DEFAULT_SYSTEM_PROMPT, llm_type="ollama", online_conf=None):
        self.system_prompt = system_prompt
        self.llm_type = llm_type
        self.online_conf = online_conf or {}
        self.llm = self._get_llm_client()
        self.model = model or self._get_default_model()

    def _get_llm_client(self):
        if self.llm_type == "ollama":
            return OllamaClient()
        elif self.llm_type == "online":
            return OnlineLLMClient(
                api_url=self.online_conf.get("api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                api_key=self.online_conf.get("api_key")
            )
        elif self.llm_type == "qwen":
            return QwenLLMClient(
                api_key=self.online_conf.get("api_key")
            )
        else:
            raise ValueError(f"不支持的llm_type: {self.llm_type}")

    def _get_default_model(self):
        models = self.llm.list_models()
        if not models:
            raise RuntimeError("未检测到可用模型，请检查配置。")
        return models[0]

    def chat(self, user_input, model=None):
        prompt = build_prompt(user_input, self.system_prompt)
        use_model = model or self.model
        return self.llm.generate(prompt, use_model)
