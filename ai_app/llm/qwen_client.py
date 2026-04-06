# qwen_client.py
from ai_app.llm.base_llm_client import BaseLLMClient
from dashscope import Generation

class QwenLLMClient(BaseLLMClient):
    def __init__(self, api_key):
        self.api_key = api_key
        self.model_name = "qwen-plus"

    def list_models(self):
        return [self.model_name]

    def generate(self, messages, model=None):
        from http import HTTPStatus
        import dashscope
        dashscope.api_key = self.api_key
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            response = Generation.call(
                model=model or self.model_name,
                messages=messages,
                result_format='message'  # ⚠️ 必须加
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                return f"通义千问API错误: {response.status_code} {getattr(response, 'message', '')}"
        except Exception as e:
            return f"通义千问请求失败: {e}"

