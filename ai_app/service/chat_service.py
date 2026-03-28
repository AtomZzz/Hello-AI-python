# chat_service.py
from ai_app.llm.ollama_client import OllamaClient
from ai_app.prompt.templates import build_prompt, DEFAULT_SYSTEM_PROMPT

class ChatService:
    def __init__(self, model=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.ollama = OllamaClient()
        self.system_prompt = system_prompt
        self.model = model or self._get_default_model()

    def _get_default_model(self):
        models = self.ollama.list_models()
        if not models:
            raise RuntimeError("未检测到本地模型，请先用 ollama pull 下载模型。")
        return models[0]

    def chat(self, user_input):
        prompt = build_prompt(user_input, self.system_prompt)
        return self.ollama.generate(prompt, self.model)

