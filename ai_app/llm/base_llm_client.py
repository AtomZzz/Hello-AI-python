# base_llm_client.py
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    @abstractmethod
    def list_models(self):
        pass

    @abstractmethod
    def generate(self, messages, model):
        pass

