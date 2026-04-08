# chat_service.py
import json
import logging
import time
import uuid

from ai_app.llm.ollama_client import OllamaClient
from ai_app.llm.online_llm_client import OnlineLLMClient
from ai_app.llm.qwen_client import QwenLLMClient
from ai_app.prompt.templates import (
    DEFAULT_SYSTEM_PROMPT,
    build_json_repair_messages,
    build_messages,
    build_router_messages,
)
from ai_app.parser.json_parser import JsonParser


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        model=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        llm_type="ollama",
        online_conf=None,
        rag_enabled=True,
    ):
        self.system_prompt = system_prompt
        self.llm_type = llm_type
        self.online_conf = online_conf or {}
        self.rag_enabled = rag_enabled
        self.llm = self._get_llm_client()
        self.model = model or self._get_default_model()
        self.json_parser = JsonParser()
        self.route_parser = JsonParser(required_keys=["use_rag", "require_json", "reason"])
        self.rag_service = None

    def _get_rag_service(self):
        if not self.rag_enabled:
            return None
        if self.rag_service is None:
            from ai_app.service.rag_service import RagService

            self.rag_service = RagService()
        return self.rag_service

    def _get_llm_client(self):
        timeout = self.online_conf.get("timeout", 60)
        if self.llm_type == "ollama":
            return OllamaClient(timeout=timeout)
        elif self.llm_type == "online":
            return OnlineLLMClient(
                api_url=self.online_conf.get("api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                api_key=self.online_conf.get("api_key"),
                timeout=timeout,
            )
        elif self.llm_type == "qwen":
            return QwenLLMClient(
                api_key=self.online_conf.get("api_key"),
                base_url=self.online_conf.get("api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                model_name=self.online_conf.get("model_name", "qwen-plus"),
                timeout=timeout,
            )
        else:
            raise ValueError(f"不支持的llm_type: {self.llm_type}")

    def _get_default_model(self):
        models = self.llm.list_models()
        if not models:
            raise RuntimeError("未检测到可用模型，请检查配置。")
        return models[0]

    def _is_dev_request(self, user_input):
        text = (user_input or "").lower()
        keywords = [
            "代码", "示例", "实现", "开发", "接口", "类", "函数", "重构", "优化", "修复",
            "python", "java", "spring", "sql", "api", "bug", "hello"
        ]
        return any(k in text for k in keywords)

    def _looks_like_rag_request(self, user_input):
        text = (user_input or "").lower()
        keywords = ["atom", "深圳", "邮箱", "网名", "内部资料", "个人信息", "联系方式", "email"]
        return any(k in text for k in keywords)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return False

    def _route_request(self, user_input, use_model):
        route = {
            "use_rag": self._looks_like_rag_request(user_input),
            "require_json": self._is_dev_request(user_input),
            "reason": "heuristic-fallback",
        }

        try:
            router_messages = build_router_messages(user_input)
            raw_route = self.llm.generate(router_messages, use_model)
            parsed = self.route_parser.parse(raw_route)
            if parsed:
                route["use_rag"] = self._to_bool(parsed.get("use_rag"))
                route["require_json"] = self._to_bool(parsed.get("require_json"))
                route["reason"] = parsed.get("reason") or "ai-router"
        except Exception as e:
            logger.warning("AI router failed, fallback to heuristic: %s", e)

        if not self.rag_enabled:
            route["use_rag"] = False

        # RAG answers should also be JSON-controlled.
        if route["use_rag"]:
            route["require_json"] = True
        return route

    def _repair_json_with_llm(self, raw_reply, use_model):
        repair_messages = build_json_repair_messages(raw_reply, self.json_parser.required_keys)
        repaired_reply = self.llm.generate(repair_messages, use_model)
        return self.json_parser.parse(repaired_reply)

    def _build_dev_fallback_json(self, raw_reply):
        return {
            "desc": "模型未按JSON格式返回，已自动进行兜底封装",
            "code": raw_reply.strip(),
            "note": "建议保留system role约束并继续优化提示词。"
        }

    def _print_rag_hits(self, hits):
        if not hits:
            print("[RAG] 未检索到命中内容")
            return
        print("[RAG] 检索命中内容:")
        for idx, hit in enumerate(hits, start=1):
            print(f"  {idx}. score={hit['score']:.4f} | {hit['text']}")

    def chat(self, user_input, model=None):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        use_model = model or self.model
        route = self._route_request(user_input, use_model)
        logger.info(
            "Route decided request_id=%s use_rag=%s require_json=%s reason=%s",
            request_id,
            route["use_rag"],
            route["require_json"],
            route.get("reason", ""),
        )

        input_for_llm = user_input
        if route["use_rag"]:
            try:
                rag_service = self._get_rag_service()
                input_for_llm, hits = rag_service.build_augmented_input(user_input)
                logger.info("RAG route enabled request_id=%s hit_count=%s", request_id, len(hits))
                self._print_rag_hits(hits)
            except Exception as e:
                logger.warning(
                    "RAG disabled request_id=%s due to error (%s). Fallback to direct LLM.",
                    request_id,
                    e,
                )
                input_for_llm = user_input

        messages = build_messages(input_for_llm, self.system_prompt, require_json=route["require_json"])
        logger.info(
            "Chat start request_id=%s llm_type=%s model=%s dev_mode=%s",
            request_id,
            self.llm_type,
            use_model,
            route["require_json"],
        )
        raw_reply = self.llm.generate(messages, use_model)

        # JSON控制流程: parse -> LLM修复 -> parse -> fallback
        if route["require_json"]:
            json_data = self.json_parser.parse(raw_reply)
            if not json_data:
                logger.warning("JSON parse failed request_id=%s, try LLM repair", request_id)
                json_data = self._repair_json_with_llm(raw_reply, use_model)
            if not json_data:
                logger.warning("JSON repair failed request_id=%s, using fallback wrapper", request_id)
                json_data = self._build_dev_fallback_json(raw_reply)
            output = json.dumps(json_data, ensure_ascii=False, indent=2)
            elapsed = time.perf_counter() - start
            logger.info("Chat done request_id=%s elapsed=%.2fs", request_id, elapsed)
            return output

        # 非开发请求保持自然文本输出
        elapsed = time.perf_counter() - start
        logger.info("Chat done request_id=%s elapsed=%.2fs", request_id, elapsed)
        return raw_reply
