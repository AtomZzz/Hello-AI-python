from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

from ai_app.rag.vector_store import SimpleVectorStore


logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        knowledge_path: str = None,
        top_k: int = 3,
        score_threshold: float = 0.35,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        default_knowledge_path = str(base_dir / "rag" / "knowledge.txt")
        self.knowledge_path = knowledge_path or default_knowledge_path
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.vector_store = SimpleVectorStore(self.knowledge_path)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        try:
            results = self.vector_store.search(query, top_k=self.top_k)
        except RuntimeError as e:
            logger.warning("RAG retrieve skipped: %s", e)
            return []
        return [item for item in results if item["score"] >= self.score_threshold]

    def build_augmented_input(self, user_input: str) -> Tuple[str, List[Dict[str, Any]]]:
        hits = self.retrieve(user_input)
        if not hits:
            return user_input, []

        context = "\n".join(
            [f"[{idx + 1}] score={hit['score']:.4f} | {hit['text']}" for idx, hit in enumerate(hits)]
        )
        augmented_input = (
            "你必须严格基于以下资料回答：\n"
            "- 不允许编造\n"
            "- 如果资料不足，必须回答\"资料中未提及\"\n"
            "- 不要补充外部事实\n\n"
            f"内部资料:\n{context}\n\n"
            f"用户问题:\n{user_input}"
        )
        return augmented_input, hits

