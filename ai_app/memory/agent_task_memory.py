# agent_task_memory.py — 向量任务记忆（FAISS + sentence-transformers）
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_embedding_document(user_question: str, final_answer: str, max_answer_chars: int = 2000) -> str:
    ans = (final_answer or "").strip()
    if len(ans) > max_answer_chars:
        ans = ans[: max_answer_chars - 20] + "…[结论已截断]"
    return f"用户任务：{user_question.strip()}\nAgent结论摘要：{ans}"


def augment_user_input_with_memory_hits(user_input: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return user_input
    lines = [
        "## 历史相似任务（向量检索，仅供参考）",
        "以下条目与当前问题在语义上较接近，可类比排查思路；事实仍以当前请求与工具 Observation 为准。",
        "",
    ]
    for i, h in enumerate(hits, start=1):
        score = float(h.get("score") or 0.0)
        doc = (h.get("text") or "").strip()
        lines.append(f"[{i}] similarity={score:.4f}")
        lines.append(doc)
        lines.append("")
    lines.append("---")
    lines.append("## 当前用户请求")
    lines.append(user_input.strip())
    return "\n".join(lines).strip()


class AgentTaskMemory:
    """增量写入的 FAISS 索引；每条记录为「任务+结论摘要」向量，用于检索相似历史任务。"""

    META_NAME = "meta.pkl"
    INDEX_NAME = "index.faiss"

    def __init__(
        self,
        persist_dir: str,
        model_name: str = _DEFAULT_MODEL,
        top_k: int = 5,
        score_threshold: float = 0.32,
        encoder=None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._encoder = encoder
        self._model = None
        self._faiss = None
        self._index = None
        self._records: List[Dict[str, Any]] = []
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()
        self._load_from_disk()

    def _ensure_faiss(self) -> None:
        if self._faiss is None:
            import faiss

            self._faiss = faiss

    def _get_sentence_model(self):
        if self._encoder is not None:
            return self._encoder
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: List[str]):
        model = self._get_sentence_model()
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def _load_from_disk(self) -> None:
        index_path = self.persist_dir / self.INDEX_NAME
        meta_path = self.persist_dir / self.META_NAME
        if not index_path.exists() or not meta_path.exists():
            return
        try:
            self._ensure_faiss()
            self._index = self._faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._records = list(meta.get("records") or [])
            self._dimension = meta.get("dimension")
            if self._dimension is None and self._index.ntotal > 0:
                self._dimension = int(self._index.d)
            self.model_name = str(meta.get("model_name") or self.model_name)
            if len(self._records) != int(self._index.ntotal):
                logger.warning(
                    "AgentTaskMemory: meta/index 行数不一致 (records=%s ntotal=%s)，已重置",
                    len(self._records),
                    self._index.ntotal,
                )
                self._records = []
                self._index = None
                self._dimension = None
        except Exception as exc:  # pragma: no cover - 损坏文件
            logger.warning("AgentTaskMemory: 加载失败，将使用空索引: %s", exc)
            self._records = []
            self._index = None
            self._dimension = None

    def _save_to_disk_unlocked(self) -> None:
        if self._index is None:
            return
        self._ensure_faiss()
        index_path = self.persist_dir / self.INDEX_NAME
        meta_path = self.persist_dir / self.META_NAME
        self._faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "records": self._records,
                    "dimension": self._dimension,
                    "model_name": self.model_name,
                },
                f,
            )

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            k_req = min(top_k or self.top_k, int(self._index.ntotal))
            vec = self._encode([q])
            scores, indices = self._index.search(vec, k_req)
            out: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if float(score) < self.score_threshold:
                    continue
                rec = self._records[int(idx)]
                text = str(rec.get("text") or "")
                meta = {k: v for k, v in rec.items() if k != "text"}
                out.append({"text": text, "score": float(score), "meta": meta})
            return out

    def add(self, document_text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        text = (document_text or "").strip()
        if not text:
            return
        row = {"text": text, **(meta or {})}
        with self._lock:
            vec = self._encode([text])
            dim = int(vec.shape[1])
            self._ensure_faiss()
            if self._index is None:
                self._dimension = dim
                self._index = self._faiss.IndexFlatIP(dim)
            elif int(self._index.d) != dim:
                logger.error("AgentTaskMemory: 向量维度不一致，跳过写入")
                return
            self._index.add(vec)
            self._records.append(row)
            self._save_to_disk_unlocked()

    def add_from_agent_run(self, original_user_input: str, agent_output_text: str) -> None:
        try:
            data = json.loads(agent_output_text)
        except Exception:
            logger.debug("AgentTaskMemory: 输出非 JSON，跳过写入")
            return
        final = str(data.get("final_answer") or "").strip()
        if not final:
            return
        orig = (original_user_input or "").strip()
        if not orig:
            return
        doc = build_embedding_document(orig, final)
        self.add(doc, {"user_input": orig, "final_answer": final})
