import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple


class SimpleVectorStore:
	"""Lightweight FAISS vector store backed by sentence-transformers embeddings."""

	def __init__(
		self,
		knowledge_path: str,
		model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
		cache_path: Optional[str] = None,
	):
		self.knowledge_path = knowledge_path
		self.model_name = model_name
		self.cache_path = cache_path or os.path.join(os.path.dirname(knowledge_path), "embeddings.pkl")
		self.embedding_model = None
		self.documents: List[str] = []
		self.index = None
		self._faiss = None

	def _read_knowledge(self) -> str:
		if not os.path.exists(self.knowledge_path):
			return ""
		with open(self.knowledge_path, "r", encoding="utf-8") as f:
			return f.read().strip()

	def _split_documents(self, raw: str) -> List[str]:
		if not raw:
			return []
		# Prefer paragraph split; fallback to line split when no blank line exists.
		chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
		if len(chunks) <= 1:
			chunks = [line.strip() for line in raw.splitlines() if line.strip()]
		return chunks

	def _build_fingerprint(self, raw: str) -> str:
		content = f"{self.model_name}\n{raw}".encode("utf-8")
		return hashlib.sha256(content).hexdigest()

	def _load_cache(self, fingerprint: str) -> Optional[Tuple[List[str], Any]]:
		if not os.path.exists(self.cache_path):
			return None
		try:
			with open(self.cache_path, "rb") as f:
				payload = pickle.load(f)
		except Exception:
			return None
		if payload.get("fingerprint") != fingerprint:
			return None
		documents = payload.get("documents") or []
		embeddings = payload.get("embeddings")
		if not documents or embeddings is None:
			return None
		return documents, embeddings

	def _save_cache(self, fingerprint: str, documents: List[str], embeddings: Any) -> None:
		payload = {
			"fingerprint": fingerprint,
			"documents": documents,
			"embeddings": embeddings,
		}
		with open(self.cache_path, "wb") as f:
			pickle.dump(payload, f)

	def load(self) -> None:
		raw = self._read_knowledge()
		self.documents = self._split_documents(raw)
		if not self.documents:
			return

		# Lazy import to avoid heavy startup cost when RAG route is not used.
		try:
			import faiss
			from sentence_transformers import SentenceTransformer
		except ImportError as e:  # pragma: no cover
			raise RuntimeError("缺少 RAG 依赖，请安装 sentence-transformers 和 faiss-cpu。") from e

		self._faiss = faiss
		fingerprint = self._build_fingerprint(raw)
		cached = self._load_cache(fingerprint)
		if cached:
			self.documents, embeddings = cached
		else:
			self.embedding_model = SentenceTransformer(self.model_name)
			embeddings = self.embedding_model.encode(
				self.documents,
				convert_to_numpy=True,
				normalize_embeddings=True,
			).astype("float32")
			self._save_cache(fingerprint, self.documents, embeddings)

		if self.embedding_model is None:
			self.embedding_model = SentenceTransformer(self.model_name)

		dim = embeddings.shape[1]
		self.index = self._faiss.IndexFlatIP(dim)
		self.index.add(embeddings)

	def _ensure_loaded(self) -> None:
		if self.index is None:
			self.load()

	def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
		self._ensure_loaded()
		if self.index is None or not self.documents:
			return []

		top_k = min(max(top_k, 1), len(self.documents))
		query_vector = self.embedding_model.encode(
			[query],
			convert_to_numpy=True,
			normalize_embeddings=True,
		).astype("float32")

		scores, indices = self.index.search(query_vector, top_k)
		results: List[Dict[str, Any]] = []
		for score, idx in zip(scores[0], indices[0]):
			if idx < 0:
				continue
			results.append(
				{
					"text": self.documents[int(idx)],
					"score": float(score),
				}
			)
		return results

