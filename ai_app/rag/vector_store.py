import os
from typing import Any, Dict, List

try:
	import faiss
except ImportError:  # pragma: no cover
	faiss = None

try:
	from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
	SentenceTransformer = None


class SimpleVectorStore:
	"""Lightweight FAISS vector store backed by sentence-transformers embeddings."""

	def __init__(
		self,
		knowledge_path: str,
		model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
	):
		self.knowledge_path = knowledge_path
		self.model_name = model_name
		self.embedding_model = None
		self.documents: List[str] = []
		self.index = None

	def _load_documents(self) -> List[str]:
		if not os.path.exists(self.knowledge_path):
			return []

		with open(self.knowledge_path, "r", encoding="utf-8") as f:
			raw = f.read().strip()

		if not raw:
			return []

		# Prefer paragraph split; fallback to line split when no blank line exists.
		chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
		if len(chunks) <= 1:
			chunks = [line.strip() for line in raw.splitlines() if line.strip()]
		return chunks

	def load(self) -> None:
		self.documents = self._load_documents()
		if not self.documents:
			return

		if faiss is None or SentenceTransformer is None:
			raise RuntimeError(
				"缺少 RAG 依赖，请安装 sentence-transformers 和 faiss-cpu。"
			)

		self.embedding_model = SentenceTransformer(self.model_name)
		embeddings = self.embedding_model.encode(
			self.documents,
			convert_to_numpy=True,
			normalize_embeddings=True,
		).astype("float32")

		dim = embeddings.shape[1]
		self.index = faiss.IndexFlatIP(dim)
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

