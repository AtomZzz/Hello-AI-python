import json
import tempfile
import unittest

import numpy as np

from ai_app.memory.agent_task_memory import (
    AgentTaskMemory,
    augment_user_input_with_memory_hits,
    build_embedding_document,
)


class UniformEncoder:
    """固定单位向量，便于在无 sentence-transformers 的环境下测 FAISS 逻辑。"""

    dim = 8

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        v = np.zeros(self.dim, dtype=np.float32)
        v[0] = 1.0
        v /= np.linalg.norm(v) + 1e-9
        return np.tile(v, (len(texts), 1))


class AgentTaskMemoryTest(unittest.TestCase):
    def test_build_embedding_document_truncates_long_answer(self):
        long_ans = "x" * 5000
        doc = build_embedding_document("问", long_ans, max_answer_chars=100)
        self.assertIn("用户任务：问", doc)
        self.assertLess(len(doc), len(long_ans))
        self.assertIn("…[结论已截断]", doc)

    def test_augment_user_input_inserts_hits(self):
        hits = [{"text": "历史A", "score": 0.91}]
        out = augment_user_input_with_memory_hits("当前问题", hits)
        self.assertIn("历史相似任务", out)
        self.assertIn("历史A", out)
        self.assertIn("当前问题", out)
        self.assertIn("当前用户请求", out)

    def test_search_and_persist_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = UniformEncoder()
            m1 = AgentTaskMemory(
                tmp,
                top_k=3,
                score_threshold=0.9,
                encoder=enc,
            )
            m1.add("用户任务：数据库连接超时\nAgent结论摘要：检查连接池", {"user_input": "q1"})
            m1.add("用户任务：Redis 超时\nAgent结论摘要：检查网络", {"user_input": "q2"})
            hits = m1.search("任意查询", top_k=2)
            self.assertGreaterEqual(len(hits), 1)
            self.assertGreaterEqual(hits[0]["score"], 0.99)

            m2 = AgentTaskMemory(
                tmp,
                top_k=3,
                score_threshold=0.9,
                encoder=enc,
            )
            hits2 = m2.search("第二次查询", top_k=2)
            self.assertEqual(len(hits2), len(hits))

    def test_add_from_agent_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = AgentTaskMemory(tmp, top_k=3, score_threshold=0.0, encoder=UniformEncoder())
            payload = json.dumps({"final_answer": "根因是端口占用"}, ensure_ascii=False)
            m.add_from_agent_run("分析日志 connection refused", payload)
            self.assertEqual(len(m._records), 1)
            self.assertIn("connection refused", m._records[0]["text"])

    def test_score_threshold_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = UniformEncoder()
            m = AgentTaskMemory(tmp, top_k=3, score_threshold=1.00001, encoder=enc)
            m.add("doc", {})
            hits = m.search("x")
            self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
