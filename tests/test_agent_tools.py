import unittest

from ai_app.agent.tools import analyze_log, summarize_text


class AgentToolsTest(unittest.TestCase):
    def test_analyze_log_extracts_error_info(self):
        log_text = """
        2026-04-09 10:00:00 INFO app started
        2026-04-09 10:00:01 ERROR database connection failed
        Traceback (most recent call last):
        ValueError: invalid input
        """
        result = analyze_log(log_text)
        self.assertEqual(result["line_count"], 4)
        self.assertGreaterEqual(result["error_count"], 2)
        self.assertTrue(result["has_traceback"])
        self.assertIn("ERROR", result["levels"])
        self.assertTrue(result["root_cause"])
        self.assertTrue(result["issue_clusters"])

    def test_analyze_log_extracts_diagnostic_root_causes(self):
        log_text = """
        2024-04-09 10:00:05 [WARNING] Slow query detected: SELECT * FROM users WHERE last_login < '2023-01-01' (execution time: 12.5s)
        2024-04-09 10:00:10 [ERROR] Deadlock found when trying to get lock; try restarting transaction
        2024-04-09 10:00:15 [ERROR] Connection timeout: Too many connections (max_connections: 151)
        2024-04-09 10:00:20 [WARNING] Table 'cache_table' is marked as crashed and should be repaired
        """
        result = analyze_log(log_text)
        self.assertGreaterEqual(len(result["root_cause"]), 1)
        self.assertIn("高频异常模式", result["root_cause"][0])
        self.assertTrue(result["evidence"])
        self.assertTrue(result["next_actions"])

    def test_summarize_text_prefers_diagnosis(self):
        analysis = {
            "root_cause": ["数据库连接数不足（max_connections）", "存在死锁竞争", "表损坏（cache_table）"],
            "evidence": ["数据库连接数不足（max_connections）（证据: 2024-04-09 10:00:15 [ERROR] Connection timeout: Too many connections (max_connections: 151)）"],
            "next_actions": ["检查连接池配置与慢 SQL，必要时提升 max_connections 并释放空闲连接。"],
            "first_timestamp": "2024-04-09 10:00:00",
            "last_timestamp": "2024-04-09 10:00:40",
            "error_count": 5,
        }
        summary = summarize_text(analysis)
        self.assertEqual(summary["severity"], "P2")
        self.assertIn("数据库连接数不足（max_connections）", summary["root_cause"])
        self.assertTrue(summary["key_evidence"])
        self.assertTrue(summary["overview"])

    def test_analyze_log_ignores_instruction_header_line(self):
        log_text = (
            "帮我分析如下错误日志：\n"
            "2026-04-09 10:00:00 [INFO] start\n"
            "2026-04-09 10:00:01 [ERROR] connect timeout"
        )
        result = analyze_log(log_text)
        self.assertEqual(result["line_count"], 2)
        self.assertGreaterEqual(result["error_count"], 1)


if __name__ == "__main__":
    unittest.main()

