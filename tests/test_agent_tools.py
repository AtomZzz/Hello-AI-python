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

    def test_summarize_text_for_structured_analysis(self):
        analysis = {
            "line_count": 9,
            "levels": {"INFO": 1, "WARN": 3, "ERROR": 5},
            "error_count": 5,
            "error_samples": ["ERROR deadlock found"],
            "first_timestamp": "2024-04-09 10:00:00",
            "last_timestamp": "2024-04-09 10:00:40",
            "has_traceback": False,
        }
        summary = summarize_text(analysis)
        self.assertIn("共解析 9 行日志", summary)
        self.assertIn("ERROR:5", summary)
        self.assertIn("Traceback=否", summary)


if __name__ == "__main__":
    unittest.main()

