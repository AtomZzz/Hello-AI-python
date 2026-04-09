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

    def test_summarize_text_returns_short_text(self):
        content = "line1\nline2\nline3\nline4"
        summary = summarize_text(content)
        self.assertIn("其余内容已省略", summary)


if __name__ == "__main__":
    unittest.main()

