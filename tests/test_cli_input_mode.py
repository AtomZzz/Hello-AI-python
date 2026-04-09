import unittest

from ai_app.main import read_user_input


class FakeIO:
    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.outputs = []

    def reader(self, prompt):
        if not self.inputs:
            raise EOFError()
        return self.inputs.pop(0)

    def writer(self, text):
        self.outputs.append(text)


class CliInputModeTest(unittest.TestCase):
    def test_single_line_input(self):
        io = FakeIO(["你好"])
        value = read_user_input(reader=io.reader, writer=io.writer)
        self.assertEqual(value, "你好")

    def test_exit_command(self):
        io = FakeIO(["exit"])
        value = read_user_input(reader=io.reader, writer=io.writer)
        self.assertEqual(value, "__exit__")

    def test_paste_mode_collects_multiple_lines(self):
        io = FakeIO([
            ":paste",
            "2024-04-09 10:00:10 [ERROR] deadlock found",
            "2024-04-09 10:00:12 [WARNING] slow query",
            ":end",
        ])
        value = read_user_input(reader=io.reader, writer=io.writer)
        self.assertIn("deadlock", value)
        self.assertIn("slow query", value)
        self.assertIn("粘贴模式", io.outputs[0])

    def test_paste_mode_can_cancel(self):
        io = FakeIO([":paste", "line1", ":cancel"])
        value = read_user_input(reader=io.reader, writer=io.writer)
        self.assertEqual(value, "")


if __name__ == "__main__":
    unittest.main()

