import json
from typing import Any, Dict, Optional

class JsonParser:
    """
    安全解析和修复大模型输出的 JSON 字符串，保证输出结构稳定。
    """
    def __init__(self, required_keys=None):
        self.required_keys = required_keys or ["desc", "code", "note"]

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        尝试解析并修复大模型输出的 JSON 字符串。
        :param text: 大模型原始输出
        :return: 解析后的 dict 或 None
        """
        # 1. 提取 JSON 代码块
        json_str = self._extract_json(text)
        if not json_str:
            return None
        # 2. 尝试解析
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试修复常见问题
            json_str = self._auto_fix(json_str)
            try:
                data = json.loads(json_str)
            except Exception:
                return None
        # 3. 校验结构
        if not self._validate(data):
            return None
        return data

    def _extract_json(self, text: str) -> Optional[str]:
        """
        提取文本中的 JSON 代码块。
        """
        import re
        # 匹配 ```json ... ``` 或 {...}
        match = re.search(r"```json([\s\S]*?)```", text)
        if match:
            return match.group(1).strip()
        match = re.search(r"({[\s\S]*})", text)
        if match:
            return match.group(1).strip()
        return None

    def _auto_fix(self, json_str: str) -> str:
        """
        尝试修复常见的 JSON 格式问题。
        """
        # 替换单引号为双引号
        json_str = json_str.replace("'", '"')
        # 去除多余逗号
        import re
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        return json_str

    def _validate(self, data: Any) -> bool:
        """
        校验 JSON 是否包含所需字段。
        """
        if not isinstance(data, dict):
            return False
        for key in self.required_keys:
            if key not in data:
                return False
        return True

