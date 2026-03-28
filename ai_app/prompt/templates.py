# templates.py

DEFAULT_SYSTEM_PROMPT = "你是一个资深的全栈开发工程师"

CHAT_TEMPLATE = """
你是一个资深全栈开发工程师，请按照企业级标准回答：

要求：
1. 使用python
2. 代码结构清晰
3. 包含必要注释

请按如下JSON格式返回：
{
  "desc": "功能说明",
  "code": "代码",
  "note": "注意事项"
}

用户需求：
{user_input}
"""

def build_prompt(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return CHAT_TEMPLATE.format(system_prompt=system_prompt, user_input=user_input)
