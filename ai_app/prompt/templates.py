# templates.py

DEFAULT_SYSTEM_PROMPT = "你是一个资深的全栈开发工程师"

CHAT_TEMPLATE = """
你是一个资深Java工程师，请按照企业级标准回答：

要求：
1. 使用Spring Boot
2. 代码结构清晰
3. 包含必要注释

用户需求：
{user_input}
"""

def build_prompt(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return CHAT_TEMPLATE.format(system_prompt=system_prompt, user_input=user_input)
