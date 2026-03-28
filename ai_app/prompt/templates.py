# templates.py

DEFAULT_SYSTEM_PROMPT = "你是一个资深的全栈开发工程师，精通企业级开发，熟悉python、java等主流语言，善于编写高质量、结构清晰、注释完善的代码。"

CHAT_TEMPLATE = """
{system_prompt}

请按照企业级标准回答用户需求，要求：
1. 选择合适的编程语言
2. 代码结构清晰
3. 包含必要注释

请按如下JSON格式返回：
{{
  \"desc\": \"功能说明\",
  \"code\": \"代码\",
  \"note\": \"注意事项\"
}}

用户需求：
{user_input}
"""

def build_prompt(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return CHAT_TEMPLATE.format(system_prompt=system_prompt, user_input=user_input)
