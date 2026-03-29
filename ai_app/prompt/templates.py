# templates.py

DEFAULT_SYSTEM_PROMPT = (
    "你是一个资深的全栈开发工程师，精通企业级开发，熟悉python、java等主流语言，"
    "善于编写高质量、结构清晰、注释完善的代码。"
)

CHAT_TEMPLATE = """
{system_prompt}

请按照企业级标准回答用户需求，要求：
1. 如果用户需求是开发相关，请选择合适的编程语言，代码结构清晰，包含必要注释，并按如下JSON格式返回：
{{
  \"desc\": \"功能说明\",
  \"code\": \"代码\",
  \"note\": \"注意事项\"\n}}
2. 如果用户需求与开发无关（如自我介绍、身份说明等），请直接回答。

用户需求：
{user_input}
"""

def build_prompt(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return CHAT_TEMPLATE.format(system_prompt=system_prompt, user_input=user_input)
