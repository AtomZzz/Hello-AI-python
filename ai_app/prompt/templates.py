# templates.py

DEFAULT_SYSTEM_PROMPT = (
    "你是一个资深的全栈开发工程师，精通企业级开发，熟悉python、java等主流语言，"
    "善于编写高质量、结构清晰、注释完善的代码。"
)

DEV_SYSTEM_INSTRUCTION_TEMPLATE = """
{system_prompt}

请按照企业级标准回答用户需求，要求：
1. 这是开发类需求，请选择合适的编程语言，代码结构清晰，包含必要注释。
2. 必须严格按如下JSON格式返回，且只返回一个JSON对象：
{{
  \"desc\": \"功能说明\",
  \"code\": \"代码\",
  \"note\": \"注意事项\"
}}
3. 不要输出 Markdown 代码块，不要输出 JSON 之外的任何说明文字。
"""

GENERAL_SYSTEM_INSTRUCTION_TEMPLATE = """
{system_prompt}

请按照企业级标准回答用户需求，表达清晰、准确、简洁。
"""


def build_messages(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT, require_json=False):
    template = DEV_SYSTEM_INSTRUCTION_TEMPLATE if require_json else GENERAL_SYSTEM_INSTRUCTION_TEMPLATE
    system_content = template.format(system_prompt=system_prompt).strip()
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]


