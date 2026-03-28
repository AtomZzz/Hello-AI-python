# templates.py

DEFAULT_SYSTEM_PROMPT = "你是一个资深的全栈开发工程师"

CHAT_TEMPLATE = """
系统提示: {system_prompt}
用户: {user_input}
"""

def build_prompt(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return CHAT_TEMPLATE.format(system_prompt=system_prompt, user_input=user_input)

