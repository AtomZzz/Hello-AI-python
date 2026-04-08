# templates.py

DEFAULT_SYSTEM_PROMPT = (
    "你是一个资深的全栈开发工程师，精通企业级开发，熟悉python、java等主流语言，"
    "善于编写高质量、结构清晰、注释完善的代码。"
)

ROUTER_SYSTEM_INSTRUCTION_TEMPLATE = """
你是请求路由器。请分析用户问题并严格返回一个JSON对象：
{{
  "use_rag": true/false,
  "require_json": true/false,
  "reason": "简短原因"
}}

判定规则：
1. 请基于语义进行判断，不要依赖固定关键词匹配。
2. 当问题可能依赖内部资料、个人档案、公司知识库、指定文档事实时，use_rag=true。
3. 当问题是开发实现、代码示例、脚本编写、架构设计、接口定义、重构优化、调试修复等技术任务时，require_json=true。
4. 普通聊天、自我介绍、开放问答、非技术闲聊通常 require_json=false。
5. 如果一个问题既涉及内部资料又需要结构化输出，则 use_rag=true 且 require_json=true。
6. 只能输出JSON，不要输出任何额外文字。

示例：
- "Atom是谁" -> {"use_rag": true, "require_json": true, "reason": "需要基于内部资料回答"}
- "写一个hello world" -> {"use_rag": false, "require_json": true, "reason": "明显是代码生成需求"}
- "介绍一下你自己" -> {"use_rag": false, "require_json": false, "reason": "普通问答"}
"""

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

JSON_REPAIR_SYSTEM_INSTRUCTION_TEMPLATE = """
你是JSON修复器。请将输入内容修复成一个合法JSON对象，并且只输出JSON。
必须包含字段: {required_keys}
不要输出 Markdown 代码块，不要输出JSON之外的内容。
"""


def build_messages(user_input, system_prompt=DEFAULT_SYSTEM_PROMPT, require_json=False):
    template = DEV_SYSTEM_INSTRUCTION_TEMPLATE if require_json else GENERAL_SYSTEM_INSTRUCTION_TEMPLATE
    system_content = template.format(system_prompt=system_prompt).strip()
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input}
    ]


def build_router_messages(user_input):
    return [
        {"role": "system", "content": ROUTER_SYSTEM_INSTRUCTION_TEMPLATE.strip()},
        {"role": "user", "content": user_input},
    ]


def build_json_repair_messages(raw_text, required_keys):
    return [
        {
            "role": "system",
            "content": JSON_REPAIR_SYSTEM_INSTRUCTION_TEMPLATE.format(
                required_keys=", ".join(required_keys)
            ).strip(),
        },
        {"role": "user", "content": raw_text},
    ]


