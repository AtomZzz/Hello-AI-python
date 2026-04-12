# templates.py

DEFAULT_SYSTEM_PROMPT = (
    "你是一个资深的全栈开发工程师，精通企业级开发，熟悉python、java等主流语言，"
    "善于编写高质量、结构清晰、注释完善的代码。"
)

ROUTER_SYSTEM_INSTRUCTION_TEMPLATE = """
你是请求路由器。请分析用户问题并严格返回一个JSON对象：
{{
  "use_agent": true/false,
  "use_rag": true/false,
  "require_json": true/false,
  "reason": "简短原因"
}}

判定规则：
1. 请基于语义进行判断，不要依赖固定关键词匹配。
2. 当问题是日志分析、错误定位、异常排查、堆栈解读等需要工具执行的任务时，use_agent=true。
3. 当问题可能依赖内部资料、个人档案、公司知识库、指定文档事实时，use_rag=true。
4. 当问题是开发实现、代码示例、脚本编写、架构设计、接口定义、重构优化、调试修复等技术任务时，require_json=true。
5. 普通聊天、自我介绍、开放问答、非技术闲聊通常 require_json=false。
6. 当 use_agent=true 时，use_rag=false（优先走 Agent 工具链）。
7. 只能输出JSON，不要输出任何额外文字。

示例：
- "帮我分析这段报错日志" -> {"use_agent": true, "use_rag": false, "require_json": true, "reason": "日志分析适合Agent工具链"}
- "Atom是谁" -> {"use_agent": false, "use_rag": true, "require_json": true, "reason": "需要基于内部资料回答"}
- "写一个hello world" -> {"use_agent": false, "use_rag": false, "require_json": true, "reason": "明显是代码生成需求"}
- "介绍一下你自己" -> {"use_agent": false, "use_rag": false, "require_json": false, "reason": "普通问答"}
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

LOG_SUMMARY_SYSTEM_INSTRUCTION_TEMPLATE = """
你是企业级日志诊断专家。请基于用户提供的原始日志与规则分析结果，输出一个稳定的JSON对象。

必须严格返回如下结构，且只能返回JSON：
{{
  "overview": "整体诊断结论",
  "severity": "P1|P2|P3|INFO",
  "root_cause": ["根因1", "根因2"],
  "key_evidence": ["关键证据1", "关键证据2"],
  "next_actions": ["建议动作1", "建议动作2"],
  "confidence": "high|medium|low"
}}

要求：
1. 优先基于日志事实输出，不要编造不存在的组件或报错。
2. 如果规则分析结果不足，请结合原始日志进行更智能的归纳。
3. root_cause、key_evidence、next_actions 必须是数组。
4. overview 必须是面向排障场景的中文结论，不要写成统计结果。
5. 不要输出 Markdown，不要输出解释文字。
"""


def build_agent_prompt(user_input, tool_specs=None, scratchpad=""):
    tool_specs = tool_specs or [
        "a. analyze_log(text) - 分析日志",
        "b. summarize_text(text) - 文本总结",
    ]
    tools_section = "\n".join(tool_specs)
    scratchpad = (scratchpad or "").strip()
    scratchpad_section = (
        f"\n\n以下是你之前的推理与工具执行记录，请基于它继续：\n{scratchpad}\n"
        if scratchpad
        else ""
    )
    return f"""
你是一个AI Agent，可以使用以下工具：

{tools_section}

你必须按ReAct流程工作，每一轮只做一件事：
- 如果需要调用工具，必须输出：
Thought: 你的思考
Action: 工具名
Action Input: 输入
- 如果你已经可以给出最终结论，输出：
Thought: 你的思考
Final Answer: 最终答案

注意：
1. 不要伪造 Observation，Observation 由系统返回。
2. 一次响应中不要同时给出 Action 和 Final Answer。
3. 优先使用工具结果进行结论。
4. 当 Observation.type=analysis_result 时，如需总结请调用 summarize_text，输入应基于 analysis_result.data。
5. Final Answer 必须严格基于已有 Observation，禁止引入 Observation 中未出现的新组件/新系统/新错误名。

用户问题：
{user_input}

{scratchpad_section}
"""


def build_agent_messages(user_input, tool_specs=None):
    return [{"role": "user", "content": build_agent_prompt(user_input, tool_specs=tool_specs).strip()}]


def build_agent_step_messages(user_input, tool_specs=None, scratchpad=""):
    return [
        {
            "role": "user",
            "content": build_agent_prompt(
                user_input,
                tool_specs=tool_specs,
                scratchpad=scratchpad,
            ).strip(),
        }
    ]


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


def build_log_summary_messages(user_input, analysis):
    return [
        {"role": "system", "content": LOG_SUMMARY_SYSTEM_INSTRUCTION_TEMPLATE.strip()},
        {
            "role": "user",
            "content": (
                "用户原始问题/日志如下：\n"
                f"{user_input}\n\n"
                "当前规则分析结果如下：\n"
                f"{analysis}"
            ),
        },
    ]


