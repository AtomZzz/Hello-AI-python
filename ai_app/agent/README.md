# Agent Module

This module adds a first agent capability for log-analysis tasks.

## Components
- `tools.py`: `analyze_log(text)` and `summarize_text(text)`.
- `executor.py`: `AgentExecutor` that plans tool usage and runs tool chain.

## Enhancements
- AI Router can decide `use_agent=true` and route requests to `AgentExecutor`.
- `AgentExecutor` supports tool registration via `register_tool(name, fn, description)`.
- Agent prompt is generated dynamically from registered tools.

## Flow
1. `ChatService` checks whether input looks like log-analysis.
2. If matched, it routes request to `AgentExecutor`.
3. Executor asks LLM for action plan and executes `analyze_log`.
4. Executor then calls `summarize_text` and returns JSON result.

## Quick test
Run from workspace root:

```powershell
python -m unittest discover -s tests -p "test_*agent*.py"
```

