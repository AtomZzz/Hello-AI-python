# Web Chat UI

A minimal web frontend for the AI app.

## Features
- Chat through browser instead of terminal.
- Supports multiline input in textarea (good for logs).
- Shows user message and AI reply.
- Displays router metadata (`use_agent`, `use_rag`, `require_json`).

## Run
From project root:

```powershell
python -m pip install -r requirements.txt
python -m ai_app.web_main
```

Open browser:
- `http://127.0.0.1:7860`

## Environment Variables
- `WEB_HOST` (default `127.0.0.1`)
- `WEB_PORT` (default `7860`)
- `WEB_DEBUG` (default `0`)
- Reuses existing model vars: `LLM_TYPE`, `QWEN_API_KEY`, `QWEN_BASE_URL`, `QWEN_MODEL`, `RAG_ENABLED`, `ROUTING_MODE`...

