# web_main.py
import logging
import os

from ai_app.web.app import create_app


if __name__ == "__main__":
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "7860"))
    debug = os.getenv("WEB_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")

    app = create_app()
    app.run(host=host, port=port, debug=debug)

