"""
Structured logging for production.
-Uses contextvar so every log line in a single request automatically attaches request_id.
-Logs to both stdout (so the container platform can collect logs) and a daily rotating file
(TimedRotatingFileHandler, when="midnight", backupCount=LOG_RETENTION_DAYS)
-> Logs older than N days are automatically deleted, N can be modified via .env (LOG_RETENTION_DAYS).
"""
import logging
import logging.handlers
import sys
import json
import contextvars
import os
from datetime import datetime, timezone

from app.core.config import get_settings

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields and isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id_ctx.get()
        return f"[{self.formatTime(record)}] {record.levelname:<8} [{record.request_id}] {record.name}: {record.getMessage()}"


def setup_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    root.handlers.clear()

    formatter = JsonFormatter() if settings.LOG_JSON else PlainFormatter()

    # --- stdout handler (container platform / docker logs) ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # --- file handler, daily rotate and delete logs older than LOG_RETENTION_DAYS ---
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(settings.LOG_DIR, "app.log"),
        when="midnight",
        interval=1,
        backupCount=settings.LOG_RETENTION_DAYS,  # logs older than N days will be removed
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("uvicorn.access", "ultralytics", "paddleocr", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
