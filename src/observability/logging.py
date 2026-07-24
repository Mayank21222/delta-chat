"""
Structured JSON logging, not print statements.

Every log line is a JSON object with at minimum: timestamp, level, message,
correlation_id, and component. This is what "we want to open your project
and understand what happened without re-running it" (assignment section 05)
actually requires -- greppable, jq-able, one line per event.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        for key in ("correlation_id", "stage", "extra_data"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(component: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(component)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    logger.addHandler(stream_handler)

    return logger


def log(logger: logging.Logger, level: str, message: str, correlation_id: str,
        stage: str | None = None, **extra) -> None:
    record_extra = {"correlation_id": correlation_id}
    if stage:
        record_extra["stage"] = stage
    if extra:
        record_extra["extra_data"] = extra
    getattr(logger, level)(message, extra=record_extra)


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]
