"""
AIDAAN Structured Logging Setup
=================================
Call `setup_logging()` once at app startup (in server.py).

Features:
  - JSON-formatted logs in production (LOG_FORMAT=json in .env)
  - Human-readable colored logs in development (default)
  - Per-module log levels (e.g., silence noisy libraries)
  - Every LLM call is traceable via [LLMClient] prefix

HOW TO DEBUG:
    Set LOG_LEVEL=DEBUG in your .env to see:
      - Full prompt previews from LLMClient
      - Intent classification results
      - Agent routing decisions
      - Cache hit/miss events

    In production, set LOG_LEVEL=INFO and LOG_FORMAT=json
    to pipe logs into your observability stack (Datadog, GCP Logging, etc.)

USAGE:
    # In server.py (already wired up):
    from app.core.logging_config import setup_logging
    setup_logging()

    # In any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("...")
"""
import logging
import sys
import os


def setup_logging():
    """
    Configure root logger for the entire AIDAAN application.
    Call once at startup.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text")  # "text" or "json"
    log_level = getattr(logging, log_level_str, logging.INFO)

    if log_format == "json":
        _setup_json_logging(log_level)
    else:
        _setup_text_logging(log_level)

    # Silence noisy third-party libraries
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        f"[Logging] Configured. level={log_level_str} format={log_format}"
    )


def _setup_text_logging(level: int):
    """Human-readable format for local development."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def _setup_json_logging(level: int):
    """
    JSON format for production / log aggregation pipelines.
    Each log line is a valid JSON object.
    """
    import json
    import traceback

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_obj = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                log_obj["exc"] = traceback.format_exception(*record.exc_info)
            return json.dumps(log_obj)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
