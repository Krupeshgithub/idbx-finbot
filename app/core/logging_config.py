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
from app.core.config.settings import settings


def setup_logging():
    """
    Configure root logger for the entire AIDAAN application.
    Integrates Google Cloud Logging if a project ID is available.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text")  # "text" or "json"
    log_level = getattr(logging, log_level_str, logging.INFO)

    # 1. Setup Local Handlers (Stdout)
    if log_format == "json":
        _setup_json_logging(log_level)
    else:
        _setup_text_logging(log_level)

    # 1b. Persistent File Logging
    _setup_file_logging(log_level)

    # 1c. Performance Logging (Dedicated file)
    _setup_performance_logging()

    # 2. Setup Google Cloud Logging (Conditional)
    _setup_gcp_logging(log_level)

    # Silence noisy third-party libraries
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        f"[Logging] Configured. level={log_level_str} format={log_format} gcp_enabled={bool(settings.GOOGLE_CLOUD_PROJECT)}"
    )


def _setup_text_logging(level: int):
    """Human-readable format for local development."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    # We use force=True to ensure basicConfig actually updates the root logger
    logging.basicConfig(level=level, handlers=[handler], force=True)


def _setup_json_logging(level: int):
    """JSON format for production logs."""
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


def _setup_file_logging(level: int):
    """Save logs to a persistent file in the logs/ directory."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_path = os.path.join(log_dir, "app.log")
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handler = logging.FileHandler(file_path)
    handler.setFormatter(logging.Formatter(fmt=fmt))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)


def _setup_performance_logging():
    """Setup a dedicated logger for performance metrics and transaction tables."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_path = os.path.join(log_dir, "performance.log")
    
    # We want a clean output without the standard log prefixes (asctime, levelname, etc.)
    # because the transaction table is already formatted.
    fmt = "%(asctime)s | %(message)s"
    handler = logging.FileHandler(file_path)
    handler.setFormatter(logging.Formatter(fmt=fmt))
    
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    perf_logger.addHandler(handler)
    # Prevent performance logs from bubbling up to the root logger (and app.log)
    perf_logger.propagate = False


def _setup_gcp_logging(level: int):
    """
    Attaches the Google Cloud Logging handler to the root logger.
    Requires GOOGLE_CLOUD_PROJECT and valid credentials.
    """
    project_id = settings.GOOGLE_CLOUD_PROJECT
    if not project_id:
        return

    try:
        from google.cloud import logging as cloud_logging
        from google.cloud.logging.handlers import CloudLoggingHandler
        
        client = cloud_logging.Client(project=project_id)
        # The handler will automatically capture all logs from the root logger
        handler = CloudLoggingHandler(client, name="aidaan-app-logs")
        handler.setLevel(level)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Note: We don't use basicConfig here because we want to ADD to existing handlers, 
        # not replace them.
    except Exception as exc:
        # Don't fail the app if logging setup fails
        print(f"[Logging] Failed to initialize GCP Logging: {exc}", file=sys.stderr)
