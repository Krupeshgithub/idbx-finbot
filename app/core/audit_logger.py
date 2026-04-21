"""
Cloud Logging Audit Logger

Satisfies the Log Audit Trail mandate by recording all Vertex AI interactions,
Model Versions, and latency to Google Cloud Logging API securely.
"""
import logging
import time
from typing import Any, Optional

from google.cloud import logging as cloud_logging
from app.core.config.settings import settings


logger = logging.getLogger(__name__)


class AuditLogger:

    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT
        self.log_name = settings.GCP_AUDIT_LOG_NAME
        self.client = None
        self.cloud_logger = None
        
    def _ensure_client(self):
        """Ensures Cloud Logging client is ready when first log is sent."""
        if self.cloud_logger is not None:
            return True
            
        if not self.project_id:
            return False

        try:
            self.client = cloud_logging.Client(project=self.project_id)
            self.cloud_logger = self.client.logger(self.log_name)
            logger.info("[AuditLogger] Initialized GCP Cloud Logging Client.")
            return True
        except Exception as exc:
            logger.error("[AuditLogger] Failed to initialize: %s", exc)
            return False
    
    def log_llm_transaction(
        self,
        prompt: str,
        response: Any,
        model_id: str,
        latency_ms: int,
        user_context: Optional[dict] = None,
        status: str = "SUCCESS"
    ):
        """
        Record a structured audit log to GCP.
        Captures the Model Version explicitly and any Desk details from the user.
        """
        if not self._ensure_client():
            return

        # Prepare payload
        payload = {
            "event_type": "LLM_TRANSACTION",
            "model_version_id": model_id,
            "latency_ms": latency_ms,
            "prompt_length": len(prompt) if prompt else 0,
            "status": status,
            "timestamp": time.time()
        }

        # Include basic user accountability data if available
        if user_context:
            payload["user"] = user_context.get("username", "anonymous")
            payload["desk_role"] = user_context.get("role", "unknown")

        # Notice: We intentionally do NOT log the raw prompt or raw response into standard
        # audit logs to prevent PII leakage, relying instead on metadata fields.

        try:
            self.cloud_logger.log_struct(
                payload,
                severity="INFO" if status == "SUCCESS" else "ERROR",
                labels={"module": "aidaan.audit"}
            )
        except Exception as exc:
            logger.error("[AuditLogger] Failed to push struct log to Cloud Logging: %s", exc)


audit_logger = AuditLogger()
