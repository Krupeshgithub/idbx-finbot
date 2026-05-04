"""
Cloud DLP Client for Sensitive Data Protection

Handles automatic redaction of Personally Identifiable Information (PII)
like emails, phone numbers, and Client IDs before they are logged or
sent to Vertex AI. Configuration is dynamic.
"""
import logging
from typing import Optional

from google.cloud import dlp_v2
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class DLPClient:
    """
    Cloud DLP Client for Sensitive Data Protection.
    Handles automatic redaction of Personally Identifiable Information (PII)
    using Google Cloud DLP.
    """

    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT
        self.location = settings.GCP_DLP_LOCATION
        self.deidentify_template = settings.GCP_DLP_DEIDENTIFY_TEMPLATE
        self.inspect_template = settings.GCP_DLP_INSPECT_TEMPLATE
        self.client = None
        self.parent = None

    def initialize(self) -> bool:
        """
        Explicitly initialize the DLP client.
        Call this at startup to avoid first-request latency.
        """
        return self._ensure_client()

    def _ensure_client(self) -> bool:
        """Lazy initializer for the DLP client."""
        if self.client is not None:
            return True
            
        if not self.project_id:
            return False

        try:
            self.client = dlp_v2.DlpServiceClient()
            self.parent = f"projects/{self.project_id}/locations/{self.location}"
            logger.info("[DLPClient] Initialized GCP DLP Client for project: %s", self.project_id)
            return True
        except Exception as exc:
            logger.error("[DLPClient] Initialization failed: %s", exc)
            return False

    def redact_pii(self, text: str) -> str:
        """
        Redacts PII from the input string using Cloud DLP.
        Returns the original text if DLP is not configured or fails.
        """
        if not text or not text.strip():
            return text

        if not self._ensure_client():
            return text
        
        try:
            item = {"value": text}

            # Prepare Inspect Config
            if self.inspect_template:
                inspect_config = None # Use template name instead
                inspect_template_name = self.inspect_template
            else:
                inspect_template_name = None
                inspect_config = {
                    "info_types": [
                        {"name": "EMAIL_ADDRESS"},
                        {"name": "PHONE_NUMBER"},
                        {"name": "CREDIT_CARD_NUMBER"},
                        {"name": "US_SOCIAL_SECURITY_NUMBER"},
                        {"name": "IP_ADDRESS"},
                        {"name": "DATE_OF_BIRTH"},
                        {"name": "INDIA_PAN_INDIVIDUAL"},
                        {"name": "INDIA_AADHAAR_INDIVIDUAL"},
                        {"name": "FINANCIAL_ACCOUNT_NUMBER"},
                        {"name": "IBAN_CODE"},
                        {"name": "PERSON_NAME"},
                    ]
                }

            # Prepare De-identify Config
            if self.deidentify_template:
                deidentify_config = None # Use template name instead
                deidentify_template_name = self.deidentify_template
            else:
                deidentify_template_name = None
                deidentify_config = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "character_mask_config": {
                                        "masking_character": "*",
                                        "number_to_mask": 0,
                                        "reverse_order": False,
                                    }
                                }
                            }
                        ]
                    }
                }

            request = {
                "parent": self.parent,
                "item": item,
            }
            
            if inspect_template_name:
                request["inspect_template_name"] = inspect_template_name
            if inspect_config:
                request["inspect_config"] = inspect_config
                
            if deidentify_template_name:
                request["deidentify_template_name"] = deidentify_template_name
            if deidentify_config:
                request["deidentify_config"] = deidentify_config

            response = self.client.deidentify_content(request=request)
            redacted = response.item.value

            # Debug: log what changed so we can verify DLP is working
            if redacted != text:
                import re
                # Count masked characters to show redaction extent
                original_len = len(text)
                redacted_len = len(redacted)
                stars = redacted.count("*")
                logger.info(
                    "[DLPClient] ✅ PII REDACTED | original_len=%s redacted_len=%s stars_inserted=%s | preview_redacted=%s",
                    original_len, redacted_len, stars,
                    redacted[:120]
                )
            else:
                logger.debug("[DLPClient] No PII detected in input (len=%s)", len(text))

            return redacted
    
        except Exception as exc:
            logger.error("[DLPClient] ❌ Redaction failed: %s", exc)
            return text

dlp_client = DLPClient()
