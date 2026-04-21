"""
Cloud DLP Client for Sensitive Data Protection

Handles automatic redaction of Personally Identifiable Information (PII)
like emails, phone numbers, and Client IDs before they are logged or
sent to Vertex AI. Configuration is dynamic.
"""
import logging

from google.cloud import dlp_v2
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class DLPClient:

    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT
        self.location = settings.GCP_DLP_LOCATION
        self.deidentify_template = settings.GCP_DLP_DEIDENTIFY_TEMPLATE
        self.inspect_template = settings.GCP_DLP_INSPECT_TEMPLATE
        self.client = None
        self.parent = None

    def _ensure_client(self):
        """Lazy initializer to ensure credentials are set before client creation."""
        if self.client is not None:
            return True
            
        if not self.project_id:
            return False

        try:
            self.client = dlp_v2.DlpServiceClient()
            self.parent = f"projects/{self.project_id}/locations/{self.location}"
            logger.info("[DLPClient] Initialized GCP DLP Client.")
            return True
        except Exception as exc:
            logger.error("[DLPClient] Failed to initialize: %s", exc)
            return False

    def redact_pii(self, text: str) -> str:
        """
        Redacts PII from the input string using Cloud DLP.
        Returns the original text if DLP is not configured or fails.
        """
        if not self._ensure_client() or not text:
            return text
        
        try:
            item = {"value": text}

            request = {
                "parent": self.parent,
                "item": item
            }

            # Optional: Use managed templates if defined
            if self.inspect_template:
                request["inspect_template_name"] = self.inspect_template
            else:
                # Fallback to standard infoTypes if no template is explicity provided
                request["inspect_config"] = {
                    "info_types": [
                        {"name": "EMAIL_ADDRESS"},
                        {"name": "PHONE_NUMBER"},
                        {"name": "CREDIT_CARD_NUMBER"},
                        {"name": "US_SOCIAL_SECURITY_NUMBER"},
                        {"name": "PERSON_NAME"}
                    ]
                }

            if self.deidentify_template:
                request["deidentify_template_name"] = self.deidentify_template
            else:
                # Fallback to basic masking if no template is provided
                request["deidentify_config"] = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "replace_config": {
                                        "new_value": {
                                            "string_value": "[REDACTED]"
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }

            response = self.client.deidentify_content(request=request)
            return response.item.value
    
        except Exception as exc:
            # We don't want DLP failure to bring down the trading bot entirely,
            # but we log it as an error.
            logger.error("[DLPClient] Redaction failed, returning original text safely. Error: %s", exc)
            return text
    
dlp_client = DLPClient()
