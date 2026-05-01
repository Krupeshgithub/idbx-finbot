"""
Guardrail Orchestrator for AIDANN

Coordinates guardrail checks across the request/response flow.
Provides centralized guardrail logic for consistency.
"""
import logging
from typing import Dict, Any, Optional

from app.core.advisory_classifier import advisory_classifier
from app.services.kill_switch import kill_switch_service
from app.core.dlp_client import dlp_client
from app.core.audit_logger import audit_logger
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class GuardrailOrchestrator:
    """
    Coordinates guardrail checks across the request/response flow.
    
    Enforcement Points:
    1. Pre-processing: Kill switch checks, advisory detection
    2. During processing: PII redaction, strategic data redaction
    3. Post-processing: Response reframing, audit logging
    """
    
    def __init__(self):
        self.kill_switch = kill_switch_service
        self.dlp = dlp_client
        self.audit = audit_logger
        self.advisory = advisory_classifier
    
    def pre_process_request(
        self,
        text: str,
        *,
        username: Optional[str],
        conversation_id: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply pre-processing guardrails.
        
        Args:
            text: User query text
            username: Username for authorization
            conversation_id: Conversation ID for tracking
            context: Request context dictionary (will be updated)
        
        Returns:
            {
                "allowed": bool,
                "blocked_reason": Optional[str],
                "advisory_detected": bool,
                "context_updates": Dict[str, Any]
            }
        """
        result = {
            "allowed": True,
            "blocked_reason": None,
            "advisory_detected": False,
            "context_updates": {}
        }
        
        # 1. Kill Switch Check
        if settings.ENABLE_KILL_SWITCH and self.kill_switch.is_active():
            if self.kill_switch.request_has_trade_intent(text):
                status = self.kill_switch.get_status()
                result["allowed"] = False
                result["blocked_reason"] = (
                    f"Venue kill switch is active: {status.get('reason')}"
                )
                
                logger.warning(
                    "[GuardrailOrchestrator] Request blocked by kill switch | "
                    "user=%s | conv_id=%s | text=%s",
                    username,
                    conversation_id,
                    text[:100]
                )
                
                # Log guardrail violation
                self.audit.log_llm_transaction(
                    prompt=text,
                    response={"blocked": True, "reason": "kill_switch_active"},
                    model_id="guardrail_check",
                    latency_ms=0,
                    user_context={"username": username},
                    status="BLOCKED"
                )
                
                return result
        
        # 2. Advisory Intent Detection
        if settings.ENABLE_ADVISORY_DETECTION:
            advisory_result = self.advisory.detect_advisory_intent(
                text,
                conversation_id=conversation_id
            )
            
            if advisory_result["is_advisory"]:
                result["advisory_detected"] = True
                result["context_updates"]["advisory_detected"] = True
                result["context_updates"]["advisory_confidence"] = advisory_result["confidence"]
                result["context_updates"]["advisory_patterns"] = advisory_result["patterns_matched"]
                
                logger.info(
                    "[GuardrailOrchestrator] Advisory intent detected | "
                    "confidence=%.2f | user=%s | conv_id=%s",
                    advisory_result["confidence"],
                    username,
                    conversation_id
                )
        
        return result
    
    def process_prompt(
        self,
        prompt: str,
        *,
        conversation_id: Optional[str]
    ) -> str:
        """
        Apply during-processing guardrails (redaction).
        
        Args:
            prompt: LLM prompt text
            conversation_id: Conversation ID for tracking
        
        Returns:
            Redacted prompt safe for external LLM
        """
        redacted_prompt = prompt
        
        # 1. PII Redaction
        if settings.ENABLE_PII_REDACTION:
            try:
                redacted_prompt = self.dlp.redact_pii(redacted_prompt)
                logger.debug(
                    "[GuardrailOrchestrator] PII redaction applied | conv_id=%s",
                    conversation_id
                )
            except Exception as exc:
                logger.error(
                    "[GuardrailOrchestrator] PII redaction failed: %s",
                    exc
                )
                # Continue without PII redaction (fail-open for availability)
        
        # 2. Strategic Data Redaction
        if settings.ENABLE_STRATEGIC_DATA_REDACTION:
            try:
                redacted_prompt = self._redact_strategic_data(redacted_prompt)
                logger.debug(
                    "[GuardrailOrchestrator] Strategic data redaction applied | conv_id=%s",
                    conversation_id
                )
            except Exception as exc:
                logger.error(
                    "[GuardrailOrchestrator] Strategic data redaction failed: %s",
                    exc
                )
                # Continue without strategic data redaction
        
        return redacted_prompt
    
    async def post_process_response(
        self,
        response: str,
        *,
        context: Dict[str, Any],
        conversation_id: Optional[str],
        original_query: Optional[str] = None
    ) -> str:
        """
        Apply post-processing guardrails (reframing).
        
        Args:
            response: Agent response text
            context: Request context with guardrail flags
            conversation_id: Conversation ID for tracking
            original_query: Original user query for reframing context
        
        Returns:
            Reframed response if advisory detected
        """
        # Advisory Response Reframing
        if context.get("advisory_detected") and settings.ENABLE_ADVISORY_DETECTION:
            try:
                reframed = await self.advisory.reframe_advisory_response(
                    response,
                    original_query or ""
                )
                
                logger.info(
                    "[GuardrailOrchestrator] Response reframed | conv_id=%s | "
                    "original_length=%d | reframed_length=%d",
                    conversation_id,
                    len(response),
                    len(reframed)
                )
                
                return reframed
            
            except Exception as exc:
                logger.error(
                    "[GuardrailOrchestrator] Response reframing failed: %s",
                    exc
                )
                # Return generic non-advisory response as fallback
                return (
                    "I can provide you with factual market data and risk metrics, "
                    "but I cannot provide investment advice or recommendations. "
                    "Please consult with a licensed financial advisor for investment decisions."
                )
        
        return response
    
    def _redact_strategic_data(self, text: str) -> str:
        """
        Redact strategic trading data patterns.
        
        Patterns:
            - Position sizes (e.g., "10M notional", "$5MM")
            - Trading signals (e.g., "buy signal", "sell indicator")
            - Risk metrics (e.g., "DV01: 1234.56")
        """
        import re
        
        # Position size patterns
        text = re.sub(
            r"\b\d+(?:\.\d+)?[MBK]?\s*(?:notional|position|size)\b",
            "[REDACTED_POSITION]",
            text,
            flags=re.IGNORECASE
        )
        
        # Dollar amount patterns
        text = re.sub(
            r"\$\d+(?:\.\d+)?(?:MM|M|K|B)?\b",
            "[REDACTED_AMOUNT]",
            text,
            flags=re.IGNORECASE
        )
        
        # Trading signal patterns
        text = re.sub(
            r"\b(?:buy|sell)\s+signal\b",
            "[REDACTED_SIGNAL]",
            text,
            flags=re.IGNORECASE
        )
        
        # Position direction patterns
        text = re.sub(
            r"\b(?:long|short)\s+\d+\b",
            "[REDACTED_DIRECTION]",
            text,
            flags=re.IGNORECASE
        )
        
        # Risk metric patterns
        text = re.sub(
            r"\b(?:DV01|PV01|VaR):\s*[\d.]+\b",
            "[REDACTED_RISK_METRIC]",
            text,
            flags=re.IGNORECASE
        )
        
        return text


# Global instance
guardrail_orchestrator = GuardrailOrchestrator()
