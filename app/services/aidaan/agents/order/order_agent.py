"""
Institutional Order Parsing Agent - AIDAAN NLP-to-Action Layer
==============================================================
The OrderAgent specializes in extracting structured institutional trade 
parameters from natural language requests. It bridges the gap between 
trader intent and the platform's order entry system.

Key Features:
- Parameter extraction for Interest Rate Swaps (SONIA/SOFR), FX, and Bonds.
- Support for complex institucional tenors (2Y, 5Y) and settlements (IMM, T+2).
- Structured JSON output for pre-filling RFQ tickets on the platform.
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.aidaan.core.base import BaseAgent
from app.schemas.aidaan import AidaanMessageResponse
from app.core.prompts import Prompts
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class OrderAgent(BaseAgent):
    """
    NLP specialist for parsing and validating institutional trade 
    parameters into structured payloads.
    """

    def __init__(self):
        """
        Initialize with a model optimized for high-fidelity extraction.
        """
        super().__init__(name="order", model_name=settings.VERTEX_AI_MODEL_NAME)

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Parses trade requests and prepares a structured staging summary.

        Args:
            text: Raw institutional trade description (e.g., "Stage 50m 5Y SOFR").
            conversation_id: Session identifier.
            context: Additional user/platform context.

        Returns:
            AidaanMessageResponse: Summary of the parsed order ticket.
        """
        logger.info(f"[OrderAgent] Attempting NLP-to-Action parse: {text}")
        context = context or {}
        
        # Externalized institutional parser prompt
        prompt = Prompts.ORDER_PARSER.format(text=text)
        
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=context.get("username"),
            )
            
            instrument = parsed.get("instrument", "Unknown Asset")
            size = parsed.get("size", 0)
            tenor = parsed.get("tenor", "Market")
            settlement = parsed.get("settlement", "Spot/T+2")
            
            reply = f"I've identified an institutional request for **{instrument}**. Preparing the staging ticket..."
            bullets = [
                f"Instrument: {instrument}",
                f"Notional: {size:,} (Base Units)",
                f"Tenor: {tenor}",
                f"Settlement: {settlement}",
                "Status: STANDBY FOR STAGING"
            ]
            
            return self.build_message_response(
                reply=reply,
                bullets=bullets,
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as e:
            return self.build_error_response(
                reply="I couldn't identify the exact trade parameters (Size, Tenor, or Instrument).",
                conversation_id=conversation_id,
                error=e,
                bullets=["NLP Error: Insufficient parameters extracted"],
                model_info=self.get_model_info(),
            )

    def get_capabilities(self) -> List[str]:
        """
        Lists specific parsing capabilities handled by this agent.
        """
        return ["nlp_to_action", "rfq_parameter_extraction", "institutional_grammar_parsing"]


# Singleton instance
order_agent = OrderAgent()
