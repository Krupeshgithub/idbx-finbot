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
from app.schemas.aidaan import ActionItem, AidaanMessageResponse
from app.core.prompts import Prompts
from app.core.config.settings import settings
from app.services.kill_switch import kill_switch_service
from app.services.aidaan.trade_parsing import infer_instrument, parse_notional, parse_settlement, parse_tenor

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

    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        instrument = infer_instrument(text, default="Swap")
        size = parse_notional(text) or 0.0
        tenor = parse_tenor(text) or "Market"
        settlement = parse_settlement(text) or "Spot/T+2"
        lowered = text.lower()
        action = "stage_rfq" if any(k in lowered for k in ["stage", "rfq", "draft"]) else "unknown"
        return {
            "instrument": instrument,
            "size": size,
            "tenor": tenor,
            "settlement": settlement,
            "action": action,
        }

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
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
        user_identity = context.get("username")
        
        # Externalized institutional parser prompt
        prompt = Prompts.ORDER_PARSER.format(text=text)
        
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=context.get("username"),
            )
            if parsed.get("error"):
                raise RuntimeError(parsed["error"])
            
            instrument = parsed.get("instrument", "Unknown Asset")
            size = parsed.get("size", 0)
            tenor = parsed.get("tenor", "Market")
            settlement = parsed.get("settlement", "Spot/T+2")

            if kill_switch_service.is_active():
                status = kill_switch_service.get_status()
                return self.build_message_response(
                    reply="Venue kill switch is active. I can parse the request, but I will not stage or draft any RFQ until the freeze is released.",
                    bullets=[
                        f"Instrument: {instrument}",
                        f"Notional: {size:,} (Base Units)",
                        f"Tenor: {tenor}",
                        f"Settlement: {settlement}",
                        f"Kill switch reason: {status.get('reason')}",
                    ],
                    actions=[],
                    conversation_id=conversation_id,
                    model_info=self.get_model_info(),
                )
            
            reply = f"I've identified an institutional request for {instrument}. Preparing the staging ticket..."
            bullets = [
                f"Instrument: {instrument}",
                f"Notional: {size:,} (Base Units)",
                f"Tenor: {tenor}",
                f"Settlement: {settlement}",
                "Status: STANDBY FOR STAGING"
            ]

            actions = [
                ActionItem(
                    kind="invoke_tool",
                    title="Draft RFQ Ticket",
                    payload={
                        "tool_name": "draft_rfq_ticket",
                        "arguments": {
                            "conversation_id": conversation_id,
                            "instrument": instrument,
                            "notional": size,
                            "tenor": tenor,
                            "settlement": settlement,
                            "user_identity": user_identity,
                        },
                    },
                    requires_human_confirm=True,
                )
            ]
            
            return self.build_message_response(
                reply=reply,
                bullets=bullets,
                actions=actions,
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as e:
            parsed = self._fallback_parse(text)
            instrument = parsed["instrument"]
            size = parsed["size"]
            tenor = parsed["tenor"]
            settlement = parsed["settlement"]

            if kill_switch_service.is_active():
                status = kill_switch_service.get_status()
                return self.build_message_response(
                    reply="Venue kill switch is active. I captured the trade parameters, but drafting is blocked until the emergency freeze is lifted.",
                    bullets=[
                        f"Instrument: {instrument}",
                        f"Notional: {int(size):,} (Base Units)" if size else "Notional: (missing)",
                        f"Tenor: {tenor}",
                        f"Settlement: {settlement}",
                        f"Kill switch reason: {status.get('reason')}",
                    ],
                    actions=[],
                    conversation_id=conversation_id,
                    model_info=self.get_model_info(),
                )

            reply = f"(Deterministic Mode) Drafting an RFQ ticket for {instrument}."
            bullets = [
                f"Instrument: {instrument}",
                f"Notional: {int(size):,} (Base Units)" if size else "Notional: (missing)",
                f"Tenor: {tenor}",
                f"Settlement: {settlement}",
                "Status: STANDBY FOR STAGING",
            ]
            actions = [
                ActionItem(
                    kind="invoke_tool",
                    title="Draft RFQ Ticket",
                    payload={
                        "tool_name": "draft_rfq_ticket",
                        "arguments": {
                            "conversation_id": conversation_id,
                            "instrument": instrument,
                            "notional": size,
                            "tenor": tenor,
                            "settlement": settlement,
                            "user_identity": user_identity,
                        },
                    },
                    requires_human_confirm=True,
                )
            ]
            logger.warning("[OrderAgent] LLM parse failed; using fallback parser: %s", e)
            return self.build_message_response(
                reply=reply,
                bullets=bullets,
                actions=actions,
                conversation_id=conversation_id,
                model_info={"agent": self.name, "llm": "fallback-parser"},
            )

    def get_capabilities(self) -> List[str]:
        """
        Lists specific parsing capabilities handled by this agent.
        """
        return ["nlp_to_action", "rfq_parameter_extraction", "institutional_grammar_parsing"]


# Singleton instance
order_agent = OrderAgent()
