"""
Coordinator Agent Implementation for AIDAAN.
The "Brain" of the operation, now with A2A and Guardrails.
"""
from typing import Any, Dict, List, Optional
from uuid import uuid4
from app.services.aidaan.base import BaseAgent
from app.schemas.aidaan import ActionItem, AidaanMessageResponse, ModelInfo
from app.services.aidaan.a2a_service import a2a_service
from app.core.guardrails import guardrails


class CoordinatorAgent(BaseAgent):
    """
    Orchestrates the conversation and delegates to specialized agents.
    """

    def __init__(self):
        super().__init__(name="coordinator", model_name="gemini-1.5-pro")

    async def handle_message(
        self, 
        text: str, 
        conversation_id: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Process user intent and return a structured response with A2A and Guardrails logic.
        """
        conv_id = conversation_id or f"conv-{uuid4().hex[:8]}"
        
        # 1. Check Compliance Guardrails
        is_advisory, reason = guardrails.is_advisory(text)
        if is_advisory:
            return AidaanMessageResponse(
                reply=f"I cannot provide financial advice. {guardrails.get_compliance_notice()}",
                bullets=[f"Compliance Block: {reason}"],
                actions=[],
                conversation_id=conv_id,
                model=self.get_model_info()
            )

        # 2. Hardcoded static intent mapping for Phase 1
        lowered_text = text.lower()
        
        if "rfq" in lowered_text and "eur/pln" in lowered_text:
            # Simulate A2A call to Risk Agent
            risk_resp = await a2a_service.query_agent("risk", text, context)
            
            return AidaanMessageResponse(
                reply="I have prepared an RFQ draft for EUR/PLN Swap 3M as requested.",
                bullets=[
                    "Instrument: EUR/PLN Swap 3M",
                    "Side: BUY",
                    "Notional: 50,000,000",
                    f"Risk Status: {risk_resp.get('status', 'Unknown')}"
                ],
                actions=[
                    ActionItem(
                        kind="draft_rfq",
                        title="Draft RFQ (review & send)",
                        payload={"instrument": "EUR/PLN Swap 3M", "side": "BUY", "notional": 50000000},
                        requires_human_confirm=True
                    )
                ],
                conversation_id=conv_id,
                model=self.get_model_info()
            )
        
        elif "fund" in lowered_text and "5bn" in lowered_text:
             # Simulate A2A call to Distributor Agent
             dist_resp = await a2a_service.query_agent("distributor", text, context)
             
             return AidaanMessageResponse(
                reply="To fund 5bn overnight, I suggest checking the O/N repo rates or executing a synthetic spread via the Swap desk.",
                bullets=[
                    "Recommendation: Overnight Repo",
                    f"Liquidity Matches: {len(dist_resp.get('matches', []))}",
                    "Liquidity Status: Ready"
                ],
                actions=[
                    ActionItem(
                        kind="open_view",
                        title="View Funding Desk",
                        payload={"view": "funding_grid", "filter": "overnight"},
                        requires_human_confirm=False
                    )
                ],
                conversation_id=conv_id,
                model=self.get_model_info()
            )

        # Default fallback
        return AidaanMessageResponse(
            reply=f"I understand your request regarding '{text}'. However, as I am currently in Phase 1 (Skeleton), I have limited processing capabilities. Would you like me to draft an RFQ or show market sentiment instead?",
            bullets=["Phase 1: Static Responses Enabled", "Compliance: Non-Advisory Verified"],
            actions=[
                ActionItem(
                    kind="explain",
                    title="What can you do?",
                    payload={"topic": "capabilities"},
                    requires_human_confirm=False
                )
            ],
            conversation_id=conv_id,
            model=self.get_model_info()
        )

    def get_capabilities(self) -> List[str]:
        return ["intent_recognition", "rfq_drafting", "multi_agent_coordination", "compliance_filtering"]


coordinator_agent = CoordinatorAgent()
