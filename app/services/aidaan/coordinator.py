"""
Coordinator Agent Implementation for AIDAAN.
The "Brain" of the operation, now with A2A and Guardrails.
"""
from typing import Any, Dict, Tuple, Optional, List
from uuid import uuid4

from app.services.aidaan.base import BaseAgent
from app.services.aidaan.registry import registry
from app.services.aidaan.session_manager import session_manager
from app.core.guardrails import guardrails
from app.schemas.aidaan import (
    ActionItem, 
    AidaanMessageResponse, 
)

# Import agents for registration
from app.services.aidaan.risk_agent import risk_agent
from app.services.aidaan.distributor_agent import distributor_agent


registry.register("risk", risk_agent)
registry.register("distributor", distributor_agent)


class CoordinatorAgent(BaseAgent):
    """
    Orchestrates the conversation and delegates to specialized agents.
    """

    def __init__(self):
        super().__init__(
            name="coordinator", 
            model_name="gemini-1.5-pro"
        )
    
    async def handle_message(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Process user intent and return a structured response
        with A2A and Guardrails logic.
        """
        conv_id = conversation_id or f"conv-{uuid4().hex[:8]}"

        # 0. Load existing context/session state
        session_context = session_manager.get_context(conv_id)
        if context:
            session_context.update(context)
            session_manager.save_context(conv_id, session_context)
        
        # 1. Action Callback Handler
        action_kind = session_context.get("kind")
        if action_kind == "draft_rfq":
             return AidaanMessageResponse(
                reply="RFQ Drafted: EUR/PLN Swap 3M @ 50M EUR. Proceed to distribution?",
                bullets=["Side: BUY", "Notional: 50,000,000", "Venue: Institutional Hybrid"],
                actions=[
                    ActionItem(
                        kind="confirm_rfq",
                        title="Yes, Distribute Now",
                        payload={"kind": "confirm_rfq"},
                        requires_human_confirm=True
                    )
                ],
                conversation_id=conv_id,
                model=self.get_model_info()
            )
        
        if action_kind == "confirm_rfq":
             return AidaanMessageResponse(
                reply="Distribution active. RFQ is now being priced by LSEG and internal nodes.",
                bullets=["Status: LIVE", "RFQ ID: RFQ-9912", "Liquidity: 3 Matches"],
                actions=[],
                conversation_id=conv_id,
                model=self.get_model_info()
            )

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
        
        # 2. Intent Classification (Agent Selection)
        agent_id, _ = self._classify_intent(text)

        # 3. Delegate to Specialized Agent
        if agent_id:
            target_agent = registry.get_agent(agent_id)
            if target_agent:
                return await target_agent.handle_message(
                    text=text, 
                    conversation_id=conv_id, 
                    context=session_context
                )
        
        # 4. Phase 1 Static Handlers
        lowered_text = text.lower()
        if "rfq" in lowered_text:
            return AidaanMessageResponse(
                reply="I can help draft an RFQ. What are the details of the request?",
                bullets=["Standard 50M threshold applied"],
                actions=[
                    ActionItem(
                        kind="draft_rfq",
                        title="Draft RFQ (review & send)",
                        payload={"kind": "draft_rfq"},
                        requires_human_confirm=True
                    )
                ],
                conversation_id=conv_id,
                model=self.get_model_info()
            )

        return AidaanMessageResponse(
            reply="AIDAAN is standing by. You can ask about Risk, Sentiment, or Liquidity Distribution.",
            bullets=["Phase 1 modularity active", "Gemini 1.5 Pro: Routing Only"],
            actions=[
                ActionItem(
                    kind="ask_risk",
                    title="Check Desk Risk",
                    payload={"query": "desk_risk"},
                    requires_human_confirm=False
                ),
                ActionItem(
                    kind="ask_liquidity",
                    title="Find Liquidity",
                    payload={"query": "liquidity"},
                    requires_human_confirm=False
                )
            ],
            conversation_id=conv_id,
            model=self.get_model_info()
        )

    def _classify_intent(
        self,
        text: str
    ) -> Tuple[Optional[str], float]:
        """
        Maps text to an Agent ID.
        In phase 2, this will be an LLM-based classifier.
        """
        lowered = text.lower()
        if any(k in lowered for k in ["risk", "check status", "sentiment"]):
            return "risk", 1.0
        if any(k in lowered for k in ["liquidity", "fund", "distribute"]):
            return "distributor", 1.0

        return None, 0.0
    
    def get_capabilities(self) -> List[str]:
        return [
            "intent_classification",
            "agent_delegation",
            "compliance_guardrails",
            "rfq_drafting"
        ]
    
coordinator_agent = CoordinatorAgent()
