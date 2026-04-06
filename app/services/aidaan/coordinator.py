"""
Coordinator Agent Implementation for AIDAAN.
The "Brain" of the operation, now with A2A and Guardrails.
"""
from typing import Any, Dict, Tuple, Optional, List
from uuid import uuid4

from app.services.aidaan.base import BaseAgent
from app.services.aidaan.a2a_service import a2a_service
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
        # Merge incoming context (if any) with session context
        if context:
            session_context.update(context)
            session_manager.save_context(conv_id, session_context)

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
        
        # 4. Fallback: Coordinator handles generic queries or RFQ drafts.
        lowered_text = text.lower()
        if "rfq" in lowered_text:
            return AidaanMessageResponse(
                reply="I can help draft an RFQ. What are the details of the request?",
                bullets=[],
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
            
        return AidaanMessageResponse(
            reply="I'm here to assist with your trading needs. Could you please clarify your request?",
            bullets=["Phase 1 modularity active", "Agent Registry initialized"],
            actions=[
                ActionItem(
                    kind="ask_clarification",
                    title="Ask for Clarification",
                    payload={"clarification": "string"},
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
        if "risk" in lowered:
            return "risk", 1.0
        if "fund" in lowered or "distribute" in lowered:
            return "distributor", 1.0
        if "sentiment" in lowered:
            return "risk", 1.0

        return None, 0.0
    
    def get_capabilities(self) -> List[str]:
        return [
            "intent_classification",
            "agent_delegation",
            "compliance_guardrails",
            "rfq_drafting"
        ]
    
coordinator_agent = CoordinatorAgent()
