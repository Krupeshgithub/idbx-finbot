"""
Institutional Coordinator Agent - AIDAAN Core Orchestrator
==========================================================
The Coordinator acts as the central nervous system of AIDAAN. It is responsible 
for intent classification, session management, and routing requests to 
specialized agents (Market, Risk, Order, Greeting).

Key Features:
- Heuristic-first routing for extreme latency/token efficiency.
- LLM fallback for ambiguous request classification.
- Centralized agent registration and lifecycle management.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.services.aidaan.core.base import BaseAgent, registry
from app.schemas.aidaan import AidaanMessageResponse
from app.core.prompts import Prompts

# Import specialized agents for registration
from app.services.aidaan.agents.market.market_agent import market_agent
from app.services.aidaan.agents.risk.risk_agent import risk_agent
from app.services.aidaan.agents.order.order_agent import order_agent
from app.services.aidaan.agents.greeting.greeting_agent import greeting_agent

# Register agents with the global registry
registry.register("market", market_agent)
registry.register("risk", risk_agent)
registry.register("order", order_agent)
registry.register("greeting", greeting_agent)

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Orchestrates user requests by intelligently routing them to specialized agents.
    """

    def __init__(self):
        """
        Initialize the Coordinator with a high-capacity model for routing tasks.
        """
        super().__init__(name="coordinator", model_name="gemini-1.5-pro")

    async def handle_message(
        self, 
        text: str, 
        conversation_id: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Entry point for all user messages. Determines the intent and delegates work.

        Args:
            text: Raw user input from the chat/voice interface.
            conversation_id: Unique session identifier for context retention.
            context: Additional metadata (username, desk_id, etc.)

        Returns:
            AidaanMessageResponse: Synthesized response from a specialized agent.
        """
        conv_id = conversation_id or f"conv-{uuid4().hex[:8]}"
        logger.info(f"[Coordinator] Handling session {conv_id}: {text[:50]}...")

        # 1. Routing phase
        agent_id = await self._route_intent(text)
        
        target_agent = registry.get_agent(agent_id or "greeting")
        if target_agent:
            return await target_agent.handle_message(
                text=text,
                conversation_id=conv_id,
                context=context
            )

        # Fallback response if no agent could handle the request
        return AidaanMessageResponse(
            reply="I'm here to help with your trading desk operations. You can ask about market analysis, risk metrics, or staging trade RFQs.",
            bullets=["Self-aware session active", "Token-optimized routing"],
            actions=[],
            conversation_id=conv_id,
            model={"agent": self.name, "llm": "fallback"}
        )

    async def _route_intent(self, text: str) -> Optional[str]:
        """
        Classification logic for routing requests.
        Priority:
        1. Heuristics (Regex/Keyword) - zero token cost, instant.
        2. LLM Classification - handles natural language ambiguity.

        Returns:
            The ID of the target agent or None.
        """
        lowered = text.lower()
        
        # --- Heuristics (Token Saving) ---
        if any(k in lowered for k in ["hi", "hello", "good morning", "hey"]):
            return "greeting"
        if any(k in lowered for k in ["price", "market", "gold", "oil", "inflation", "gdp"]):
            return "market"
        if any(k in lowered for k in ["risk", "dv01", "pv01", "limit"]):
            return "risk"
        if any(k in lowered for k in ["stage", "rfq", "buy", "sell", "sonia", "sofr", "order"]):
            return "order"

        # --- LLM Intent Check ---
        prompt = Prompts.COORDINATOR_ROUTER.format(text=text)
        try:
            parsed = await self.llm.generate_json(prompt)
            return parsed.get("intent")
        except Exception as e:
            logger.warning(f"[Coordinator] LLM routing failed: {e}")
            return None

    def get_capabilities(self) -> List[str]:
        """
        Returns the core capabilities of the Coordinator.
        """
        return ["intent_routing", "agent_orchestration", "multi_agent_showcase"]


# Singleton instance
coordinator_agent = CoordinatorAgent()
