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
from typing import Any, Dict, List, Optional
from uuid import uuid4
import time

from app.services.aidaan.core.base import BaseAgent, registry
from app.schemas.aidaan import AidaanMessageResponse
from app.core.prompts import Prompts
from app.core.config.settings import settings

# Import specialized agents for registration
from app.services.aidaan.agents.market.market_agent import market_agent
from app.services.aidaan.agents.risk.risk_agent import risk_agent
from app.services.aidaan.agents.order.order_agent import order_agent
from app.services.aidaan.agents.greeting.greeting_agent import greeting_agent
from app.services.aidaan.agents.context.context_agent import context_agent
from app.services.aidaan.runtime_context import runtime_context_service

# Register agents with the global registry
registry.register("market", market_agent)
registry.register("risk", risk_agent)
registry.register("order", order_agent)
registry.register("greeting", greeting_agent)
registry.register("context", context_agent)

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Orchestrates user requests by intelligently routing them to specialized agents.
    """

    def __init__(self):
        """
        Initialize the Coordinator with a high-capacity model for routing tasks.
        """
        super().__init__(name="coordinator", model_name=settings.VERTEX_AI_MODEL_NAME)

    async def handle_message(
        self, 
        text: str, 
        conversation_id: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
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
        start_time = time.monotonic()

        # 1. Routing phase
        context = context or {}
        agent_id = await self._route_intent(
            text,
            conversation_id=conv_id,
            username=context.get("username"),
        )
        
        target_agent = registry.get_agent(agent_id or "greeting")
        if target_agent:
            response = await target_agent.handle_message(
                text=text,
                conversation_id=conv_id,
                context=context,
                tool_callback=tool_callback,
            )
            # Inject latency into the final response
            response.latency_ms = (time.monotonic() - start_time) * 1000
            return response

        # Fallback response if no agent could handle the request
        latency_ms = (time.monotonic() - start_time) * 1000
        return self.build_message_response(
            reply="I'm here to help with your trading desk operations. You can ask about market analysis, risk metrics, or staging trade RFQs.",
            bullets=["Self-aware session active", "Token-optimized routing"],
            conversation_id=conv_id,
            model_info={"agent": self.name, "llm": "fallback"},
            latency_ms=latency_ms,
        )

    async def _route_intent(
        self,
        text: str,
        *,
        conversation_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Optional[str]:
        """
        Classification logic for routing requests.
        Priority:
        1. Heuristics (Regex/Keyword) - zero token cost, instant.
        2. LLM Classification - handles natural language ambiguity.

        Returns:
            The ID of the target agent or None.
        """
        lowered = text.lower()
        
        # --- 1. Comprehensive Market & Risk Priority (Longer queries) ---
        market_keywords = [
            "price", "market", "gold", "oil", "inflation", "gdp", "apple", "google", "meta", "tesla", 
            "microsoft", "amazon", "nvidia", "aapl", "goog", "msft", "tsla", "nvda", "index", "sp500", 
            "nasdaq", "dow", "ftse", "dax", "nifty", "dividend", "yield", "earnings", "eps", "pe ratio", 
            "revenue", "balance sheet", "income statement", "cash flow", "commodity", "crude", "silver", "news", "sentiment"
        ]
        risk_keywords = ["risk", "dv01", "pv01", "limit", "exposure", "var", "stress", "compliance"]
        
        # If query is substantial, check market/risk keywords first to avoid false-positive greeting triggers
        if len(lowered.split()) > 3:
            if any(k in lowered for k in market_keywords):
                return "market"
            if any(k in lowered for k in risk_keywords):
                return "risk"

        # --- 2. Heuristics (Token Saving / High Speed) ---
        if any(k in lowered for k in ["hi", "hello", "good morning", "hey", "greeting"]):
            return "greeting"
        
        # fallback keywords for short queries
        if any(k in lowered for k in market_keywords):
            return "market"
            
        if any(k in lowered for k in risk_keywords):
            return "risk"
            
        if any(k in lowered for k in ["stage", "rfq", "buy", "sell", "sonia", "sofr", "order", "execution", "quote"]):
            return "order"
            
        if any(k in lowered for k in ["history", "previous", "earlier", "desk", "profile", "counterparty", "context", "session", "last question", "memory"]):
            return "context"

        # --- LLM Intent Check ---
        prompt = Prompts.COORDINATOR_ROUTER.format(text=text)
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=username,
                response_schema=Prompts.INTENT_SCHEMA,
                system_instruction=Prompts.TRADER_SYSTEM_INSTRUCTION + "\nRole: Intent Classifier"
            )
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
