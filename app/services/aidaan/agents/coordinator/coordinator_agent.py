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
from app.services.aidaan.agents.operational.operational_agent import operational_agent
from app.services.aidaan.runtime_context import runtime_context_service

# Register agents with the global registry
registry.register("market", market_agent)
registry.register("risk", risk_agent)
registry.register("order", order_agent)
registry.register("greeting", greeting_agent)
registry.register("context", context_agent)
registry.register("operational", operational_agent)

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

    @staticmethod
    def _get_last_specialist_agent(conversation_id: Optional[str]) -> Optional[str]:
        """
        Recover the last non-greeting specialist from persisted history.
        """
        if not conversation_id:
            return None

        history = runtime_context_service.get_recent_history(conversation_id)
        for item in reversed(history):
            agent_name = item.get("agent_name")
            if agent_name in {"market", "risk", "order", "operational", "context"}:
                return "operational" if agent_name == "context" else agent_name
        return None

    @staticmethod
    def _has_pending_follow_up(conversation_id: Optional[str]) -> bool:
        """
        Detect whether the previous assistant turn left an unanswered follow-up prompt.
        """
        if not conversation_id:
            return False

        history = runtime_context_service.get_recent_history(conversation_id)
        guidance = runtime_context_service.build_continuity_guidance(history)
        return bool(guidance.get("pending_follow_up"))

    @staticmethod
    def _should_escalate_route_check(text: str, parsed: Dict[str, Any]) -> bool:
        """
        Escalate ambiguous turns to the reasoning model for better continuity.
        """
        word_count = len(text.strip().split())
        confidence = float(parsed.get("confidence") or 0.0)
        return (
            word_count <= 4
            or confidence < 0.75
            or bool(parsed.get("is_follow_up"))
            or bool(parsed.get("is_history_query"))
            or (
                parsed.get("intent") == "greeting"
                and not bool(parsed.get("is_standalone_greeting"))
            )
        )

    async def _classify_route(
        self,
        *,
        text: str,
        conversation_id: Optional[str],
        username: Optional[str],
        model_name: str,
    ) -> Dict[str, Any]:
        prompt = Prompts.COORDINATOR_ROUTER.format(text=text)
        return await self.generate_json_response(
            prompt,
            conversation_id=conversation_id,
            username=username,
            response_schema=Prompts.ROUTING_DECISION_SCHEMA,
            system_instruction=(
                Prompts.TRADER_SYSTEM_INSTRUCTION
                + "\nRole: Intent Classifier and continuity-aware router."
                + " Use recent conversation memory to decide whether the message is a follow-up."
                + " Never label a context-dependent acknowledgement or clarification as a greeting."
            ),
            model_override=model_name,
        )

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
        lowered = text.strip().lower()

        # --- Primary Path: Heuristic-first routing (0ms latency) ---
        if lowered in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}:
            return "greeting"
            
        # Basic keyword matching for obvious intents to save LLM calls
        if lowered.startswith("stage ") or lowered.startswith("rfq ") or lowered.startswith("buy ") or lowered.startswith("sell "):
            return "order"
        if any(k in lowered for k in ["dv01", "pv01", "var 95", "stress test"]):
            return "risk"
        if any(k in lowered for k in ["my history", "my previous", "audit log"]):
            return "operational"

        last_specialist_agent = self._get_last_specialist_agent(conversation_id)
        if (
            last_specialist_agent
            and self._has_pending_follow_up(conversation_id)
            and len(lowered.split()) <= 6
        ):
            return last_specialist_agent

        # --- Secondary Path: Fast LLM intent routing ---
        # Use the dedicated router model first, then escalate ambiguous continuity checks.
        try:
            parsed = await self._classify_route(
                text=text,
                conversation_id=conversation_id,
                username=username,
                model_name=settings.VERTEX_AI_ROUTER_MODEL_NAME,
            )

            if self._should_escalate_route_check(text, parsed):
                parsed = await self._classify_route(
                    text=text,
                    conversation_id=conversation_id,
                    username=username,
                    model_name=settings.VERTEX_AI_REASONING_MODEL_NAME,
                )

            intent = parsed.get("intent") or "greeting"
            is_follow_up = bool(parsed.get("is_follow_up"))
            is_history_query = bool(parsed.get("is_history_query"))
            is_standalone_greeting = bool(parsed.get("is_standalone_greeting"))

            if is_history_query or intent == "context":
                return "operational"

            if is_follow_up and last_specialist_agent:
                return last_specialist_agent

            if intent == "greeting" and not is_standalone_greeting and last_specialist_agent:
                return last_specialist_agent

            return intent
        except Exception as e:
            logger.warning(f"[Coordinator] LLM routing failed: {e}")
            # --- Ultimate fallback ---
            if last_specialist_agent and len(lowered.split()) <= 4:
                return last_specialist_agent
            if any(k in lowered for k in ["risk", "compliance"]):
                return "risk"
            if any(k in lowered for k in ["price", "market", "stock", "news", "quote"]):
                return "market"
            if any(k in lowered for k in ["order", "execution"]):
                return "order"
            return None

    def get_capabilities(self) -> List[str]:
        """
        Returns the core capabilities of the Coordinator.
        """
        return ["intent_routing", "agent_orchestration", "multi_agent_showcase"]


# Singleton instance
coordinator_agent = CoordinatorAgent()
