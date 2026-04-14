"""
Coordinator Agent for AIDAAN
==============================
The "Brain" — classifies intent and delegates to specialized agents.

CHANGES FROM ORIGINAL:
- Intent classifier prompt moved to `Prompts.INTENT_CLASSIFIER`.
- LLM call uses `llm_client.generate_json_sync()` instead of raw SDK.
- `_classifier_executor` removed — handled inside llm_client.
- Static reply strings replaced with `Prompts.*` constants.
"""
import logging
from typing import Any, Dict, Optional, Tuple, List
from uuid import uuid4

from app.services.aidaan.base import BaseAgent
from app.services.aidaan.registry import registry
from app.services.aidaan.session_manager import session_manager
from app.core.guardrails import guardrails
from app.core.prompts import Prompts
from app.schemas.aidaan import ActionItem, AidaanMessageResponse
from app.core.config.settings import settings
from app.services.aidaan.risk_agent import risk_agent
from app.services.aidaan.market_agent import market_agent
from app.services.aidaan.distributor_agent import distributor_agent
from app.services.aidaan.greeting_agent import greeting_agent

# Register all agents at startup
registry.register("risk", risk_agent)
registry.register("distributor", distributor_agent)
registry.register("market", market_agent)
registry.register("greeting", greeting_agent)

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Orchestrates the conversation and delegates to specialized agents.
    """

    def __init__(self):
        super().__init__(
            name="coordinator",
            model_name=settings.VERTEX_AI_MODEL_NAME
        )

    async def handle_message(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Main entry point: apply guardrails → classify intent → delegate.
        """
        conv_id = conversation_id or f"conv-{uuid4().hex[:8]}"
        logger.info(f"[Coordinator] conv_id={conv_id} | text={text!r}")

        # Load + merge session context
        session_context = session_manager.get_context(conv_id)
        if context:
            session_context.update(context)
            session_manager.save_context(conv_id, session_context)

        # --- Action Callback: user clicked a button ---
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

        # --- Guardrails check ---
        is_advisory, reason = guardrails.is_advisory(text)
        if is_advisory:
            return AidaanMessageResponse(
                reply=f"I cannot provide financial advice. {Prompts.COMPLIANCE_NOTICE}",
                bullets=[f"Compliance Block: {reason}"],
                actions=[],
                conversation_id=conv_id,
                model=self.get_model_info()
            )

        # --- Intent classification → agent delegation ---
        agent_id, _ = self._classify_intent(text)
        if agent_id:
            target_agent = registry.get_agent(agent_id)
            if target_agent:
                return await target_agent.handle_message(
                    text=text,
                    conversation_id=conv_id,
                    context=session_context
                )

        # --- Static Phase 1 fallback handlers ---
        if "rfq" in text.lower():
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
            reply=Prompts.FALLBACK_GENERAL_STANDBY,
            bullets=["Phase 1 modularity active", f"Model: {settings.VERTEX_AI_MODEL_NAME}"],
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

    def _classify_intent(self, text: str) -> Tuple[Optional[str], float]:
        """
        Gemini-powered intent classification.
        Uses Prompts.INTENT_CLASSIFIER — edit the prompt there, not here.

        Returns:
            (agent_id, confidence) e.g. ("market", 0.9) or (None, 0.0)
        """
        prompt = Prompts.INTENT_CLASSIFIER.format(text=text)

        try:
            parsed = self.llm.generate_json_sync(prompt, timeout=8)

            if "error" in parsed:
                logger.warning(f"[Coordinator._classify_intent] LLM error: {parsed['error']}")
                return self._keyword_fallback(text)

            intent = parsed.get("intent", "general")
            confidence = float(parsed.get("confidence", 0.8))
            reason = parsed.get("reason", "")
            logger.info(
                f"[Coordinator._classify_intent] intent={intent} "
                f"confidence={confidence:.2f} reason={reason!r}"
            )

            if intent in ("market", "risk", "distributor", "greeting"):
                return intent, confidence
            return None, 0.0

        except Exception as e:
            logger.error(
                f"[Coordinator._classify_intent] Unexpected error: {type(e).__name__}: {e}. "
                "Falling back to keywords."
            )
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> Tuple[Optional[str], float]:
        """
        Hard keyword fallback — only used if Gemini classifier fails.
        """
        lowered = text.lower()

        market_triggers = [
            "price", "quote", "ohlcv", "intraday", "earnings", "market cap",
            "52-week", "stock", "ticker", "aapl", "msft", "tsla", "amzn",
            "googl", "nvda", "avgo", "nflx", "amd", "meta", "hdb", "infy",
            "apple", "microsoft", "tesla", "amazon", "nvidia", "netflix",
            "broadcom", "alphabet", "infosys", "hdfc bank", "home depot",
        ]
        if any(k in lowered for k in market_triggers):
            return "market", 0.7

        if any(k in lowered for k in ["risk", "sentiment", "volatility", "limits"]):
            return "risk", 0.7

        if any(k in lowered for k in ["liquidity", "rfq", "distribute", "fund overnight"]):
            return "distributor", 0.7

        return None, 0.0

    def get_capabilities(self) -> List[str]:
        return [
            "intent_classification",
            "agent_delegation",
            "compliance_guardrails",
            "rfq_drafting"
        ]


coordinator_agent = CoordinatorAgent()
