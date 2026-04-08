"""
Coordinator Agent Implementation for AIDAAN
The "Brain" of the operation, now with A2A and Guardrails
"""
import re
import json
import logging
from typing import Any, Dict, Tuple, Optional, List
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from app.services.aidaan.base import BaseAgent
from app.services.aidaan.registry import registry
from app.services.aidaan.session_manager import session_manager
from app.core.guardrails import guardrails
from app.schemas.aidaan import (
    ActionItem,
    AidaanMessageResponse
)

from app.core.config.settings import settings
from app.services.aidaan.risk_agent import risk_agent
from app.services.aidaan.market_agent import market_agent
from app.services.aidaan.distributor_agent import distributor_agent


# Registry configuration
registry.register("risk", risk_agent)
registry.register("distributor", distributor_agent)
registry.register("market", market_agent)


logger = logging.getLogger(__name__)
_classifier_executor = ThreadPoolExecutor(max_workers=2)


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
        Process user intent and return a structured response
        with A2A and Guardrails logic.
        """
        conv_id = conversation_id or f"conv-{uuid4().hex[:8]}"

        logger.info(f"[DEBUG] Incoming text: {text}")

        # Load existing context/session state
        session_context = session_manager.get_context(conv_id)
        if context:
            session_context.update(context)
            session_manager.save_context(conv_id, session_context)
        
        # Action Callback Handler
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
        
        is_advisory, reason = guardrails.is_advisory(text)
        if is_advisory:
            return AidaanMessageResponse(
                reply=f"I cannot provide financial advice. {guardrails.get_compliance_notice()}",
                bullets=[f"Compliance Block: {reason}"],
                actions=[],
                conversation_id=conv_id,
                model=self.get_model_info()
            )

        # Intent Classification
        agent_id, _ = self._classify_intent(text)

        # Delegate to Specialized Agent
        if agent_id:
            target_agent = registry.get_agent(agent_id)
            if target_agent:
                return await target_agent.handle_message(
                    text=text,
                    conversation_id=conv_id,
                    context=session_context
                )
            
        # Phase 1 Static Handlers
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

    def _classify_intent(self, text: str) -> Tuple[Optional[str], float]:
        """
        Gemini-powered intent classifier.
        Decides: market | risk | distributor | general
        """
        prompt = f"""You are an intent classifier for an institutional trading assistant called AIDAAN.

            Classify this trader message into exactly ONE category.

            Message: "{text}"

            Categories:
            1. "market"      - Anything about stock prices, quotes, company performance, historical data,
                            tickers, company names (Apple, HDFC, Nvidia etc), indices, charts, OHLCV,
                            earnings, market cap, volume, intraday/daily/weekly data.
                            Examples: "NVDA", "Apple price", "How is Tesla doing?", "HDFC Bank",
                                        "Show me Google's chart", "What was Microsoft last week?"

            2. "risk"        - Desk risk checks, pre-trade risk, sentiment analysis, volatility assessment,
                            limit checks, compliance checks.
                            Examples: "Check my desk risk", "What's market sentiment?",
                                        "Am I within limits?", "Check status"

            3. "distributor" - Liquidity discovery, RFQ distribution, trade funding, matching,
                            finding counterparties.
                            Examples: "Find liquidity", "Distribute this RFQ", "Fund 5bn overnight",
                                        "Who can price EUR/PLN?"

            4. "general"     - Greetings, help requests, unclear messages, non-market questions.
                            Examples: "Hello", "What can you do?", "Help me"

            Return ONLY raw JSON. No markdown. No explanation.
            Format: {{"intent": "market|risk|distributor|general", "confidence": 0.0-1.0, "reason": "one line why"}}

            Classify: "{text}"
        """
        try:
            # run_in_executor returns a coroutine -- run it synchronously here
            future = _classifier_executor.submit(
                self.model.generate_content, 
                prompt
            )
            result_raw = future.result(timeout=8)
            raw = result_raw.text.strip()

            # Clean JSON
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                intent     = parsed.get("intent", "general")
                confidence = float(parsed.get("confidence", 0.8))
                reason     = parsed.get("reason", "")
                logger.info(f"[Classifier] intent={intent} conf={confidence} reason={reason}")

                if intent == "market":
                    return "market", confidence
                elif intent == "risk":
                    return "risk", confidence
                elif intent == "distributor":
                    return "distributor", confidence
                else:
                    return None, 0.0

        except Exception as e:
            logger.error(f"[Classifier] Gemini failed: {type(e).__name__}: {e}. Falling back to keywords.")
            return self._keyword_fallback(text)

        return None, 0.0

    def _keyword_fallback(self, text: str) -> Tuple[Optional[str], float]:
        """
        Hard keyword fallback — only runs if Gemini classifier fails.
        Much tighter than before to avoid false positives.
        """
        lowered = text.lower()

        # Only very specific financial terms — no generic words like "up/down/market"
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
