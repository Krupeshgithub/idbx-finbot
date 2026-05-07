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
import re

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
from app.services.aidaan.agents.calculation.calculation_agent import calculation_agent
from app.services.aidaan.runtime_context import runtime_context_service
from app.services.kill_switch import kill_switch_service
from app.core.guardrail_orchestrator import guardrail_orchestrator

# Register agents with the global registry
registry.register("market", market_agent)
registry.register("risk", risk_agent)
registry.register("order", order_agent)
registry.register("greeting", greeting_agent)
registry.register("context", context_agent)
registry.register("operational", operational_agent)
registry.register("calculation", calculation_agent)

logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("performance")


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
    def _default_routing_decision(intent: str, reason: str, **overrides: Any) -> Dict[str, Any]:
        decision = {
            "intent": intent,
            "sub_intent": "general",
            "confidence": 0.9,
            "control_signal": "none",
            "is_follow_up": False,
            "is_history_query": False,
            "is_standalone_greeting": False,
            "reason": reason,
        }
        decision.update(overrides)
        return decision

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize user text for lightweight heuristic checks.
        """
        lowered = text.strip().lower()
        compact = re.sub(r"(.)\1{2,}", r"\1", lowered)
        compact = re.sub(r"\s+", " ", compact).strip()
        return compact

    def _is_greeting_phrase(self, compact_text: str) -> bool:
        """
        Detect standalone greeting variants without paying an LLM routing cost.

        Supports common address forms such as "hello aidaan" or "hi bot"
        while avoiding longer conversational turns that should go through
        normal routing.
        """
        greeting_set = {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening",
            "good afternoon",
        }
        greeting_prefixes = (
            "hi ",
            "hello ",
            "hey ",
            "good morning ",
            "good evening ",
            "good afternoon ",
        )
        assistant_names = {"aidaan", "bot", "assistant", "team"}

        if compact_text in greeting_set:
            logger.info("[Coordinator] Greeting heuristic matched exact phrase | text=%s", compact_text)
            return True

        if compact_text.startswith(greeting_prefixes):
            trailing_tokens = compact_text.split()[1:]
            if trailing_tokens and all(token in assistant_names for token in trailing_tokens):
                logger.info(
                    "[Coordinator] Greeting heuristic matched addressed phrase | text=%s",
                    compact_text,
                )
                return True

        return False

    def _is_explicit_fresh_query(self, lowered_text: str) -> bool:
        """
        Detect whether a short message is actually a fresh user request.

        This guard prevents pending follow-up continuity from swallowing
        short but explicit queries such as "Should I buy LSEG now?".
        """
        explicit_phrases = [
            "should i",
            "should we",
            "what is happening",
            "why is",
            "what happened",
            "compare ",
            "analyze ",
            "analyse ",
            "about ",
            "tell me about",
            "ke bare me",
            "ke baare me",
            "batao",
            "batado",
            "batato",
            "latest ",
            "current ",
            "price ",
            "news ",
            "outlook ",
            "trend ",
            "rsi ",
            "macd ",
        ]
        explicit_keywords = {
            "buy",
            "sell",
            "price",
            "news",
            "compare",
            "analysis",
            "analyze",
            "analyse",
            "latest",
            "today",
            "now",
            "quote",
            "outlook",
            "trend",
            "why",
            "rsi",
            "macd",
            "support",
            "resistance",
            "about",
            "batao",
            "baare",
        }

        if any(phrase in lowered_text for phrase in explicit_phrases):
            logger.info(
                "[Coordinator] Fresh-query guard matched explicit phrase | text=%s",
                lowered_text[:120],
            )
            return True

        tokens = set(re.findall(r"[a-z0-9]+", lowered_text))
        if tokens & explicit_keywords:
            logger.info(
                "[Coordinator] Fresh-query guard matched keyword set | keywords=%s text=%s",
                sorted(tokens & explicit_keywords),
                lowered_text[:120],
            )
            return True

        return False

    @staticmethod
    def _is_follow_up_bridge_message(lowered_text: str) -> bool:
        """
        Detect short continuity bridges that genuinely refer to the prior thread.

        This prevents arbitrary short messages (e.g., random names) from being
        auto-routed to an existing pending follow-up owner.
        """
        normalized = re.sub(r"\s+", " ", lowered_text.strip().lower())
        if not normalized:
            return False

        bridge_exact = {
            "yes", "yeah", "yep", "sure", "ok", "okay", "continue", "go ahead",
            "do it", "proceed", "carry on", "same", "same one",
            "that one", "this one", "more", "details", "tell me more",
        }
        if normalized in bridge_exact:
            return True

        bridge_patterns = [
            r"^(continue|proceed)\b",
            r"^(go ahead|do it)\b",
            r"^(yes|yeah|yep|sure|ok|okay)\b",
            r"^(more|details|elaborate)\b",
            r"^(that|this)\s+(one|thread|flow|analysis)\b",
            r"^(same|same one|same thread)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in bridge_patterns)

    def _normalize_history_route(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep history and memory lookups on a single top-level route.

        The external router schema may still emit "context", but downstream
        execution should consistently use the operational specialist.
        """
        intent = decision.get("intent")
        sub_intent = decision.get("sub_intent")
        is_history_query = bool(decision.get("is_history_query"))
        if is_history_query or (intent == "context" and sub_intent == "history_lookup"):
            logger.info(
                "[Coordinator] Normalizing history route to operational | prior_intent=%s sub_intent=%s reason=%s",
                intent,
                decision.get("sub_intent"),
                decision.get("reason"),
            )
            decision["intent"] = "operational"
            decision["sub_intent"] = "history_lookup"
            decision["is_history_query"] = True
        return decision

    def _is_direct_history_query(self, lowered_text: str) -> bool:
        """
        Detect direct conversation-memory requests without over-triggering on
        casual words like "earlier" inside unrelated questions.
        """
        normalized_text = self._normalize_text(lowered_text)
        history_patterns = [
            r"^what did i ask\b",
            r"^what was my last question\b",
            r"^what did i say\b",
            r"^what did we discuss\b",
            r"^what were we discussing\b",
            r"^summari[sz]e (our|the) (chat|conversation|history)\b",
            r"^show (me )?(my|the) (history|previous questions|recent questions)\b",
            r"^remind me what i asked\b",
            r"^what happened earlier in (this|the) (chat|conversation)\b",
            # "give me last N questions" variants
            r"^give me (my |the )?(last|previous|recent) \d+ questions?\b",
            r"^(last|previous|recent) \d+ questions?\b",
            r"^show (me )?(last|previous|recent) \d+ questions?\b",
            r"^(meri|mere|mera) (last|pichle|purani) \d+ questions?\b",
            # "my questions" / "my history"
            r"^(give me |show me |list )?(my|the) (questions|history|chat history|conversation)\b",
            r"^(what|which) questions? (did i|have i) ask",
            r"^(list|show) (all |my )?(previous |recent |last )?(questions?|queries)\b",
        ]
        matched_pattern = next((pattern for pattern in history_patterns if re.search(pattern, normalized_text)), None)
        if matched_pattern:
            logger.info(
                "[Coordinator] Direct history-query heuristic matched | pattern=%s text=%s",
                matched_pattern,
                normalized_text[:160],
            )
            return True
        return False

    def _is_conversation_logic_query(self, lowered_text: str) -> bool:
        """
        Detect questions about routing, follow-up ownership, or conversational
        interpretation rather than market content itself.
        """
        normalized_text = self._normalize_text(lowered_text)
        logic_patterns = [
            r"\bfresh question\b",
            r"\bcontinuation trap\b",
            r"\bwhich thread\b",
            r"\bwhich agent\b",
            r"\bhow would you classify\b",
            r"\bhow did you classify\b",
            r"\bhow would you interpret\b",
            r"\bwhat would happen if\b",
            r"\bignore your previous follow-up\b",
            r"\bi am not asking for history\b",
            r"\bshould that stop belong to\b",
        ]
        matched_pattern = next((pattern for pattern in logic_patterns if re.search(pattern, normalized_text)), None)
        if matched_pattern:
            logger.info(
                "[Coordinator] Conversation-logic heuristic matched | pattern=%s text=%s",
                matched_pattern,
                normalized_text[:160],
            )
            return True
        return False

    def _is_out_of_domain(self, lowered_text: str) -> bool:
        """
        Fast heuristic check for obviously out-of-domain questions.
        Rejects non-financial queries before expensive DB/LLM operations.
        
        Returns True if the query is clearly outside financial/trading domain.
        """
        normalized_text = self._normalize_text(lowered_text)
        
        # Personal/social questions that have no financial context
        out_of_domain_patterns = [
            r"\b(marriage|wedding|sadi|shadi)\b",
            r"\b(wife|husband|spouse|girlfriend|boyfriend)\b",
            r"\b(krishna|ram|shiva|jesus|allah|god|bhagwan|deity)\b",
            r"\b(horoscope|astrology|kundli|zodiac)\b",
            r"\b(recipe|cooking|food preparation)\b",
            r"\b(movie|film|cinema|bollywood|hollywood)\b",
            r"\b(cricket|football|sports|match score)\b",
            r"\b(weather forecast|temperature|rain)\b",
            r"\b(health|medical|doctor|disease|medicine)\b",
            r"\b(travel|vacation|holiday|tourism)\b",
        ]
        
        # Check if query contains ONLY out-of-domain terms with NO financial terms
        financial_terms = {
            "stock", "price", "market", "trade", "buy", "sell", "invest", "portfolio",
            "risk", "return", "bond", "equity", "forex", "crypto", "bitcoin", "eth",
            "ticker", "quote", "volume", "chart", "analysis", "rsi", "macd", "sma",
            "revenue", "earnings", "profit", "loss", "dividend", "yield", "rate",
            "fund", "asset", "liability", "capital", "liquidity", "hedge", "option",
            "future", "derivative", "commodity", "currency", "exchange", "index",
            # Conversation/session meta-queries — always in-domain for this assistant
            "question", "questions", "history", "asked", "previous", "last", "recent",
            "conversation", "chat", "session", "discussed", "said", "told",
        }
        
        # If text contains any financial term, it's in-domain
        if any(term in normalized_text for term in financial_terms):
            return False
        
        # Check for out-of-domain patterns
        for pattern in out_of_domain_patterns:
            if re.search(pattern, normalized_text):
                logger.info(
                    "[Coordinator] Out-of-domain heuristic matched | pattern=%s text=%s",
                    pattern,
                    normalized_text[:160],
                )
                return True
        
        # Additional check: Personal information without financial context
        personal_info_patterns = [
            r"^(my|mera|meri)\s+(name|age|height|weight|looks?|appearance)",
            r"^(i am|main)\s+(handsome|beautiful|smart|tall|short)",
            r"(kab|when)\s+(hogi|hoga|will happen|will be)",
        ]
        
        for pattern in personal_info_patterns:
            if re.search(pattern, normalized_text):
                logger.info(
                    "[Coordinator] Personal info without financial context | pattern=%s text=%s",
                    pattern,
                    normalized_text[:160],
                )
                return True
        
        return False

    def _heuristic_route_decision(
        self,
        text: str,
        *,
        last_specialist_agent: Optional[str],
        has_pending_follow_up: bool,
    ) -> Optional[Dict[str, Any]]:
        lowered = text.strip().lower()
        if not lowered:
            return None

        def _log_heuristic(decision: Dict[str, Any], trigger: str) -> Dict[str, Any]:
            logger.info(
                "[Coordinator] Heuristic route matched | trigger=%s intent=%s sub_intent=%s control=%s confidence=%.2f follow_up=%s reason=%s",
                trigger,
                decision.get("intent"),
                decision.get("sub_intent"),
                decision.get("control_signal"),
                float(decision.get("confidence") or 0.0),
                decision.get("is_follow_up"),
                decision.get("reason"),
            )
            return decision

        # NEW: FX Spread Detection (MUST come before general market keywords)
        fx_spread_patterns = [
            r"\b(bid.?ask|bid/ask|spread)\b.*\b(eur|usd|gbp|jpy|chf|aud|cad|nzd|forex|fx|currency|pair)\b",
            r"\b(eur|usd|gbp|jpy|chf|aud|cad|nzd)\s*/\s*(eur|usd|gbp|jpy|chf|aud|cad|nzd)\b.*\b(spread|bid|ask|quote)\b",
            r"\bcurrent\b.*\b(bid.?ask|spread)\b.*\b(eur|usd|gbp|jpy|forex|fx)\b",
            r"\b(forex|fx|currency)\b.*\b(spread|bid.?ask|quote)\b",
        ]
        
        if any(re.search(p, lowered, re.IGNORECASE) for p in fx_spread_patterns):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "FX spread query matched heuristic.",
                sub_intent="fx_analysis",
                confidence=0.97,
            ), "fx_spread_keyword")

        # NEW: FX Exchange Rate Detection (for simple rate queries)
        fx_rate_patterns = [
            r"\b(exchange rate|conversion rate|rate)\b.*\b(eur|usd|gbp|jpy|chf|aud|cad|nzd)\b",
            r"\b(eur|usd|gbp|jpy|chf|aud|cad|nzd)\s*(to|vs|versus|against)\s*(eur|usd|gbp|jpy|chf|aud|cad|nzd)\b",
            r"\bconvert\b.*\b(eur|usd|gbp|jpy|chf|aud|cad|nzd)\b",
        ]
        
        if any(re.search(p, lowered, re.IGNORECASE) for p in fx_rate_patterns):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "FX exchange rate query matched heuristic.",
                sub_intent="fx_analysis",
                confidence=0.95,
            ), "fx_rate_keyword")

        # NEW: TOP STOCKS / RANKINGS DETECTION (CRITICAL - High Priority)
        # Patterns for "top stocks", "largest companies", "biggest stocks", etc.
        top_stocks_patterns = [
            r"\btop\s+\d*\s*(stocks?|companies|equities|shares)\b",
            r"\b(largest|biggest|top)\s+(us|american|usa)?\s*(stocks?|companies)\b",
            r"\b(best|top)\s+performing\s+stocks?\b",
            r"\bmarket\s+leaders?\b",
            r"\bmega.?cap\s+stocks?\b",
            r"\bblue.?chip\s+stocks?\b",
            r"\bs&p\s*500\s+top\b",
            r"\bnasdaq\s+top\b",
            r"\bmost\s+valuable\s+companies\b",
        ]
        
        # Also check for specific ranking queries
        ranking_keywords = {
            "top stocks", "largest stocks", "biggest stocks", "top companies",
            "largest companies", "biggest companies", "market leaders",
            "top 10", "top 5", "top 20", "top ten", "top five",
        }
        
        has_top_stocks_pattern = any(re.search(p, lowered, re.IGNORECASE) for p in top_stocks_patterns)
        has_ranking_keyword = any(keyword in lowered for keyword in ranking_keywords)
        
        if has_top_stocks_pattern or has_ranking_keyword:
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Top stocks/rankings query matched heuristic (market cap rankings).",
                sub_intent="market_analysis",
                confidence=0.98,
            ), "top_stocks_ranking")

        # NEW: TOP GAINERS/LOSERS DETECTION
        gainers_losers_patterns = [
            r"\btop\s+gainers?\b",
            r"\btop\s+losers?\b",
            r"\bmost\s+active\b",
            r"\bbiggest\s+movers?\b",
            r"\blargest\s+gains?\b",
            r"\blargest\s+losses?\b",
        ]
        
        if any(re.search(p, lowered, re.IGNORECASE) for p in gainers_losers_patterns):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Top gainers/losers query matched heuristic.",
                sub_intent="market_analysis",
                confidence=0.97,
            ), "top_gainers_losers")

        continue_set = {
            "yes", "y", "haan", "ha", "han", "yep", "yeah", "ok", "okay", "sure",
            "continue", "proceed", "do it", "go ahead"
        }
        stop_set = {"no", "n", "na", "nah", "nahi", "nope", "stop", "cancel"}
        clarify_set = {"oh", "what", "what?", "huh", "confused", "simplify"}

        compact = self._normalize_text(text)

        if self._is_greeting_phrase(compact):
            return _log_heuristic(self._default_routing_decision(
                "greeting",
                "Standalone greeting matched heuristic.",
                is_standalone_greeting=True,
                confidence=0.99,
            ), "greeting")

        # Corporate Knowledge Detection (IDBX/AIDANN internal queries)
        # Route questions about company, leadership, capabilities to operational agent.
        # IMPORTANT: If the same turn also contains clear market/trading signals,
        # do not force heuristic routing — let LLM router decide final ownership.
        corporate_keywords = [
            "chairman", "ceo", "leadership", "nicholas", "runcorn",
            "who made", "who created", "who built", "who developed",
            "what is idbx", "what is aidann", "tell me about idbx", "tell me about aidann",
            "company mission", "mission statement", "company info",
            "data privacy", "dlp", "data loss prevention",
            "can aidann execute", "can aidann trade", "aidann boundaries",
            "security boundaries", "execution boundaries",
            "aidann capabilities", "what can aidann do", "aidann features"
        ]

        market_or_trade_markers = {
            "stock", "price", "quote", "ticker", "market", "news", "rsi", "macd",
            "chart", "trend", "outlook", "buy", "sell", "aapl", "apple",
        }
        corporate_hit = any(keyword in lowered for keyword in corporate_keywords)
        has_brand_reference = bool(re.search(r"\b(idbx|aidann)\b", lowered))
        market_hit = any(marker in lowered for marker in market_or_trade_markers)

        if (corporate_hit or has_brand_reference) and market_hit:
            logger.info(
                "[Coordinator] Mixed corporate + market query detected; deferring to LLM router | text_preview=%s",
                lowered[:120],
            )
            return None

        if corporate_hit:
            return _log_heuristic(self._default_routing_decision(
                "operational",
                "Corporate knowledge query matched heuristic (IDBX/AIDANN internal information).",
                sub_intent="corporate_knowledge",
                confidence=0.97,
            ), "corporate_knowledge")

        # ------------------------------------------------------------------
        # Profile & preference recall heuristics (MUST run before "what is"
        # educational heuristics; otherwise we mis-route to Market).
        # ------------------------------------------------------------------
        if any(
            phrase in lowered
            for phrase in [
                "what is my desk",
                "what's my desk",
                "my risk limit",
                "default tenor",
            ]
        ):
            return _log_heuristic(self._default_routing_decision(
                "operational",
                "User profile recall query matched heuristic.",
                sub_intent="profile_recall",
                confidence=0.97,
            ), "profile_recall")

        if any(
            phrase in lowered
            for phrase in [
                "carrying trades overnight",
                "carry trades overnight",
                "holding risk overnight",
                "hold risk overnight",
                "carry positions past close",
                "hold positions overnight",
                "overnight risk",
            ]
        ):
            return _log_heuristic(self._default_routing_decision(
                "operational",
                "User preference recall (overnight risk) matched heuristic.",
                sub_intent="preference_recall",
                confidence=0.97,
            ), "preference_recall")

        if compact in continue_set | stop_set | clarify_set:
            control_signal = (
                "continue" if compact in continue_set
                else "stop" if compact in stop_set
                else "clarify"
            )
            target_intent = last_specialist_agent or "market"
            return _log_heuristic(self._default_routing_decision(
                target_intent,
                "Brief conversational control signal matched heuristic.",
                sub_intent="control",
                control_signal=control_signal,
                is_follow_up=bool(last_specialist_agent and has_pending_follow_up),
                confidence=0.98,
            ), "control_signal")

        if lowered.startswith(("stage ", "rfq ", "buy ", "sell ")):
            # If the staging request includes pre-conditions (fetch/calculate/consult/check),
            # route to market agent first — it has MCP tools + A2A to handle all pre-steps.
            # Only route directly to order agent for simple "stage X notional Y tenor Z" requests.
            has_preconditions = bool(re.search(
                r"\b(fetch|calculate|consult|check|first|before|if sentiment|run finbert|"
                r"dv01|limit|yield|news|sentiment|risk agent|FinBERT)\b",
                lowered,
            ))
            if has_preconditions:
                logger.info(
                    "[Coordinator] Stage/RFQ with pre-conditions detected — routing to market agent for orchestration | text_preview=%s",
                    lowered[:120],
                )
                return _log_heuristic(self._default_routing_decision(
                    "market",
                    "Stage/RFQ with pre-conditions (fetch/calculate/consult) — market agent orchestrates first.",
                    sub_intent="market_analysis",
                    confidence=0.97,
                ), "order_with_preconditions")
            return _log_heuristic(self._default_routing_decision(
                "order",
                "Execution or staging verb matched heuristic.",
                confidence=0.97,
            ), "order_keyword")

        # Phase 4 Optimization: Expanded keyword sets
        technical_keywords = {"sma", "macd", "rsi", "ema", "bollinger", "stochastic", "indicator", "chart", "trend"}
        risk_keywords = {"dv01", "pv01", "var 95", "stress test", "risk capital", "hard limit", "risk desk"}
        crypto_keywords = {"btc", "eth", "bitcoin", "ethereum", "solana", "sol", "xrp", "ripple", "crypto", "defi", "nft", "bnb", "doge", "dogecoin", "avax", "ada", "cardano"}
        macro_keywords = {"fed funds", "federal funds", "yield curve", "treasury", "10y yield", "2y yield", "cpi", "inflation", "gdp", "unemployment", "fomc", "basis points", "bps", "rate hike", "rate cut"}
        
        # Phase 4: New keyword sets for better heuristic coverage
        investment_keywords = {
            "invest", "investment", "portfolio", "allocation", "diversify", "diversification",
            "asset", "wealth", "retirement", "savings", "mutual fund", "etf", "index fund"
        }
        
        educational_keywords = {
            "explain", "what is", "how does", "tell me about", "overview", "basics",
            "introduction", "guide", "tutorial", "learn", "understand", "meaning"
        }
        
        comparison_keywords = {
            "compare", "comparison", "vs", "versus", "difference between", "better",
            "worse", "which one", "which is", "should i choose"
        }
        
        sentiment_keywords = {
            "sentiment", "bullish", "bearish", "optimistic", "pessimistic", "mood",
            "feeling", "opinion", "view", "outlook", "forecast"
        }

        # Priority 1: If both market and risk elements exist, route to market_agent (the orchestrator)
        if any(k in lowered for k in technical_keywords) and any(k in lowered for k in risk_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Complex multi-agent query detected. Routing to Market Orchestrator for tool coordination.",
                sub_intent="technical_indicator",
                confidence=0.99,
            ), "multi_agent_market_first")

        # Phase 4: Investment/allocation queries
        if any(k in lowered for k in investment_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Investment/allocation keyword matched heuristic.",
                sub_intent="education",
                confidence=0.92,
            ), "investment_keyword")

        # Phase 4: Educational queries
        if any(k in lowered for k in educational_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Educational keyword matched heuristic.",
                sub_intent="education",
                confidence=0.90,
            ), "educational_keyword")

        # Phase 4: Comparison queries
        if any(k in lowered for k in comparison_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Comparison keyword matched heuristic.",
                sub_intent="market_analysis",
                confidence=0.93,
            ), "comparison_keyword")

        # Phase 4: Sentiment queries
        if any(k in lowered for k in sentiment_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Sentiment analysis keyword matched heuristic.",
                sub_intent="market_analysis",
                confidence=0.91,
            ), "sentiment_keyword")

        # Calculation/Math queries
        # STRICT RULE: Only route to calculation agent when:
        # 1. Query contains explicit portfolio math keywords AND
        # 2. No live data fetch is needed (no fetch/pull/get verbs) AND
        # 3. No market/crypto/FX context is present
        # The calculation agent has NO MCP tools — it cannot fetch live data.
        calculation_keywords = {
            "sharpe ratio", "weighted beta", "value at risk",
            "portfolio optimization", "minimum variance",
            "rebalance my portfolio", "optimize my portfolio",
            "var 95", "var 99",
        }

        # Guard: broad set — if ANY of these are present, route to market/risk instead
        live_data_guard = {
            # Fetch verbs
            "fetch", "pull", "get ", "retrieve", "show me", "find",
            # Market instruments
            "bitcoin", "btc", "eth", "ethereum", "crypto", "altcoin",
            "aapl", "msft", "tsla", "nvda", "reliance", "infy", "hdfc",
            "stock", "price", "quote", "ticker",
            # FX
            "usd", "inr", "jpy", "eur", "gbp", "fx", "forex", "exchange rate",
            "carry trade", "convert",
            # Options
            "straddle", "strangle", "call", "put", "strike", "premium",
            "implied volatility", "iv", "dte", "options",
            # News/sentiment
            "news", "sentiment", "finbert", "bullish", "bearish",
            # Risk agent
            "risk agent", "desk limit", "consult",
            # Bond/macro
            "g-sec", "gsec", "treasury", "yield", "bond", "rbi", "fed",
        }
        has_live_data_context = any(k in lowered for k in live_data_guard)

        if any(k in lowered for k in calculation_keywords) and not has_live_data_context:
            return _log_heuristic(self._default_routing_decision(
                "calculation",
                "Explicit portfolio math keyword matched with no live data context.",
                sub_intent="portfolio_calculation",
                confidence=0.94,
            ), "calculation_keyword")

        if any(k in lowered for k in technical_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Technical indicator keyword matched heuristic.",
                sub_intent="technical_indicator",
                confidence=0.98,
            ), "technical_keyword")

        if any(k in lowered for k in risk_keywords):
            return _log_heuristic(self._default_routing_decision(
                "risk",
                "Risk metric keyword matched heuristic.",
                confidence=0.97,
            ), "risk_keyword")

        # Crypto direct routing — zero LLM cost
        if any(k in lowered for k in crypto_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Crypto asset keyword matched heuristic.",
                sub_intent="market_analysis",
                confidence=0.96,
            ), "crypto_keyword")

        # Macro/Yields direct routing — zero LLM cost
        if any(k in lowered for k in macro_keywords):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Macro/Yields keyword matched heuristic.",
                sub_intent="market_analysis",
                confidence=0.95,
            ), "macro_keyword")

        if self._is_direct_history_query(lowered):
            return _log_heuristic(self._normalize_history_route(self._default_routing_decision(
                "context",
                "History lookup keyword matched heuristic.",
                sub_intent="history_lookup",
                is_history_query=True,
                confidence=0.96,
            )), "history_keyword")

        # (profile_recall handled earlier)

        if self._is_conversation_logic_query(lowered):
            return _log_heuristic(self._default_routing_decision(
                "operational",
                "Conversation-state reasoning matched heuristic.",
                sub_intent="conversation_logic",
                confidence=0.95,
            ), "conversation_logic")

        if any(
            phrase in lowered
            for phrase in ["what is happening", "what's happening", "today", "outlook", "trend", "why is", "why did", "current price", "latest price"]
        ):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Live/recent market-analysis phrasing matched heuristic.",
                sub_intent="market_analysis",
                confidence=0.91,
            ), "market_analysis_phrase")

        if any(phrase in lowered for phrase in ["what is", "explain", "how does", "tell me about", "overview of"]):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Educational phrasing matched heuristic for market concept explanation.",
                sub_intent="education",
                confidence=0.9,
            ), "education_phrase")

        if any(k in lowered for k in ["rsi", "macd", "sma", "ema", "bollinger", "moving average", "support", "resistance"]):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "Technical indicator keyword matched heuristic.",
                sub_intent="technical_indicator",
                confidence=0.94,
            ), "technical_keyword")

        if any(k in lowered for k in ["news", "headline", "catalyst", "announcement"]):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "News keyword matched heuristic.",
                sub_intent="news",
                confidence=0.92,
            ), "news_keyword")

        if any(k in lowered for k in ["outlook", "trend", "market", "stock", "price", "quote", "bullish", "bearish"]):
            return _log_heuristic(self._default_routing_decision(
                "market",
                "General market-analysis keyword matched heuristic.",
                sub_intent="market_analysis",
                confidence=0.85,
            ), "market_keyword")

        return None

    @staticmethod
    def _get_last_specialist_agent(history: List[Dict[str, Any]]) -> Optional[str]:
        """
        Recover the last non-greeting specialist from recent history.
        """
        for item in reversed(history):
            agent_name = item.get("agent_name")
            if agent_name in {"market", "risk", "order", "operational", "context"}:
                return "operational" if agent_name == "context" else agent_name
        return None

    @staticmethod
    def _has_pending_follow_up(guidance: Dict[str, Any]) -> bool:
        """
        Detect whether the previous assistant turn left an unanswered follow-up prompt.
        """
        return bool(guidance.get("pending_follow_up"))

    @staticmethod
    def _get_pending_follow_up_owner(guidance: Dict[str, Any]) -> Optional[str]:
        """
        Recover which specialist owns the currently pending follow-up.
        """
        owner = guidance.get("pending_follow_up_owner")
        logger.info(
            "[Coordinator] Pending follow-up owner resolved | owner=%s pending=%s",
            owner,
            guidance.get("pending_follow_up"),
        )
        return owner

    @staticmethod
    def _get_routing_memory(conversation_id: Optional[str]) -> Dict[str, Any]:
        """
        Load conversation memory once for coordinator routing.
        """
        if not conversation_id:
            return {
                "history": [],
                "guidance": {},
                "last_specialist_agent": None,
                "has_pending_follow_up": False,
                "pending_follow_up_owner": None,
            }

        history = runtime_context_service.get_recent_history(conversation_id)
        guidance = runtime_context_service.build_continuity_guidance(history)
        return {
            "history": history,
            "guidance": guidance,
            "last_specialist_agent": CoordinatorAgent._get_last_specialist_agent(history),
            "has_pending_follow_up": CoordinatorAgent._has_pending_follow_up(guidance),
            "pending_follow_up_owner": CoordinatorAgent._get_pending_follow_up_owner(guidance),
        }

    @staticmethod
    def _should_escalate_route_check(text: str, parsed: Dict[str, Any]) -> bool:
        """
        Escalate ambiguous turns to the reasoning model for better continuity.
        """
        word_count = len(text.strip().split())
        confidence = float(parsed.get("confidence") or 0.0)
        return (
            confidence < 0.60
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
        logger.info(
            "[Coordinator] Invoking LLM router | model=%s text_preview=%s",
            model_name,
            text[:120],
        )
        prompt = Prompts.COORDINATOR_ROUTER.format(text=text)
        parsed = await self.generate_json_response(
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
        logger.info(
            "[Coordinator] LLM router result | model=%s intent=%s sub_intent=%s control=%s confidence=%.2f follow_up=%s history=%s standalone_greeting=%s reason=%s",
            model_name,
            parsed.get("intent"),
            parsed.get("sub_intent"),
            parsed.get("control_signal"),
            float(parsed.get("confidence") or 0.0),
            parsed.get("is_follow_up"),
            parsed.get("is_history_query"),
            parsed.get("is_standalone_greeting"),
            parsed.get("reason"),
        )
        return parsed

    async def handle_message(
        self, 
        text: str, 
        conversation_id: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        """
        Entry point for all user messages. Determines the intent and delegates work.
        """
        # Reset performance metrics for the new request
        from app.services.aidaan.performance import clear_metrics, format_metrics_table
        clear_metrics()

        conv_id = conversation_id or f"conv-{uuid4().hex[:8]}"
        logger.info(f"[Coordinator] Handling session {conv_id}: {text[:50]}...")
        start_time = time.monotonic()
        logger.info(f"[TIMING] Request received | conv_id={conv_id}")

        # Ultra-fast path: "remember/store ... reply only X"
        lowered_text = text.strip().lower()
        reply_only_match = re.search(
            r"\breply only\s+([A-Za-z0-9_-]{1,32})\.?\s*$",
            text.strip(),
            flags=re.IGNORECASE,
        )
        if reply_only_match and any(
            lowered_text.startswith(prefix)
            for prefix in ("remember:", "remember ", "store this:", "store this ", "save this:", "save this ")
        ):
            token = reply_only_match.group(1)
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "[Coordinator] Reply-only fast path engaged | conv_id=%s token=%s",
                conv_id,
                token,
            )
            return self.build_message_response(
                reply=token,
                bullets=[],
                conversation_id=conv_id,
                model_info=self.get_model_info(model_override="reply-only-fastpath"),
                latency_ms=latency_ms,
            )

        # 0. Early domain validation - reject out-of-domain queries immediately
        if self._is_out_of_domain(lowered_text):
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "[Coordinator] Out-of-domain query rejected early | conv_id=%s latency_ms=%.1f text=%s",
                conv_id,
                latency_ms,
                text[:100],
            )
            return self.build_message_response(
                reply="Yeh prashn humare domain se bahar hai. Hum financial markets, jaise ki stocks, bonds, FX, aur commodities se related jaankari aur analysis provide karte hain. Humare paas mythological, personal, ya general knowledge ke questions ka answer dene ke liye tools ya data nahi hai.",
                bullets=[],
                conversation_id=conv_id,
                model_info=self.get_model_info(model_override="domain-guard"),
                latency_ms=latency_ms,
            )

        # 1. Pre-processing Guardrails
        context = context or {}
        username = context.get("username")
        
        guardrail_result = guardrail_orchestrator.pre_process_request(
            text,
            username=username,
            conversation_id=conv_id,
            context=context
        )
        
        # Update context with guardrail flags
        context.update(guardrail_result.get("context_updates", {}))
        
        # If request is blocked by guardrails, return early
        if not guardrail_result["allowed"]:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "[Coordinator] Request blocked by guardrails | conv_id=%s reason=%s",
                conv_id,
                guardrail_result["blocked_reason"]
            )
            return self.build_message_response(
                reply=guardrail_result["blocked_reason"],
                bullets=[
                    "Request blocked by institutional guardrails",
                    "No action was taken"
                ],
                conversation_id=conv_id,
                model_info=self.get_model_info(model_override="guardrail-block"),
                latency_ms=latency_ms,
                guardrails=[{
                    "type": "kill_switch_block",
                    "reason": guardrail_result["blocked_reason"]
                }]
            )
        
        # Log advisory detection if present
        if guardrail_result["advisory_detected"]:
            logger.info(
                "[Coordinator] Advisory intent detected | conv_id=%s confidence=%.2f",
                conv_id,
                context.get("advisory_confidence", 0.0)
            )

        # 2. Routing phase (legacy kill switch check kept for backward compatibility)
        if kill_switch_service.is_active() and kill_switch_service.request_has_trade_intent(text):
            status = kill_switch_service.get_status()
            logger.warning(
                "[Coordinator] Kill switch blocked trade-intent request before routing | conversation_id=%s reason=%s",
                conv_id,
                status.get("reason"),
            )
            return self.build_message_response(
                reply="Venue kill switch is active. I can acknowledge the request, but I will not stage, draft, or advance any trade-affecting workflow until the freeze is lifted.",
                bullets=[
                    "Trade-affecting request detected",
                    "No RFQ draft or staging action was created",
                    f"Kill switch reason: {status.get('reason')}",
                ],
                conversation_id=conv_id,
                model_info=self.get_model_info(model_override="kill-switch-guard"),
                latency_ms=(time.monotonic() - start_time) * 1000,
            )
        routing = await self._route_intent(
            text,
            conversation_id=conv_id,
            username=context.get("username"),
        )
        routing_elapsed = time.monotonic() - start_time
        logger.info(f"[TIMING] Routing completed | elapsed={routing_elapsed:.3f}s | agent={routing.get('intent')}")
        
        agent_id = routing.get("intent") or "greeting"
        context["routing"] = routing
        logger.info(
            "[Coordinator] Final route | conversation_id=%s agent=%s sub_intent=%s control=%s confidence=%.2f reason=%s",
            conv_id,
            agent_id,
            routing.get("sub_intent"),
            routing.get("control_signal"),
            float(routing.get("confidence") or 0.0),
            routing.get("reason"),
        )

        target_agent = registry.get_agent(agent_id or "greeting")
        if target_agent:
            agent_start = time.monotonic()
            logger.info(f"[TIMING] Delegating to {agent_id} agent | conv_id={conv_id}")
            
            response = await target_agent.handle_message(
                text=text,
                conversation_id=conv_id,
                context=context,
                tool_callback=tool_callback,
            )

            # ------------------------------------------------------------------
            # Mandatory IDBX Data Provenance footer (end-user friendly)
            # ------------------------------------------------------------------
            if response and isinstance(getattr(response, "reply", None), str):
                has_provenance = "**IDBX Data Provenance:**" in response.reply
                if not has_provenance:
                    # Only append when we can state a concrete source (avoid misleading footers).
                    if agent_id == "market":
                        response.reply = response.reply.rstrip() + "\n\n**IDBX Data Provenance:** Alpha Vantage Live Feed"
            
            agent_elapsed = time.monotonic() - agent_start
            logger.info(f"[TIMING] Agent {agent_id} completed | elapsed={agent_elapsed:.3f}s")
            
            # Post-processing Guardrails: Reframe advisory responses
            if context.get("advisory_detected"):
                original_reply = response.reply
                reframed_reply = await guardrail_orchestrator.post_process_response(
                    original_reply,
                    context=context,
                    conversation_id=conv_id,
                    original_query=text
                )
                
                if reframed_reply != original_reply:
                    response.reply = reframed_reply
                    logger.info(
                        "[Coordinator] Response reframed for advisory compliance | conv_id=%s",
                        conv_id
                    )
                    
                    # Add guardrail metadata to response
                    if not hasattr(response, 'guardrails') or response.guardrails is None:
                        response.guardrails = []
                    response.guardrails.append({
                        "type": "advisory_reframed",
                        "confidence": context.get("advisory_confidence", 0.0)
                    })
            
            # Inject latency into the final response
            response.latency_ms = (time.monotonic() - start_time) * 1000
            
            # Final Transaction Log for speed and accuracy tracking
            perf_logger.info("*************")
            perf_logger.info(f"TRANSACTION_LOG | conv_id={conv_id}")
            perf_logger.info(f"QUESTION: {text}")
            perf_logger.info(f"ANSWER: {response.reply}")
            perf_logger.info(f"TOTAL_LATENCY: {response.latency_ms/1000:.3f}s")
            
            # Consolidated Performance Table
            from app.services.aidaan.performance import format_metrics_table
            performance_table = format_metrics_table()
            perf_logger.info("\n" + performance_table)
            perf_logger.info("*************")
            
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
    ) -> Dict[str, Any]:
        """
        Classification logic for routing requests.
        Priority:
        1. Heuristics (Regex/Keyword) - zero token cost, instant.
        2. LLM Classification - handles natural language ambiguity.

        Returns:
            The ID of the target agent or None.
        """
        lowered = text.strip().lower()
        routing_memory = self._get_routing_memory(conversation_id)
        last_specialist_agent = routing_memory["last_specialist_agent"]
        has_pending_follow_up = routing_memory["has_pending_follow_up"]
        pending_follow_up_owner = routing_memory["pending_follow_up_owner"]
        continuity_target = pending_follow_up_owner or last_specialist_agent

        logger.info(
            "[Coordinator] Static heuristic routing disabled; using LLM-first routing | last_specialist=%s pending_owner=%s pending_follow_up=%s text_preview=%s",
            last_specialist_agent,
            pending_follow_up_owner,
            has_pending_follow_up,
            text[:120],
        )

        # --- Secondary Path: Fast LLM intent routing ---
        # Use the dedicated router model first, then escalate ambiguous continuity checks.
        # Phase 4 Optimization: Skip Pro escalation for high-confidence results
        try:
            parsed = await self._classify_route(
                text=text,
                conversation_id=conversation_id,
                username=username,
                model_name=settings.VERTEX_AI_ROUTER_MODEL_NAME,
            )

            # Phase 4: Check if we can skip Pro escalation
            confidence = float(parsed.get("confidence") or 0.0)
            is_follow_up = bool(parsed.get("is_follow_up"))
            
            # Skip Pro if high confidence and not a follow-up
            if confidence > 0.85 and not is_follow_up:
                logger.info(
                    "[Coordinator] High-confidence Flash result, skipping Pro escalation | confidence=%.2f intent=%s",
                    confidence,
                    parsed.get("intent")
                )
                # Continue with Flash result
            elif self._should_escalate_route_check(text, parsed):
                # Only escalate for ambiguous cases
                logger.info(
                    "[Coordinator] Escalating to Pro model | confidence=%.2f is_follow_up=%s",
                    confidence,
                    is_follow_up
                )
                parsed = await self._classify_route(
                    text=text,
                    conversation_id=conversation_id,
                    username=username,
                    model_name=settings.VERTEX_AI_REASONING_MODEL_NAME,
                )
            else:
                logger.info(
                    "[Coordinator] Using Flash result without escalation | confidence=%.2f",
                    confidence
                )

            intent = parsed.get("intent") or "greeting"
            parsed["sub_intent"] = parsed.get("sub_intent") or "general"
            parsed["control_signal"] = parsed.get("control_signal") or "none"
            is_follow_up = bool(parsed.get("is_follow_up"))
            is_history_query = bool(parsed.get("is_history_query"))
            is_standalone_greeting = bool(parsed.get("is_standalone_greeting"))

            parsed = self._normalize_history_route(parsed)
            if parsed.get("intent") == "operational":
                return parsed

            if (
                is_follow_up
                and continuity_target
                and parsed.get("intent") in {"greeting", "context"}
                and not self._is_explicit_fresh_query(lowered)
            ):
                parsed["intent"] = continuity_target
                logger.info(
                    "[Coordinator] Limited continuity override applied for ambiguous follow-up | agent=%s sub_intent=%s control=%s",
                    parsed.get("intent"),
                    parsed.get("sub_intent"),
                    parsed.get("control_signal"),
                )
                return parsed

            if intent == "greeting" and not is_standalone_greeting and continuity_target:
                parsed["intent"] = continuity_target
                logger.info(
                    "[Coordinator] Non-standalone greeting rerouted to previous specialist | agent=%s",
                    parsed.get("intent"),
                )
                return parsed

            return parsed
        except Exception as e:
            logger.warning(f"[Coordinator] LLM routing failed: {e}")
            # --- Ultimate fallback ---
            if last_specialist_agent and len(lowered.split()) <= 4:
                return self._default_routing_decision(
                    last_specialist_agent,
                    "Fallback continuity route to previous specialist.",
                    sub_intent="control",
                    control_signal="continue",
                    is_follow_up=True,
                    confidence=0.7,
                )
            if any(k in lowered for k in ["risk", "compliance"]):
                return self._default_routing_decision("risk", "Fallback keyword route to risk.", confidence=0.7)
            if any(k in lowered for k in ["price", "market", "stock", "news", "quote"]):
                return self._default_routing_decision(
                    "market",
                    "Fallback keyword route to market.",
                    sub_intent="market_analysis",
                    confidence=0.7,
                )
            if any(k in lowered for k in ["order", "execution"]):
                return self._default_routing_decision("order", "Fallback keyword route to order.", confidence=0.7)
            return self._default_routing_decision("greeting", "Fallback default route.", confidence=0.5)

    def get_capabilities(self) -> List[str]:
        """
        Returns the core capabilities of the Coordinator.
        """
        return ["intent_routing", "agent_orchestration", "multi_agent_showcase"]


# Singleton instance
coordinator_agent = CoordinatorAgent()
