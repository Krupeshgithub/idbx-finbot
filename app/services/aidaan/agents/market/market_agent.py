"""
Institutional market agent for Vertex-first market orchestration.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.core.config.settings import settings
from app.core.prompts import Prompts
from app.schemas.aidaan import AidaanMessageResponse
from app.services.aidaan.core.base import BaseAgent

logger = logging.getLogger(__name__)


class MarketAgent(BaseAgent):
    """
    Vertex-first market agent.

    This agent intentionally avoids manual ticker aliases and heuristic
    fast-paths so symbol resolution, tool choice, and response synthesis
    remain model-driven.
    """

    def __init__(self) -> None:
        """
        Initialize the market agent with the default market model.
        """
        super().__init__(name="market", model_name=settings.VERTEX_AI_MODEL_NAME)

    @staticmethod
    def _normalize_market_text(text: str) -> str:
        """
        Normalize user text for lightweight execution guards.
        """
        return re.sub(r"\s+", " ", text.strip().lower())

    def _resolve_execution_plan(self, *, sub_intent: str, control_signal: str, text: str) -> Dict[str, Any]:
        """
        Enhanced execution plan resolver with explicit sentiment detection.
        """
        normalized_text = self._normalize_market_text(text)
        
        # Detect freshness requirements
        freshness_requested = bool(
            re.search(r"\b(today|now|latest|current|live|realtime|real-time)\b", normalized_text)
        )
        freshness_negated = bool(
            re.search(
                r"\b(do not|don't|without|no need for|not asking for)\b.{0,40}\b(live|current|real[- ]?time)\b",
                normalized_text,
            )
            or re.search(
                r"\b(live|current|real[- ]?time)\b.{0,40}\b(data)\b.{0,20}\b(do not|don't|without|not needed|not required)\b",
                normalized_text,
            )
        )
        requires_freshness = freshness_requested and not freshness_negated

        # NEW: Explicit sentiment detection (HIGHEST PRIORITY)
        sentiment_keywords = [
            r"\bsentiment\b",
            r"\bfinbert\b",
            r"\bbullish\b",
            r"\bbearish\b",
            r"\bnews\b",
            r"\bheadlines?\b",
            r"\bmarket mood\b",
            r"\boptimistic\b",
            r"\bpessimistic\b",
            r"\boutlook\b",
            r"\bfeeling\b.*\bmarket\b",
            r"\bmarket\b.*\bfeeling\b",
        ]
        
        sentiment_requested = any(
            re.search(pattern, normalized_text) for pattern in sentiment_keywords
        )
        
        # If sentiment is explicitly requested, ALWAYS enable tools
        if sentiment_requested:
            logger.info(
                "[MarketAgent] 🎯 SENTIMENT DETECTED - forcing tool mode | text_preview=%s",
                text[:100]
            )
            return {
                "tool_mode": "full",
                "model_name": settings.VERTEX_AI_MODEL_NAME,
                "fast_path_kind": None,
                "reason": "Sentiment analysis requires FinBERT + news tools",
            }

        # Default plan
        plan = {
            "tool_mode": "full",
            "model_name": settings.VERTEX_AI_MODEL_NAME,
            "fast_path_kind": None,
        }

        # Heavy reasoning detection
        HEAVY_REASONING_PATTERNS = [
            r"\bdcf\b", r"\bwacc\b", r"\bblack.?swan\b", r"\bvar\b.*\bscenario\b",
            r"\bcorrelation.?matrix\b", r"\bdelta.?neutral\b", r"\bsensitivity.?matrix\b",
            r"\bstress.?test\b", r"\barbitrage.?simulation\b",
        ]
        needs_heavy_reasoning = any(
            re.search(p, normalized_text) for p in HEAVY_REASONING_PATTERNS
        )

        if needs_heavy_reasoning:
            plan["model_name"] = settings.VERTEX_AI_REASONING_MODEL_NAME

        # Control signal handling
        if control_signal == "stop":
            plan.update({
                "tool_mode": "none",
                "model_name": settings.VERTEX_AI_MODEL_NAME,
                "fast_path_kind": "stop_acknowledgement",
            })
        elif control_signal == "clarify":
            plan.update({
                "tool_mode": "none",
                "model_name": settings.VERTEX_AI_MODEL_NAME,
                "fast_path_kind": "clarify_response",
            })
        elif sub_intent == "education" and not requires_freshness and not sentiment_requested:
            # Educational queries without freshness or sentiment can use model-only path
            # BUT: Keep tools available for accuracy (model can choose to use them)
            logger.info(
                "[MarketAgent] 🔧 education mode WITH tools enabled for accuracy"
            )
            # Keep tool_mode as "full" - do NOT set to "none"

        logger.info(
            "[MarketAgent] Execution plan resolved | sub_intent=%s control=%s tool_mode=%s model=%s fast_path=%s sentiment_requested=%s freshness_requested=%s freshness_negated=%s requires_freshness=%s",
            sub_intent,
            control_signal,
            plan["tool_mode"],
            plan["model_name"],
            plan["fast_path_kind"],
            sentiment_requested,
            freshness_requested,
            freshness_negated,
            requires_freshness,
        )
        return plan

    def _build_stop_response(self, *, conversation_id: str, model_name: str) -> AidaanMessageResponse:
        """
        Return an immediate acknowledgement for explicit stop signals.
        """
        logger.info("[MarketAgent] Fast path engaged | kind=stop_acknowledgement")
        return self.build_message_response(
            reply="[Direct Answer]\nAcknowledged. I will stop the current market analysis.",
            bullets=["Current analysis halted", "No additional tools were used"],
            conversation_id=conversation_id,
            model_info=self.get_model_info(model_override=model_name),
        )

    @staticmethod
    def _build_market_system_instruction(sub_intent: str, control_signal: str) -> str:
        """
        Enhanced system instruction with explicit sentiment handling.
        """
        base = (
            Prompts.TRADER_SYSTEM_INSTRUCTION
            + "\nRole: Senior Interbank analyst."
            + " Resolve the correct ticker, market, index, company, or concept from the user's wording before using tools."
            + " \nCRITICAL RULES FOR DATA/A2A:\n"
            + " 1. TICKER RESOLUTION: Common company names map to tickers as follows:\n"
            + "    - Google → GOOGL, Apple → AAPL, Microsoft → MSFT, Amazon → AMZN, Tesla → TSLA\n"
            + "    - Meta/Facebook → META, Nvidia → NVDA, Intel → INTC, AMD → AMD, IBM → IBM\n"
            + "    - Exxon/Exxon Mobil/Exxon Mobile → XOM, Chevron → CVX, Shell → SHEL\n"
            + "    - Netflix → NFLX, Uber → UBER, Airbnb → ABNB, PayPal → PYPL\n"
            + "    - If user says 'google price', immediately use ticker GOOGL. Do NOT call search_ticker for common names.\n"
            + " 2. EXACT NUMBERS: If the user asks for exact stock prices/dates, MUST use Alpha Vantage exact data.\n"
            + " 3. TICKER VALIDATION: If search_ticker returns no matches or an error, you MUST try common ticker aliases first (e.g. 'Exxon Mobile' / 'Exxon Mobil' → XOM, 'Alphabet' → GOOGL, 'Meta' → META). If no alias resolves it, inform the user the ticker was not found. NEVER fabricate data for an unresolved ticker.\n"
            + " 4. AGENT-TO-AGENT (A2A) COLLABORATION: Before providing trading advice, order staging, or confirming a position size, you MUST internally consult the 'risk' agent using your `consult_specialist_agent` tool. This ensures desk limits are securely verified within the Privacy Vault.\n"
            + " 5. If the current user turn is brief or ambiguous, use recent conversation memory and any pending follow-up prompt to infer the intended continuation.\n"
            + " 6. When the previous assistant turn offered optional next-step analysis and the user appears to accept it, continue that analysis instead of discussing the ambiguity.\n"
            + " 7. ANTI-HALLUCINATION: NEVER fabricate, invent, or estimate specific financial numbers (prices, yields, rates, volumes, EPS, P/E ratios) without tool data. If tools are not available and the user asks for specific numbers, explicitly state: 'I do not have live data for this — please use the fetch tools or provide the data.' Do NOT make up plausible-looking tables or figures.\n"
            + " 8. **SENTIMENT REQUESTS (CRITICAL)**: If the user asks for 'sentiment', 'FinBERT', 'news sentiment', 'bullish/bearish', 'latest news', 'headlines', or 'market mood' for ANY instrument:\n"
            + "    - You MUST call `get_market_news` with the appropriate ticker (e.g., tickers='MSFT' for Microsoft)\n"
            + "    - Do NOT skip this step or claim you cannot do it\n"
            + "    - The tool automatically runs FinBERT sentiment analysis on the headlines\n"
            + "    - Even if the query also asks for other data (price, volume, etc.), you MUST still call get_market_news\n"
            + "    - Example: 'Compare sentiment of MSFT to its volume' → Call BOTH get_market_news AND get_daily_series\n"
            + " 9. RISK AGENT: If the user says 'consult risk agent', 'check desk limit', or 'does this breach' — you MUST call `consult_specialist_agent` with target_agent='risk'. Do NOT skip this step.\n"
            + " 10. **TOP STOCKS / RANKINGS (NEW - CRITICAL)**: If the user asks for 'top stocks', 'largest companies', 'biggest stocks', 'top 10 stocks', etc.:\n"
            + "    - DEFAULT INTERPRETATION: 'Top stocks' = largest by market capitalization (industry standard)\n"
            + "    - IMMEDIATELY call `get_top_stocks_by_market_cap(limit=N)` where N is the requested number (default 10)\n"
            + "    - Do NOT ask for clarification unless user explicitly requests a different metric\n"
            + "    - Format response as Markdown table with: Rank, Company, Ticker, Market Cap, Price, Sector\n"
            + "    - For 'top gainers' → call `get_top_gainers(limit=N)`\n"
            + "    - For 'top losers' → call `get_top_losers(limit=N)`\n"
            + "    - For 'most active' → call `get_top_stocks_by_volume(limit=N)`\n"
            + "    - This is NOT hallucination: we use known mega-cap list + live API data\n"
        )

        if sub_intent == "education":
            return (
                base
                + " The user is asking for explanation or overview."
                + " You MUST provide accurate, factual information."
                + " If you need specific data (prices, dates, metrics), use the available tools."
                + " NEVER say 'I am unable to' or 'I cannot' — you have tools available."
                + " Structure your response with clear sections: [Direct Answer], [How It Works], [Why It Matters], [Optional Follow-up]."
                + " Avoid forced trade implication language for pure educational questions."
            )

        if sub_intent == "technical_indicator":
            return (
                base
                + " Focus on the requested indicators and keep the response compact."
                + " Include only the most relevant price levels and avoid unrelated fundamental exposition."
            )

        if sub_intent == "news":
            return (
                base
                + " If news is retrieved but `analytics_status` is 'ERROR_FALLBACK', admit the sentiment model is down and do not guess a Bullish/Bearish trend."
                + " If entity-specific news cannot be retrieved at all, say that cleanly and stop there."
                + " Do not pad missing-news cases with speculative market narratives or recycled technical levels."
                + " Only discuss implications when they are directly supported by retrieved news."
            )

        if control_signal == "stop":
            return (
                base
                + " The user appears to be declining further analysis."
                + " Reply briefly, acknowledge the stop, and offer a lightweight alternative only if helpful."
            )

        if control_signal == "clarify":
            return (
                base
                + " The user appears confused or wants simplification."
                + " Simplify the prior answer or ask one short clarifying question."
            )

        return base

    async def handle_message(
        self,
        text: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
        enable_streaming: bool = False,
    ) -> AidaanMessageResponse:
        """
        Run a full Vertex-driven market workflow for the user query.

        The model is responsible for:
        - identifying the correct instrument or company
        - choosing the relevant MCP tools
        - synthesizing the final trader-facing answer

        Args:
            enable_streaming: If True, returns a tuple (response, streaming_chunks)
                             where streaming_chunks is an async generator that yields
                             real-time token-by-token output from Vertex AI.
                             If False, returns AidaanMessageResponse with pre-built reply.
        """
        logger.info("[MarketAgent] Analyzing market query via Vertex-first orchestration: %s", text)
        market_start = time.monotonic()
        context = context or {}
        
        # TEST LOGGING
        query_word_count = len(text.strip().split())
        logger.info(f"[TEST] Query word count: {query_word_count} | text_preview={text[:60]}")
        
        routing = context.get("routing") or {}
        sub_intent = routing.get("sub_intent", "general")
        control_signal = routing.get("control_signal", "none")
        logger.info(
            "[MarketAgent] Routing context received | sub_intent=%s control=%s confidence=%s reason=%s",
            sub_intent,
            control_signal,
            routing.get("confidence"),
            routing.get("reason"),
        )
        execution_plan = self._resolve_execution_plan(
            sub_intent=sub_intent,
            control_signal=control_signal,
            text=text,
        )
        tool_mode = execution_plan["tool_mode"]
        model_name = execution_plan["model_name"]
        fast_path_kind = execution_plan["fast_path_kind"]

        if fast_path_kind == "stop_acknowledgement":
            return self._build_stop_response(conversation_id=conversation_id, model_name=model_name)

        orchestration_prompt = Prompts.MARKET_ORCHESTRATION.format(text=text)
        logger.info(
            "[MarketAgent] Dispatching market generation | model=%s tool_mode=%s fast_path=%s streaming=%s",
            model_name,
            tool_mode,
            fast_path_kind,
            enable_streaming,
        )
        
        llm_start = time.monotonic()
        logger.info(f"[TIMING] Starting LLM orchestration | model={model_name} | tool_mode={tool_mode}")

        try:
            # If streaming is enabled AND tools are needed, use the hybrid approach:
            # 1. Call tools synchronously (blocking) to gather complete data
            # 2. Stream the synthesis phase token-by-token
            if enable_streaming and tool_mode == "full":
                logger.info(f"[MarketAgent] 🔥 STREAMING PATH ACTIVATED | enable_streaming={enable_streaming} | tool_mode={tool_mode}")
                return await self._handle_message_streaming(
                    text=text,
                    orchestration_prompt=orchestration_prompt,
                    conversation_id=conversation_id,
                    model_name=model_name,
                    sub_intent=sub_intent,
                    control_signal=control_signal,
                    context=context,
                    tool_callback=tool_callback,
                )

            # Non-streaming path (original behavior)
            logger.info(f"[MarketAgent] 📝 NON-STREAMING PATH | enable_streaming={enable_streaming} | tool_mode={tool_mode}")
            response_data = await self.generate_json_response(
                orchestration_prompt,
                conversation_id=conversation_id,
                model_override=model_name,
                username=context.get("username"),
                use_mcp_tools=(tool_mode == "full"),
                tool_callback=tool_callback,
                response_schema=Prompts.STANDARD_RESPONSE_SCHEMA,
                system_instruction=self._build_market_system_instruction(sub_intent, control_signal),
            )
            
            llm_elapsed = time.monotonic() - llm_start
            logger.info(f"[TIMING] LLM orchestration completed | elapsed={llm_elapsed:.3f}s")
            
            logger.info(
                "[MarketAgent] Response generated | format=%s bullets=%s reply_preview=%s",
                response_data.get("format"),
                len(response_data.get("bullets", [])),
                str(response_data.get("reply", ""))[:160],
            )
            return self.build_message_response(
                reply=response_data.get("reply", "Market data analysis complete."),
                bullets=response_data.get("bullets", []),
                conversation_id=conversation_id,
                model_info=self.get_model_info(model_override=model_name),
            )
        except Exception as exc:
            logger.warning(
                "[MarketAgent] Market orchestration failed | sub_intent=%s control=%s error=%s",
                sub_intent,
                control_signal,
                exc,
            )
            return self.build_error_response(
                reply="I encountered an issue while performing the market analysis. Please verify the instrument or request.",
                conversation_id=conversation_id,
                error=exc,
                model_info=self.get_model_info(model_override=model_name),
            )

    async def _handle_message_streaming(
        self,
        text: str,
        orchestration_prompt: str,
        conversation_id: str,
        model_name: str,
        sub_intent: str,
        control_signal: str,
        context: Dict[str, Any],
        tool_callback: Optional[callable],
    ):
        """
        Streaming-enabled market workflow:
        1. Phase 1 (Blocking): Call MCP tools and gather complete data
        2. Phase 2 (Streaming): Stream the synthesis token-by-token

        Returns a tuple: (AidaanMessageResponse, streaming_chunks_generator)
        The caller must iterate over the generator to get real-time token-by-token output.
        """
        from app.core.llm_client import llm_client
        from app.services.aidaan.streaming_service import streaming_service

        logger.info("[MarketAgent] Streaming mode | Phase 1: Gathering tool data...")

        # Phase 1: Call tools synchronously (blocking) to get complete data
        # This uses the same MCP tool loop as non-streaming, ensuring identical
        # tool selection and data gathering logic.
        tool_results_raw = await llm_client.generate_json(
            prompt=orchestration_prompt,
            model_override=model_name,
            use_mcp_tools=True,
            tool_callback=tool_callback,
            system_instruction=self._build_market_system_instruction(sub_intent, control_signal),
        )

        logger.info("[MarketAgent] Streaming mode | Phase 2: Streaming synthesis...")

        # Phase 2: Stream the synthesis using the gathered tool data
        # Build a synthesis prompt that includes the tool results as context
        tool_context = f"Tool Results:\n{json.dumps(tool_results_raw, ensure_ascii=False, indent=2)[:12000]}"
        synthesis_prompt = (
            f"User Query: {text}\n\n"
            f"{tool_context}\n\n"
            f"=== SYNTHESIS INSTRUCTIONS ===\n"
            f"Using the tool results above, write a complete professional trader response.\n"
            f"Structure (MANDATORY — all 4 parts required):\n"
            f"  [Direct Answer] — State the key fact(s) directly (price, rate, data summary).\n"
            f"  [Market Insight] — Explain what the data means in trading terms (flows, momentum, context).\n"
            f"  [Trade Implication] — What should the trader infer or act on?\n"
            f"  [Optional Follow-up] — Offer one relevant next-step question.\n"
            f"Formatting: Use Markdown tables for any tabular data (OHLCV, historical series, rankings).\n"
            f"Language: Mirror the user's language (Hinglish, English, etc.).\n"
            f"IMPORTANT: Write a FULL response — do not truncate or summarise prematurely.\n"
            f"Do NOT wrap the response in JSON or code blocks — return plain text only."
        )

        # Create a response object (metadata only, no reply text yet)
        response = self.build_message_response(
            reply="",  # Will be populated by streaming
            bullets=[],
            conversation_id=conversation_id,
            model_info=self.get_model_info(model_override=model_name),
        )

        # Create the streaming generator
        streaming_chunks = streaming_service.stream_response_async(
            prompt=synthesis_prompt,
            model_name=model_name,
            system_instruction=self._build_market_system_instruction(sub_intent, control_signal),
            enable_thinking=False,
            conversation_id=conversation_id,
            username=context.get("username"),
        )

        # Return tuple: (response_metadata, streaming_generator)
        return (response, streaming_chunks)

    def get_capabilities(self) -> List[str]:
        """
        Return the market coverage supported by the agent.
        """
        return [
            "market_quotes",
            "historical_price_series",
            "technical_analysis",
            "fundamental_analysis",
            "multi_tool_market_orchestration",
        ]


market_agent = MarketAgent()
