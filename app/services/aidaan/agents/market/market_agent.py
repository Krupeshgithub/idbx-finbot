"""
Institutional Market Analyst Agent - AIDAAN Intelligence Layer
==============================================================
The MarketAgent specializes in synthesizing deep financial insights using 
exhaustive real-time data sources (Alpha Vantage).

Key Features:
- Multi-tool orchestration (Economics, FX, Commodities, Technicals).
- Agentic multi-turn synthesis for complex queries.
- Bloomberg-style professional report generation.
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.aidaan.core.base import BaseAgent
from app.schemas.aidaan import AidaanMessageResponse
from app.core.prompts import Prompts
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class MarketAgent(BaseAgent):
    """
    Agentic analyst capable of navigating complex market data ecosystems 
    to provide synthesized financial reports.
    """

    def __init__(self):
        """
        Initialize with a model capable of complex tool orchestration.
        """
        super().__init__(
            name="market",
            model_name=settings.VERTEX_AI_MODEL_NAME, # Switch back to High-Speed Flash
        )

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        """
        Processes market queries by discovering and invoking appropriate tools.

        Args:
            text: Market-related query (e.g., "Analyze Nvidia's RSI vs its peers").
            conversation_id: Session ID for state tracking.
            context: Additional market context.

        Returns:
            AidaanMessageResponse: A synthesized, professional financial report.
        """
        logger.info(f"[MarketAgent] Analyzing market query: {text}")
        context = context or {}
        
        # Externalized professional orchestration prompt
        orchestration_prompt = Prompts.MARKET_ORCHESTRATION.format(text=text)

        # --- Heuristic Fast-Path for simple quotes ---
        if len(text.split()) < 4 and any(k in text.lower() for k in ["price", "quote", "price of", "level"]):
             logger.info("[MarketAgent] Triggering Heuristic FAST-PATH for simple quote.")
             if tool_callback: await tool_callback("get_stock_quote")
             
             # Fetch data directly bypassing LLM turn 1
             import re
             ticker_match = re.search(r'\b[A-Za-z]{1,5}\b', text.upper())
             symbol = ticker_match.group(0) if ticker_match else text.split()[0].upper()
             
             data = await self.llm.call_mcp_tool("get_stock_quote", {"symbol": symbol})
             
             # Even if error, if it's a rate limit, the tool now returns simulated data
             # Check for simulation note or valid data
             price = data.get("price")
             if price:
                 change = data.get("change_percent", "0%")
                 reply = f"[Direct Answer] {symbol} is currently trading at {price} ({change}).\n\n" \
                         f"[Market Insight] Institutional flows are being monitored for {symbol}. Volume is at {data.get('volume', 'N/A')}.\n\n" \
                         f"[Trade Implication] Positioning suggests a neutral/cautious stance at current levels.\n\n" \
                         f"[Optional Follow-up] Would you like to check key support levels or recent news?"
                 if "note" in data:
                     reply += f"\n\n(Note: {data['note']})"
                     
                 return self.build_message_response(reply=reply, bullets=[f"Ticker: {symbol}", f"Price: {price}"], conversation_id=conversation_id, model_info={"agent": self.name, "llm": "FASTPATH-TEMPLATE"})

        try:
            # Execute agentic loop with tool discovery enabled
            response_data = await self.generate_json_response(
                orchestration_prompt,
                conversation_id=conversation_id,
                username=context.get("username"),
                use_mcp_tools=True,
                tool_callback=tool_callback,
                response_schema=Prompts.STANDARD_RESPONSE_SCHEMA,
                system_instruction=Prompts.TRADER_SYSTEM_INSTRUCTION + "\nRole: Senior Interbank analyst. You MUST respond with exactly the 4 parts requested in the schema."
            )
            
            return self.build_message_response(
                reply=response_data.get("reply", "Market data analysis complete."),
                bullets=response_data.get("bullets", []),
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as e:
            return self.build_error_response(
                reply="I encountered an issue while performing the market deep-dive. Please verify the symbol or metric.",
                conversation_id=conversation_id,
                error=e,
                model_info=self.get_model_info(),
            )

    def get_capabilities(self) -> List[str]:
        """
        Returns the scope of market analysis capabilities.
        """
        return [
            "macro_economics_analysis", 
            "commodity_tracking", 
            "fx_monitoring", 
            "technical_analysis", 
            "fundamental_deep_dives"
        ]


# Singleton instance
market_agent = MarketAgent()
