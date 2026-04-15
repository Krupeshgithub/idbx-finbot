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
        super().__init__(name="market", model_name="gemini-1.5-pro")

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None
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
        
        # Externalized professional orchestration prompt
        orchestration_prompt = Prompts.MARKET_ORCHESTRATION.format(text=text)

        try:
            # Execute agentic loop with tool discovery enabled
            response_data = await self.llm.generate_json(
                prompt=orchestration_prompt,
                use_mcp_tools=True
            )
            
            return AidaanMessageResponse(
                reply=response_data.get("reply", "Market data analysis complete."),
                bullets=response_data.get("bullets", []),
                actions=[], 
                conversation_id=conversation_id,
                model=self.get_model_info()
            )
        except Exception as e:
            internal_error = str(e)
            logger.error(f"[MarketAgent] Analysis failed: {internal_error}")
            return AidaanMessageResponse(
                reply="I encountered an issue while performing the market deep-dive. Please verify the symbol or metric.",
                bullets=[internal_error],
                actions=[],
                conversation_id=conversation_id,
                model=self.get_model_info()
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
