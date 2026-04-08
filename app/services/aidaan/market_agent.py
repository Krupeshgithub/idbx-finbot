"""
Market Data Agent for AIDAAN.
Full pipeline: Gemini intent → Alpha Vantage live data → Gemini formatted response.
Handles any natural language market question — no ticker required from user.
"""
import logging
from typing import Any, Dict, List, Optional

from app.services.aidaan.base import BaseAgent
from app.services.market.alphavantage_client import alphavantage_client
from app.services.market.gemini_client import gemini_client
from app.schemas.aidaan import ActionItem, AidaanMessageResponse

logger = logging.getLogger(__name__)


class MarketAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="market_agent",
            model_name="gemini-1.0-pro"
        )

    async def handle_message(
        self,
        text: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Getting message and handle by the market response.
        """
        intent = await gemini_client.extract_market_intent(text)
        data_type = intent.get("type", "unknown")
        symbols = intent.get("symbols", [])
        interval = intent.get("interval", "5min")
        ctx_note = intent.get("context", "")

        logger.info(f"[MarketAgent] type={data_type}, symbols={symbols}, text='{text}'")

        if data_type == "unknown" or not symbols:
            return AidaanMessageResponse(
                reply=(
                    "I couldn't identify a specific stock or market data request. "
                    "Try asking like: 'What is Apple's current price?' or "
                    "'Show me Nvidia's performance this week.'"
                ),
                bullets=[
                    "Supported: current quotes, daily OHLCV, intraday OHLCV",
                    "You can use company names or ticker symbols",
                    "Examples: Apple, HDFC Bank, Nvidia, Tesla, GOOGL"
                ],
                actions=[],
                conversation_id=conversation_id,
                model=self.get_model_info()
            )
        
        # Fetch live data from Alpha Vantage
        market_data = {}

        if data_type == "quote":
            market_data = await alphavantage_client.get_bulk_quotes(symbols)

        elif data_type == "daily_ohlcv":
            # Fetch daily for first symbol; if multiple, fetch all in parallel
            if len(symbols) == 1:
                market_data = await alphavantage_client.get_daily_ohlcv(symbols[0])
            else:
                # Multiple symbols: get quotes + note daily not bulk-supported
                market_data = await alphavantage_client.get_bulk_quotes(symbols)
                data_type = "quote"
        
        elif data_type == "intraday_ohlcv":
            market_data = await alphavantage_client.get_intraday_ohlcv(
                symbols[0],
                interval
            )

        # Gemini formats a natural language response
        formatted = await gemini_client.format_market_response(
            user_text=text,
            market_data=market_data,
            data_type=data_type,
            context=ctx_note
        )

        # Build action buttons
        actions = [
            ActionItem(
                kind="refresh_quote",
                title=f"Refresh {', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''}",
                payload={"symbols": symbols, "type": data_type},
                requires_human_confirm=False
            )
        ]

        return AidaanMessageResponse(
            reply=formatted["reply"],
            bullets=formatted["bullets"],
            actions=actions,
            conversation_id=conversation_id,
            model=self.get_model_info()
        )
    
    def get_capabilities(self) -> List[str]:
        return [
            "realtime_equity_quotes",
            "daily_ohlcv",
            "intraday_ohlcv",
            "company_name_to_ticker_resolution",
            "natural_language_market_questions"
        ]

market_agent = MarketAgent()
