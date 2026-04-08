"""
Alpha Vantage Client for AIDAAN.
"""
import asyncio
import httpx
import logging
from typing import Any, Dict, List

from app.core.config.settings import settings

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


class AlphaVantageClient:

    def __init__(self):
        self.api_key = settings.ALPHA_VANTAGE_API_KEY
        self.timeout = 10.0

    async def _get(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Internal async HTTP GET.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(ALPHA_VANTAGE_BASE, params=params)
                response.raise_for_status()
                data = response.json()
            
                if "Error Message" in data:
                    return {"error": data["Error Message"]}
                if "Note" in data:
                    return {"error": "API rate limit reached. Please wait 1 minute."}
                if "Information" in data:
                    return {"error": "Premium endpoint not available on this API key."}

                return data
        
        except httpx.TimeoutException:
            return {"error": "Request timed out."}
        except Exception as exc:
            logger.error(f"Alpha Vantage fetch error: {exc}")
            return {"error": str(exc)}

    async def get_single_quote(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Returns fields: symbol, price, change, change_percent, volume, latest_day
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        raw = await self._get(params)

        if "error" in raw:
            return {"symbol": symbol, "error": raw["error"]}

        quote = raw.get("Global Quote", {})
        if not quote:
            return {"symbol": symbol, "error": "No data returned for this symbol."}

        return {
            "symbol":         quote.get("01. symbol", symbol),
            "price":          quote.get("05. price", "N/A"),
            "open":           quote.get("02. open", "N/A"),
            "high":           quote.get("03. high", "N/A"),
            "low":            quote.get("04. low", "N/A"),
            "volume":         quote.get("06. volume", "N/A"),
            "latest_day":     quote.get("07. latest trading day", "N/A"),
            "prev_close":     quote.get("08. previous close", "N/A"),
            "change":         quote.get("09. change", "N/A"),
            "change_percent": quote.get("10. change percent", "N/A"),
        }

    async def get_bulk_quotes(
        self,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Returns: { "data": [ {symbol, price, change_percent, ...}, ...] }
        """
        tasks = [self.get_single_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        return {"data": list(results)}
    
    async def get_daily_ohlcv(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Daily OHLCV - free tier, last 100 days
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key
        }
        return await self._get(params)
    
    async def get_intraday_ohlcv(
        self,
        symbol: str,
        interval: str = "5min"
    ) -> Dict[str, Any]:
        """
        Intraday OHLCV -- free tier.
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key
        }
        return await self._get(params)

alphavantage_client = AlphaVantageClient()
