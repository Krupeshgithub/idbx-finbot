"""
Gemini Market Intent Client for AIDAAN
=======================================
Handles market-specific LLM tasks:
  - Company name extraction
  - Intent detection (quote / ohlcv / intraday)
  - Natural language response formatting

CHANGES FROM ORIGINAL:
    - Removed duplicate `genai.configure()` and `GenerativeModel(...)`.
    - All prompts moved to `app.core.prompts.Prompts`.
    - All LLM calls go through the shared `llm_client`.
    - `_generate()` replaced with `llm_client.generate_json()`.
    - `resolve_ticker()` bug fixed: was calling `.get()` on a requests.Response object.
"""
import logging
from typing import Any, Dict, List

import requests

from app.core.llm_client import llm_client
from app.core.prompts import Prompts

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Market-specific LLM helper.
    Uses the shared llm_client — no direct Gemini SDK calls here.
    """

    # ------------------------------------------------------------------
    async def extract_companies(self, user_text: str) -> List[str]:
        """
        Extract company / stock names from free-form text.

        Returns:
            List of company names, e.g. ["Apple", "Tesla"]
        """
        prompt = Prompts.EXTRACT_COMPANIES.format(text=user_text)
        try:
            result = await llm_client.generate_json(prompt)
            return result.get("companies", [])
        except Exception as e:
            logger.error(f"[GeminiClient.extract_companies] FAILED: {e}")
            return []

    # ------------------------------------------------------------------
    async def detect_intent(self, user_text: str) -> Dict[str, Any]:
        """
        Classify market query type (quote / daily_ohlcv / intraday_ohlcv / unknown).

        Returns:
            {"type": "quote", "interval": "5min"}
        """
        prompt = Prompts.DETECT_INTENT.format(text=user_text)
        try:
            return await llm_client.generate_json(prompt)
        except Exception as e:
            logger.error(f"[GeminiClient.detect_intent] FAILED: {e}")
            return {"type": "unknown", "interval": "5min"}

    # ------------------------------------------------------------------
    def resolve_ticker(self, company: str) -> str:
        """
        Resolve a company name to a ticker symbol via Yahoo Finance search.

        Returns:
            Ticker string (e.g. "AAPL") or None if not found.

        Debug tip:
            If this returns None unexpectedly, check the Yahoo Finance URL
            response manually: https://query1.finance.yahoo.com/v1/finance/search?q=Apple
        """
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company}"
            res = requests.get(url, timeout=3)
            data = res.json()  # ← BUG FIX: original code called .get() on Response object

            quotes = data.get("quotes", [])
            if quotes:
                return quotes[0].get("symbol")

        except Exception as e:
            logger.error(f"[GeminiClient.resolve_ticker] FAILED for '{company}': {e}")

        return None

    # ------------------------------------------------------------------
    async def extract_market_intent(self, user_text: str) -> Dict[str, Any]:
        """
        Full pipeline: extract companies → resolve tickers → detect intent.

        Returns:
            {
                "type": "quote|daily_ohlcv|intraday_ohlcv|unknown",
                "symbols": ["AAPL", "TSLA"],
                "interval": "5min",
                "context": "Apple, Tesla"
            }
        """
        try:
            # Step 1: Extract companies
            companies = await self.extract_companies(user_text)

            # Step 2: Resolve tickers
            symbols = []
            for company in companies:
                ticker = self.resolve_ticker(company)
                if ticker:
                    symbols.append(ticker)

            # Step 3: Detect intent
            intent = await self.detect_intent(user_text)

            return {
                "type": intent.get("type", "unknown"),
                "symbols": symbols,
                "interval": intent.get("interval", "5min"),
                "context": ", ".join(companies),
            }

        except Exception as e:
            logger.error(f"[GeminiClient.extract_market_intent] Pipeline FAILED: {e}")
            return {
                "type": "unknown",
                "symbols": [],
                "interval": "5min",
                "context": "",
            }

    # ------------------------------------------------------------------
    async def format_market_response(
        self,
        user_text: str,
        market_data: Dict[str, Any],
        data_type: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Format raw Alpha Vantage data into a trader-friendly reply using Gemini.

        Returns:
            {"reply": "...", "bullets": ["...", "..."]}
        """
        import json as _json

        prompt = Prompts.FORMAT_MARKET_RESPONSE.format(
            user_text=user_text,
            context=context,
            market_data=_json.dumps(market_data)[:3000],
        )

        try:
            result = await llm_client.generate_json(prompt)
            return {
                "reply": result.get("reply", ""),
                "bullets": result.get("bullets", []),
            }
        except Exception as e:
            logger.error(f"[GeminiClient.format_market_response] FAILED: {e}")
            return self._fallback_format(market_data, data_type)

    # ------------------------------------------------------------------
    def _fallback_format(self, market_data: Dict, data_type: str) -> Dict[str, Any]:
        """
        Static fallback when Gemini formatting fails.
        """
        if "error" in market_data:
            return {"reply": f"Error: {market_data['error']}", "bullets": []}

        if "data" in market_data:
            bullets = []
            for d in market_data["data"][:5]:
                bullets.append(
                    f"{d.get('symbol')} ${d.get('price')} ({d.get('change_percent')})"
                )
            return {"reply": "Market data retrieved.", "bullets": bullets}

        return {"reply": "Data received.", "bullets": []}


# Singleton
gemini_client = GeminiClient()
