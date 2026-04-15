"""
Gemini Market Intent Client for AIDAAN
======================================
Uses heuristics first and Gemini only when needed, to keep token usage low.
"""
import logging
import hashlib
import re
from typing import Any, Dict, List

from app.core.cache import cache
from app.core.config.settings import settings
from app.core.llm_client import llm_client
from app.core.prompts import Prompts

logger = logging.getLogger(__name__)

INTERVAL_MAP = {
    "1min": "1min",
    "one minute": "1min",
    "5min": "5min",
    "5 minute": "5min",
    "five minute": "5min",
    "15min": "15min",
    "15 minute": "15min",
    "fifteen minute": "15min",
    "30min": "30min",
    "30 minute": "30min",
    "thirty minute": "30min",
    "60min": "60min",
    "60 minute": "60min",
    "hourly": "60min",
    "1 hour": "60min",
}


class GeminiClient:
    """
    Market-specific helper that prefers rule-based parsing and formatting.
    """

    def parse_market_request_heuristic(self, user_text: str) -> Dict[str, Any]:
        """
        Fast local parser so we avoid Gemini for straightforward requests.
        """
        text = (user_text or "").strip()
        lowered = text.lower()

        query_type = "quote"
        if any(token in lowered for token in ["intraday", "minute", "hourly", "1min", "5min", "15min", "30min", "60min"]):
            query_type = "intraday_ohlcv"
        elif any(token in lowered for token in ["daily", "day-wise", "historical", "history", "ohlcv", "open", "high", "low", "close", "week", "month"]):
            query_type = "daily_ohlcv"

        interval = "5min"
        for label, normalized in INTERVAL_MAP.items():
            if label in lowered:
                interval = normalized
                break

        tickers = re.findall(r"\b[A-Z][A-Z0-9.\-]{1,9}\b", text)
        cleaned_tickers = [
            token for token in tickers
            if token.upper() not in {"OHLCV", "RSI", "USD", "INR", "ETF"}
        ]

        company_phrases = []
        for separator in ["compare", "vs", "and", ","]:
            if separator in lowered:
                parts = re.split(r"\bcompare\b|\bvs\b|\band\b|,", text, flags=re.IGNORECASE)
                company_phrases = [part.strip(" ?.") for part in parts if part.strip(" ?.")]
                break

        identifiers = cleaned_tickers or company_phrases or [text]
        looks_like_ticker_query = bool(cleaned_tickers) and len(cleaned_tickers) == len(identifiers)

        if not text:
            query_type = "unknown"
            identifiers = []

        return {
            "query_type": query_type if identifiers else "unknown",
            "identifiers": identifiers[:4],
            "interval": interval,
            "needs_symbol_search": not looks_like_ticker_query,
            "source": "heuristic",
        }

    async def parse_market_request(self, user_text: str) -> Dict[str, Any]:
        """
        Use heuristics first, then Gemini only for ambiguous requests.
        """
        heuristic = self.parse_market_request_heuristic(user_text)
        if heuristic["query_type"] != "unknown" and heuristic["identifiers"]:
            return heuristic

        digest = hashlib.sha256((user_text or "").encode("utf-8")).hexdigest()
        cache_key = f"market:parse:{digest}"
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        prompt = Prompts.MARKET_REQUEST_PARSER.format(text=user_text)

        try:
            result = await llm_client.generate_json(prompt)
            identifiers = result.get("identifiers", [])
            if not isinstance(identifiers, list):
                identifiers = []

            interval = result.get("interval", "5min") or "5min"
            parsed = {
                "query_type": result.get("query_type", "unknown"),
                "identifiers": identifiers[:4],
                "interval": interval,
                "needs_symbol_search": bool(result.get("needs_symbol_search", True)),
                "source": "gemini",
            }
            cache.set(cache_key, parsed, expire=settings.LLM_CACHE_EXPIRE)
            return parsed
        except Exception as exc:
            logger.error("[GeminiClient.parse_market_request] FAILED: %s", exc)
            return heuristic

    async def format_market_response(
        self,
        user_text: str,
        market_data: Dict[str, Any],
        data_type: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Low-frequency fallback formatter. The main market flow now formats locally.
        """
        import json as _json

        prompt = Prompts.FORMAT_MARKET_RESPONSE.format(
            user_text=user_text,
            context=context,
            data_type=data_type,
            market_data=_json.dumps(market_data)[:3000],
        )

        try:
            result = await llm_client.generate_json(prompt)
            return {
                "reply": result.get("reply", ""),
                "bullets": result.get("bullets", []),
            }
        except Exception as exc:
            logger.error("[GeminiClient.format_market_response] FAILED: %s", exc)
            return self._fallback_format(market_data, data_type)

    def _fallback_format(self, market_data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        if market_data.get("error"):
            return {"reply": f"Error: {market_data['error']}", "bullets": []}

        if data_type == "quote" and "quotes" in market_data:
            bullets: List[str] = []
            for item in market_data["quotes"][:5]:
                if item.get("error"):
                    bullets.append(f"{item.get('symbol', 'Unknown')}: {item['error']}")
                    continue
                bullets.append(
                    f"{item.get('symbol')} ${item.get('price')} ({item.get('change_percent')})"
                )
            return {"reply": "Market data retrieved.", "bullets": bullets}

        return {"reply": "Data received.", "bullets": []}


gemini_client = GeminiClient()
