"""
Gemini Client for AIDAAN
- Intent detection
- Company extraction
- Dynamic ticker resolution
- Clean JSON output (no parsing hacks)
"""
import json
import logging
import asyncio
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

import requests
import google.generativeai as genai

from app.core.config.settings import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


class GeminiClient:

    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)

        self.model = genai.GenerativeModel(
            model_name=settings.VERTEX_AI_MODEL_NAME,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )

        # Startup test
        try:
            res = self.model.generate_content("Return JSON: {\"status\": \"ok\"}")
            logger.info(f"[Gemini OK]: {res.text}")
        except Exception as e:
            logger.error(f"[Gemini FAIL]: {e}")

    # ----------------------------------------
    async def _generate(self, prompt: str, retries: int = 2) -> Dict:
        loop = asyncio.get_event_loop()

        for attempt in range(retries):
            try:
                response = await loop.run_in_executor(
                    _executor,
                    lambda: self.model.generate_content(prompt)
                )
                return json.loads(response.text)

            except Exception as e:
                logger.warning(f"[Gemini retry {attempt+1}] {e}")

                if attempt == retries - 1:
                    raise e

    # ----------------------------------------
    async def extract_companies(self, user_text: str) -> List[str]:
        prompt = f"""
        Extract all company or stock names from the text.

        Text: "{user_text}"

        Return JSON:
        {{"companies": ["Apple", "Tesla"]}}
        """

        try:
            result = await self._generate(prompt)
            return result.get("companies", [])
        except Exception as e:
            logger.error(f"[Extract Company FAILED]: {e}")
            return []

    # ----------------------------------------
    async def detect_intent(self, user_text: str) -> Dict[str, Any]:
        prompt = f"""
        Classify trading intent.

        Text: "{user_text}"

        Types:
        - quote
        - daily_ohlcv
        - intraday_ohlcv
        - unknown

        Return JSON:
        {{"type": "quote", "interval": "5min"}}
        """

        try:
            return await self._generate(prompt)
        except Exception as e:
            logger.error(f"[Intent FAILED]: {e}")
            return {"type": "unknown", "interval": "5min"}

    # ----------------------------------------
    def resolve_ticker(self, company: str) -> str:
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company}"
            res = requests.get(url, timeout=3)

            if "quotes" in res and len(res["quotes"]) > 0:
                return res["quotes"][0].get("symbol")

        except Exception as e:
            logger.error(f"[Ticker resolve FAILED]: {company} | {e}")

        return None

    # ----------------------------------------
    async def extract_market_intent(self, user_text: str) -> Dict[str, Any]:

        try:

            # Step 1: Extract companies
            companies = await self.extract_companies(user_text)

            # Step 2: Resolve tickers
            symbols = []
            for c in companies:
                ticker = self.resolve_ticker(c)
                if ticker:
                    symbols.append(ticker)

            # Step 3: Detect intent
            intent = await self.detect_intent(user_text)

            return {
                "type": intent.get("type", "unknown"),
                "symbols": symbols,
                "interval": intent.get("interval", "5min"),
                "context": ", ".join(companies)
            }

        except Exception as e:
            logger.error(f"[Pipeline FAILED]: {e}")
            return {
                "type": "unknown",
                "symbols": [],
                "interval": "5min",
                "context": ""
            }

    # ----------------------------------------
    async def format_market_response(
        self,
        user_text: str,
        market_data: Dict[str, Any],
        data_type: str,
        context: str = ""
    ) -> Dict[str, Any]:

        prompt = f"""
        You are AIDAAN, an institutional trading assistant.

        Question: "{user_text}"
        Context: {context}

        Market Data:
        {json.dumps(market_data)[:3000]}

        Return JSON:
        {{
          "reply": "short answer with numbers",
          "bullets": ["point1", "point2", "point3"]
        }}

        Rules:
        - Be precise
        - Include numbers
        - No buy/sell advice
        """

        try:
            result = await self._generate(prompt)
            return {
                "reply": result.get("reply", ""),
                "bullets": result.get("bullets", [])
            }

        except Exception as e:
            logger.error(f"[Format FAILED]: {e}")
            return self._fallback_format(market_data, data_type)

    # ----------------------------------------
    def _fallback_format(self, market_data, data_type):

        if "error" in market_data:
            return {
                "reply": f"Error: {market_data['error']}",
                "bullets": []
            }

        if "data" in market_data:
            bullets = []
            for d in market_data["data"][:5]:
                bullets.append(
                    f"{d.get('symbol')} ${d.get('price')} ({d.get('change_percent')})"
                )

            return {
                "reply": "Market data retrieved.",
                "bullets": bullets
            }

        return {
            "reply": "Data received.",
            "bullets": []
        }


# Singleton
gemini_client = GeminiClient()
