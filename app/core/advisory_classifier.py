"""
Advisory Intent Classifier for AIDANN

Detects financial advisory intent in user queries and reframes responses
to comply with financial advisory regulations.
"""
import logging
import re
from typing import Dict, Any, Optional, List

from app.core.llm_client import llm_client
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class AdvisoryIntentClassifier:
    """
    Detects advisory intent in user queries using LLM-based classification
    and pattern matching.
    """
    
    # Explicit advisory patterns
    ADVISORY_PATTERNS = [
        r"\bshould\s+i\s+(?:buy|sell|trade|invest|hold)\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\bis\s+this\s+a\s+good\s+(?:trade|investment|deal)\b",
        r"\bdo\s+you\s+(?:recommend|suggest|advise)\b",
        r"\bwould\s+you\s+(?:recommend|suggest|advise)\b",
        r"\byour\s+(?:recommendation|advice|opinion)\b",
        r"\btell\s+me\s+(?:what|which)\s+to\s+(?:buy|sell|trade)\b",
        r"\bhelp\s+me\s+(?:decide|choose)\b",
    ]
    
    # Recommendation phrases to block in responses
    RECOMMENDATION_PHRASES = [
        r"\byou\s+should\s+(?:buy|sell|trade|invest|hold)\b",
        r"\bi\s+(?:recommend|suggest|advise)\b",
        r"\bmy\s+(?:recommendation|advice)\s+is\b",
        r"\bthis\s+is\s+a\s+good\s+(?:trade|investment|opportunity)\b",
        r"\byou\s+might\s+want\s+to\s+(?:buy|sell|consider)\b",
        r"\bit\s+would\s+be\s+(?:wise|smart|good)\s+to\b",
    ]
    
    def __init__(self):
        self.confidence_threshold = settings.ADVISORY_CONFIDENCE_THRESHOLD
        self._cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    def detect_advisory_intent(
        self,
        text: str,
        *,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect advisory intent in user query.
        Fast path: pattern matching first (zero LLM cost).
        Slow path: LLM only for short/ambiguous queries with no pattern match.
        """
        if not text or not text.strip():
            return {
                "is_advisory": False,
                "confidence": 0.0,
                "patterns_matched": [],
                "reason": "Empty query"
            }

        # ── Fast-reject: clearly non-advisory operational queries ──────────
        # These are always data/calculation requests — skip LLM entirely.
        NON_ADVISORY_PREFIXES = (
            "fetch ", "get ", "pull ", "show ", "calculate ", "compute ",
            "what is the ", "what are the ", "stage ", "rfq ",
        )
        NON_ADVISORY_KEYWORDS = {
            "rsi", "macd", "sma", "ema", "bbands", "bollinger",
            "price", "quote", "daily series", "historical",
            "dv01", "pv01", "var ", "volatility", "sharpe",
            "exchange rate", "federal funds", "cpi", "gdp",
            "earnings", "income statement", "balance sheet",
            "news", "sentiment", "finbert",
        }
        lowered = text.strip().lower()
        if any(lowered.startswith(p) for p in NON_ADVISORY_PREFIXES):
            return {
                "is_advisory": False,
                "confidence": 0.0,
                "patterns_matched": [],
                "reason": "Non-advisory operational query (fast-reject by prefix)"
            }
        if any(k in lowered for k in NON_ADVISORY_KEYWORDS) and len(lowered.split()) > 8:
            # Long queries with data keywords are almost never advisory
            return {
                "is_advisory": False,
                "confidence": 0.0,
                "patterns_matched": [],
                "reason": "Non-advisory data/calculation query (fast-reject by keyword)"
            }

        # Check cache first
        if conversation_id:
            cache_key = f"advisory:{conversation_id}:{hash(text)}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(
                    "[AdvisoryClassifier] Cache HIT | conv_id=%s",
                    conversation_id
                )
                return cached_result
        
        # Pattern-based detection (fast path — zero LLM cost)
        patterns_matched = self._match_advisory_patterns(text)
        if patterns_matched:
            result = {
                "is_advisory": True,
                "confidence": 1.0,
                "patterns_matched": patterns_matched,
                "reason": "Explicit advisory patterns detected"
            }
            if conversation_id:
                self._put_in_cache(cache_key, result)
            logger.warning(
                "[AdvisoryClassifier] Advisory intent detected | patterns=%s | text=%s",
                patterns_matched,
                text[:100]
            )
            return result
        
        # LLM-based detection ONLY for short/ambiguous queries
        # Skip LLM for long queries (>15 words) — they're almost always data requests
        word_count = len(text.strip().split())
        if word_count > 15:
            result = {
                "is_advisory": False,
                "confidence": 0.1,
                "patterns_matched": [],
                "reason": "Long query without advisory patterns — skipping LLM (cost optimization)"
            }
            if conversation_id:
                self._put_in_cache(cache_key, result)
            return result

        llm_result = self._llm_detect_advisory(text)
        
        result = {
            "is_advisory": llm_result["confidence"] >= self.confidence_threshold,
            "confidence": llm_result["confidence"],
            "patterns_matched": [],
            "reason": llm_result["reason"]
        }
        
        if conversation_id:
            self._put_in_cache(cache_key, result)
        
        if result["is_advisory"]:
            logger.warning(
                "[AdvisoryClassifier] Advisory intent detected via LLM | confidence=%.2f | text=%s",
                result["confidence"],
                text[:100]
            )
        
        return result
    
    def _match_advisory_patterns(self, text: str) -> List[str]:
        """Match explicit advisory patterns in text."""
        normalized = text.lower()
        matched = []
        
        for pattern in self.ADVISORY_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                matched.append(pattern)
        
        return matched
    
    def _llm_detect_advisory(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to detect advisory intent in subtle cases.
        
        Returns:
            {
                "confidence": float,
                "reason": str
            }
        """
        prompt = f"""You are a compliance classifier for a financial trading assistant.

        Your task is to determine if the following user query is seeking financial advice or recommendations.

        Advisory queries include:
        - Asking what to buy/sell/trade
        - Asking if something is a good investment
        - Asking for recommendations or suggestions
        - Asking what action to take

        Non-advisory queries include:
        - Asking for market data or facts
        - Asking for risk calculations
        - Asking for historical information
        - Asking for operational information

        User query: "{text}"

        Respond with ONLY a JSON object in this exact format:
        {{
        "is_advisory": true or false,
        "confidence": 0.0 to 1.0,
        "reason": "brief explanation"
        }}"""

        try:
            response = llm_client.generate_json_sync(
                prompt=prompt,
                model_override=settings.GEMINI_MODEL_NAME,
                timeout=8
            )
            
            return {
                "confidence": float(response.get("confidence", 0.0)),
                "reason": response.get("reason", "LLM classification")
            }
        
        except Exception as exc:
            logger.error(
                "[AdvisoryClassifier] LLM detection failed: %s",
                exc
            )
            # Fail-safe: assume not advisory if LLM fails
            return {
                "confidence": 0.0,
                "reason": f"LLM detection failed: {exc}"
            }
    
    async def reframe_advisory_response(
        self,
        original_response: str,
        query: str
    ) -> str:
        """
        Reframe advisory response to factual information only.
        
        Args:
            original_response: Original response that may contain recommendations
            query: Original user query
        
        Returns:
            Reframed response without recommendations
        """
        # Check if response contains recommendation phrases
        has_recommendations = any(
            re.search(pattern, original_response, re.IGNORECASE)
            for pattern in self.RECOMMENDATION_PHRASES
        )
        
        if not has_recommendations:
            # Response is already compliant
            return original_response
        
        # Use LLM to reframe the response
        prompt = f"""You are a compliance officer for a financial trading assistant.

        The user asked: "{query}"

        The assistant generated this response: "{original_response}"

        This response contains financial advice or recommendations, which violates regulations.

        Rewrite the response to provide ONLY factual information without any recommendations or advice.

        Guidelines:
        - Remove phrases like "you should", "I recommend", "this is a good"
        - Replace recommendations with factual data
        - Provide risk metrics, historical data, and current market conditions
        - Do NOT suggest any action

        Reframed response:"""

        try:
            reframed = await llm_client.generate_text(
                prompt=prompt,
                model_override=settings.GEMINI_MODEL_NAME
            )
            
            logger.info(
                "[AdvisoryClassifier] Response reframed | original_length=%d | reframed_length=%d",
                len(original_response),
                len(reframed)
            )
            
            return reframed.strip()
        
        except Exception as exc:
            logger.error(
                "[AdvisoryClassifier] Response reframing failed: %s",
                exc
            )
            # Fail-safe: return generic non-advisory response
            return (
                "I can provide you with factual market data and risk metrics, "
                "but I cannot provide investment advice or recommendations. "
                "Please consult with a licensed financial advisor for investment decisions."
            )
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache if not expired."""
        import time
        
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            age = time.time() - timestamp
            if age < self._cache_ttl:
                return cached_data
            else:
                del self._cache[cache_key]
        
        return None
    
    def _put_in_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store value in cache with timestamp."""
        import time
        
        # Simple cache eviction: remove oldest if cache is too large
        if len(self._cache) >= 100:
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            del self._cache[oldest_key]
        
        self._cache[cache_key] = (data, time.time())


# Global instance
advisory_classifier = AdvisoryIntentClassifier()
