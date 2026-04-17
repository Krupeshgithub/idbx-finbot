"""
Institutional Greeting Agent - AIDAAN Personalized Experience Layer
==================================================================
The GreetingAgent provides warm, professional, and personalized welcomes 
to traders, leveraging session context and timezone-aware logic.

Key Features:
- Heuristic-based instant responses for common greetings (Token optimized).
- Personalized greetings using username and desk metadata.
- Contextual synthesis of system tips and market summaries.
"""

import logging
import datetime
import pytz
from typing import Any, Dict, List, Optional

from app.services.aidaan.core.base import BaseAgent
from app.schemas.aidaan import AidaanMessageResponse
from app.core.prompts import Prompts
from app.core.config.settings import settings
from app.services.aidaan.providers.risk_provider import JSONRiskProvider
import os

logger = logging.getLogger(__name__)


class GreetingAgent(BaseAgent):
    """
    Expert in professional welcomes and desk-ready interactions.
    Handles personalized greetings to improve user engagement and trust.
    """

    def __init__(self):
        """
        Initialize the agent with a standard conversation model.
        """
        super().__init__(name="greeting", model_name=settings.VERTEX_AI_MODEL_NAME)
        # Load risk data for premium greetings
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "providers", "data_dictionary.json")
        self.risk_provider = JSONRiskProvider(data_path)

    def _get_time_of_day(self, timezone_str: str = "Asia/Kolkata") -> str:
        """
        Determines the current time of day category for a given timezone.
        """
        try:
            tz = pytz.timezone(timezone_str)
        except Exception:
            tz = pytz.timezone("Asia/Kolkata")
            
        now = datetime.datetime.now(tz)
        hour = now.hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        """
        Synthesizes a warm, professional welcome message.

        Args:
            text: Raw input (e.g., "Hello AIDAAN").
            conversation_id: Session identifier.
            context: User metadata including 'username' and 'timezone'.

        Returns:
            AidaanMessageResponse: A personalized greeting with market tips.
        """
        context = context or {}
        username = context.get("username", "Trader")
        timezone = context.get("timezone", "Asia/Kolkata")
        time_of_day = self._get_time_of_day(timezone)
        
        # --- Heuristic Optimization ---
        # If it's a simple greeting, use the calculated time of day immediately.
        lowered = text.lower().strip()
        risk_metrics = self.risk_provider.get_risk_metrics("EUR/USD")
        soft_limit = risk_metrics.get("soft_limit", "50,000,000 USD")
        
        if lowered in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]:
            cap_time = time_of_day.capitalize()
            return self.build_message_response(
                reply=f"Good {time_of_day}, {username}. Welcome to AIDAAN. How may I assist your desk today?\n"
                      f"Your current **EUR/USD** notional soft limit is **{soft_limit}**.",
                bullets=["AIDAAN Desk Link Active.", f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}"],
                conversation_id=conversation_id,
                model_info={"agent": self.name, "llm": "premium-heuristic"},
            )

        # --- LLM Synthesis ---
        # For more complex introductory messages or introductions.
        prompt = Prompts.GREETING_SYNTHESIS.format(
            username=username,
            time_of_day=time_of_day,
            text=text,
            soft_limit=soft_limit
        )
        
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=username,
            )
            return self.build_message_response(
                reply=parsed.get("reply", f"Good {time_of_day}, {username}."),
                bullets=parsed.get("bullets", ["Ready for trade orchestration."]),
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as e:
            logger.warning(f"[GreetingAgent] Synthesis failed: {str(e)}")
            return self.build_message_response(
                reply=f"Good {time_of_day}, {username}. AIDAAN Desk is active and ready for your commands.",
                bullets=["Session securely established."],
                conversation_id=conversation_id,
                model_info={"agent": self.name, "llm": "fallback"},
            )

    def get_capabilities(self) -> List[str]:
        """
        Lists interaction capabilities.
        """
        return ["personalized_greeting", "token_optimized_welcome", "institutional_etiquette"]


# Singleton instance
greeting_agent = GreetingAgent()
