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
from app.db.repositories import normalize_username
from app.db.operational.service import operational_data_service

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

    @staticmethod
    def _format_limit(limit_row: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Format a desk limit row for concise trader-facing output.
        """
        if not limit_row:
            return None

        soft_limit = limit_row.get("soft_limit")
        if soft_limit in (None, ""):
            return None

        currency = limit_row.get("currency") or "USD"
        try:
            return f"{float(soft_limit):,.0f} {currency}"
        except (TypeError, ValueError):
            return f"{soft_limit} {currency}".strip()

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
        resolved_username = normalize_username(context.get("username"))
        username = resolved_username or "Trader"
        timezone = context.get("timezone", "Asia/Kolkata")
        try:
            tz = pytz.timezone(timezone)
        except Exception:
            tz = pytz.timezone("Asia/Kolkata")
        now = datetime.datetime.now(tz)
        time_of_day = self._get_time_of_day(timezone)
        
        greeting_context = operational_data_service.get_greeting_context(resolved_username, instrument="EUR/USD")
        desk = greeting_context.get("desk") or {}
        membership = greeting_context.get("desk_membership") or {}
        selected_limit = greeting_context.get("selected_limit")
        soft_limit = self._format_limit(selected_limit)
        desk_name = desk.get("name")
        desk_title = membership.get("title")

        # --- LLM Synthesis ---
        prompt = Prompts.GREETING_SYNTHESIS.format(
            username=username,
            time_of_day=time_of_day,
            text=text,
            desk_name=desk_name or "Unknown desk",
            desk_title=desk_title or "Trader",
            soft_limit=soft_limit or "Unavailable"
        )
        
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=resolved_username,
            )
            return self.build_message_response(
                reply=parsed.get("reply", f"Good {time_of_day}, {username}."),
                bullets=parsed.get("bullets", ["Ready for trade orchestration."]),
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as e:
            logger.warning(f"[GreetingAgent] Synthesis failed: {str(e)}")
            reply = f"Good {time_of_day}, {username}. AIDAAN Desk is active and ready for your commands."
            if desk_name:
                reply = f"Good {time_of_day}, {username}. {desk_name} is active and ready for your commands."
            if soft_limit:
                reply += f" Your EUR/USD soft limit is {soft_limit}."
            return self.build_message_response(
                reply=reply,
                bullets=[f"Time: {now.strftime('%H:%M:%S %Z')}", "Session securely established."],
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
