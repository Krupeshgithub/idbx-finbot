"""
Greeting Agent for AIDAAN
=========================
Handles timezone-aware, personalized welcomes for institutional traders.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for environments without zoneinfo
    ZoneInfo = None

from app.services.aidaan.base import BaseAgent
from app.core.prompts import Prompts
from app.schemas.aidaan import AidaanMessageResponse

logger = logging.getLogger(__name__)


class GreetingAgent(BaseAgent):
    """
    Specialized agent for personalized user greetings.
    """

    def __init__(self):
        super().__init__(
            name="greeting"
            # model_name defaults to settings.VERTEX_AI_MODEL_NAME in BaseAgent
        )

    async def handle_message(
        self,
        text: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Generates a personalized greeting based on time of day and timezone.
        """
        context = context or {}
        username = context.get("username", "Trader")
        timezone_str = context.get("timezone", "UTC")
        
        time_of_day = self._get_time_of_day(timezone_str)
        
        prompt = Prompts.GREETING_TEMPLATE.format(
            username=username,
            time_of_day=time_of_day,
            timezone=timezone_str
        )
        
        logger.info(f"[GreetingAgent] Generating greeting for {username} in {timezone_str}")
        
        try:
            # Using generate_json (async fallback) or generate_json_sync if needed
            # BaseAgent provides self.llm (LLMClient)
            parsed = await self.llm.generate_json(prompt)
            
            if "error" in parsed:
                return self._fallback_greeting(username, time_of_day, conversation_id)

            return AidaanMessageResponse(
                reply=parsed.get("reply", f"{time_of_day}, {username}! How can I help you today?"),
                bullets=parsed.get("bullets", ["AIDAAN Institutional Desk is ready."]),
                actions=[], # No specific actions for greeting yet
                conversation_id=conversation_id,
                model=self.get_model_info()
            )

        except Exception as e:
            logger.error(f"[GreetingAgent] Error: {e}")
            return self._fallback_greeting(username, time_of_day, conversation_id)

    def _get_time_of_day(self, timezone_str: str) -> str:
        """
        Calculates 'Good morning', 'Good afternoon', or 'Good evening' based on timezone.
        """
        try:
            if ZoneInfo:
                tz = ZoneInfo(timezone_str)
                now = datetime.now(tz)
            else:
                now = datetime.utcnow() # Fallback if no zoneinfo

            hour = now.hour
            if 5 <= hour < 12:
                return "Good morning"
            elif 12 <= hour < 17:
                return "Good afternoon"
            elif 17 <= hour < 22:
                return "Good evening"
            else:
                return "Good evening" # Or 'Good night' if preferred for institutional
        except Exception as e:
            logger.warning(f"[GreetingAgent] Timezone error for {timezone_str}: {e}")
            return "Hello"

    def _fallback_greeting(self, username: str, time_of_day: str, conversation_id: str) -> AidaanMessageResponse:
        """
        Static fallback if LLM fails.
        """
        return AidaanMessageResponse(
            reply=f"{time_of_day}, {username}! How can I help you with the desk today?",
            bullets=["AIDAAN Institutional Desk is standing by."],
            actions=[],
            conversation_id=conversation_id,
            model=self.get_model_info()
        )

    def get_capabilities(self) -> List[str]:
        return ["personalized_greeting", "timezone_awareness"]


greeting_agent = GreetingAgent()
