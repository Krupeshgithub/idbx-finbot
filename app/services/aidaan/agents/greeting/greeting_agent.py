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
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.services.aidaan.core.base import BaseAgent
from app.schemas.aidaan import AidaanMessageResponse
from app.core.prompts import Prompts

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
        super().__init__(name="greeting", model_name="gemini-1.5-pro")

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Synthesizes a warm, professional welcome message.

        Args:
            text: Raw input (e.g., "Hello AIDAAN").
            conversation_id: Session identifier.
            context: User metadata including 'username'.

        Returns:
            AidaanMessageResponse: A personalized greeting with market tips.
        """
        context = context or {}
        username = context.get("username", "Trader")
        
        # --- Heuristic Optimization ---
        # If it's a sterile greeting, avoid LLM call to save tokens and minimize latency.
        lowered = text.lower().strip()
        if lowered in ["hi", "hello", "hey", "good morning", "good evening"]:
            return AidaanMessageResponse(
                reply=f"Hello, {username}. AIDAAN Desk is standing by. How can I assist with your trades today?",
                bullets=["AIDAAN is ready for Market Analysis & Order Staging."],
                actions=[],
                conversation_id=conversation_id,
                model={"agent": self.name, "llm": "heuristic"}
            )

        # --- LLM Synthesis ---
        # For more complex introductory messages or introductions.
        prompt = Prompts.GREETING_SYNTHESIS.format(
            username=username,
            text=text
        )
        
        try:
            parsed = await self.llm.generate_json(prompt)
            return AidaanMessageResponse(
                reply=parsed.get("reply", f"Welcome back, {username}."),
                bullets=parsed.get("bullets", ["Ready for trade orchestration."]),
                actions=[],
                conversation_id=conversation_id,
                model=self.get_model_info()
            )
        except Exception as e:
            logger.warning(f"[GreetingAgent] Synthesis failed: {str(e)}")
            return AidaanMessageResponse(
                reply=f"Good day, {username}. AIDAAN Desk is active and ready for your commands.",
                bullets=["Session securely established."],
                actions=[],
                conversation_id=conversation_id,
                model={"agent": self.name, "llm": "fallback"}
            )

    def get_capabilities(self) -> List[str]:
        """
        Lists interaction capabilities.
        """
        return ["personalized_greeting", "token_optimized_welcome", "institutional_etiquette"]


# Singleton instance
greeting_agent = GreetingAgent()
