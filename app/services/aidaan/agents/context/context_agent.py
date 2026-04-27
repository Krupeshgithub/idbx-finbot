"""
Operational context agent for desk/user/rfq history queries.
"""
import logging
from typing import Any, Dict, List, Optional

from app.core.config.settings import settings
from app.core.prompts import Prompts
from app.schemas.aidaan import AidaanMessageResponse
from app.services.aidaan.core.base import BaseAgent
from app.services.aidaan.runtime_context import runtime_context_service

logger = logging.getLogger(__name__)


class ContextAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="context", model_name=settings.VERTEX_AI_MODEL_NAME)

    async def handle_message(
        self,
        text: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        context = context or {}
        username = context.get("username")
        prompt = Prompts.CONTEXT_SYNTHESIS.format(text=text)
        prompt = runtime_context_service.build_prompt_context(
            base_prompt=prompt,
            conversation_id=conversation_id,
            username=username,
        )
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=username,
                system_instruction=(
                    Prompts.TRADER_SYSTEM_INSTRUCTION
                    + "\nRole: Operational and history specialist."
                    + " Use continuity guidance to resolve brief follow-up turns against the latest persisted assistant question."
                    + " Only answer with pure history/context output when the user is actually asking about prior conversation or operational records."
                ),
            )
            return self.build_message_response(
                reply=parsed.get("reply", "Operational context retrieved."),
                bullets=parsed.get("bullets", []),
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as e:
            logger.warning("[ContextAgent] Context synthesis failed: %s", e)
            return self.build_error_response(
                reply="I couldn't assemble the desk context cleanly.",
                conversation_id=conversation_id,
                error=e,
                bullets=["Operational context retrieval failed."],
                model_info=self.get_model_info(),
            )

    def get_capabilities(self) -> List[str]:
        return [
            "conversation_memory",
            "user_profile_context",
            "desk_context",
            "rfq_history",
            "operational_audit_summary",
        ]


context_agent = ContextAgent()
