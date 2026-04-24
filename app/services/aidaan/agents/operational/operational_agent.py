"""
Operational data agent for public-schema reads and aidaan sidecar history.
"""
import logging
from typing import Any, Dict, List, Optional

from app.core.config.settings import settings
from app.core.prompts import Prompts
from app.schemas.aidaan import AidaanMessageResponse
from app.services.aidaan.core.base import BaseAgent

logger = logging.getLogger(__name__)


class OperationalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="operational", model_name=settings.VERTEX_AI_MODEL_NAME)

    async def handle_message(
        self,
        text: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        context = context or {}
        prompt = Prompts.OPERATIONAL_ORCHESTRATION.format(text=text)
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=context.get("username"),
                use_mcp_tools=True,
                tool_callback=tool_callback,
                response_schema=Prompts.OPERATIONAL_RESPONSE_SCHEMA,
                system_instruction=(
                    Prompts.TRADER_SYSTEM_INSTRUCTION
                    + "\nRole: Operational data specialist. Use tools before answering table-backed questions."
                    + " Use recent conversation memory to distinguish a true history query from a simple follow-up acceptance."
                    + " If the user gives a brief continuation to a prior specialist follow-up, do not answer with a generic ambiguity explanation."
                ),
            )
            return self.build_message_response(
                reply=parsed.get("reply", "Operational data retrieved."),
                bullets=parsed.get("bullets", []),
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as exc:
            logger.warning("[OperationalAgent] Tool-backed synthesis failed: %s", exc)
            return self.build_error_response(
                reply="I couldn't complete the operational data lookup cleanly.",
                conversation_id=conversation_id,
                error=exc,
                bullets=["Operational data retrieval failed."],
                model_info=self.get_model_info(),
            )

    def get_capabilities(self) -> List[str]:
        return [
            "public_schema_reading",
            "aidaan_history_lookup",
            "desk_context_queries",
            "counterparty_search",
            "audit_and_tool_trace_lookup",
        ]


operational_agent = OperationalAgent()
