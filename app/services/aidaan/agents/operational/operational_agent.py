"""
Operational data agent for public-schema reads and aidaan sidecar history.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from app.core.config.settings import settings
from app.core.prompts import Prompts
from app.schemas.aidaan import AidaanMessageResponse
from app.services.aidaan.core.base import BaseAgent
from app.services.aidaan.runtime_context import runtime_context_service

logger = logging.getLogger(__name__)


class OperationalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="operational", model_name=settings.VERTEX_AI_MODEL_NAME)

    @staticmethod
    def _clean_message_text(value: Any) -> str:
        """
        Normalize stored conversation text for concise history summaries.
        """
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _is_control_like_message(text: str) -> bool:
        """
        Detect brief acknowledgement turns so history summaries can highlight
        the last substantive ask instead of a simple yes/no.
        """
        compact = text.strip().lower()
        return compact in {
            "yes", "y", "haan", "ha", "han", "yep", "yeah", "ok", "okay", "sure",
            "continue", "proceed", "do it", "go ahead",
            "no", "n", "na", "nah", "nahi", "nope", "stop", "cancel",
            "oh", "what", "what?", "huh", "confused", "simplify",
        }

    @staticmethod
    def _is_substantive_control_turn(text: str) -> bool:
        """
        Detect longer stop/clarify instructions so they are not treated as
        substantive user asks in history summaries.
        """
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return any(
            phrase in normalized
            for phrase in [
                "stop that",
                "stop this",
                "stop the thread",
                "ignore your previous follow-up",
                "just acknowledge and wait",
                "do not want the follow-up",
                "do not want the comparison",
                "do not continue",
            ]
        )

    def _extract_history_facts(self, conversation_id: str) -> Dict[str, Any]:
        """
        Build a compact deterministic history summary from persisted memory.
        """
        history = runtime_context_service.get_recent_history(conversation_id)
        continuity = runtime_context_service.build_continuity_guidance(history)

        user_messages = [
            self._clean_message_text(item.get("content"))
            for item in history
            if item.get("role") == "user" and self._clean_message_text(item.get("content"))
        ]
        assistant_messages = [
            self._clean_message_text(item.get("content"))
            for item in history
            if item.get("role") == "assistant" and self._clean_message_text(item.get("content"))
        ]
        substantive_user_messages = [
            message
            for message in user_messages
            if not self._is_control_like_message(message) and not self._is_substantive_control_turn(message)
        ]

        facts = {
            "history_count": len(history),
            "last_user_message": user_messages[-1] if user_messages else "",
            "last_substantive_user_message": substantive_user_messages[-1] if substantive_user_messages else "",
            "previous_substantive_user_message": substantive_user_messages[-2] if len(substantive_user_messages) >= 2 else "",
            "last_assistant_message": assistant_messages[-1] if assistant_messages else "",
            "pending_follow_up": bool(continuity.get("pending_follow_up")),
            "last_follow_up_prompt": self._clean_message_text(continuity.get("last_follow_up_prompt")),
        }
        logger.info(
            "[OperationalAgent] History facts extracted | history_count=%s user_messages=%s assistant_messages=%s pending_follow_up=%s",
            facts["history_count"],
            len(user_messages),
            len(assistant_messages),
            facts["pending_follow_up"],
        )
        return facts

    def _build_history_response(self, *, conversation_id: str) -> AidaanMessageResponse:
        """
        Return a strict no-tool conversation-memory summary.
        """
        logger.info("[OperationalAgent] History fast path engaged | conversation_id=%s", conversation_id)
        facts = self._extract_history_facts(conversation_id)

        if facts["history_count"] == 0:
            reply = (
                "[Direct Answer]\nI do not see prior conversation history for this session.\n\n"
                "[Context Detail]\nAsk a few questions first, and I can summarize them back to you."
            )
            bullets = ["No persisted history found"]
        else:
            primary_ask = facts["last_substantive_user_message"] or facts["last_user_message"] or "No prior user ask found."
            previous_ask = facts["previous_substantive_user_message"]
            last_assistant = facts["last_assistant_message"] or "No prior assistant reply found."

            detail_lines = [f'Your most recent substantive query was: "{primary_ask}"']
            if previous_ask:
                detail_lines.append(f'An earlier substantive query was: "{previous_ask}"')
            detail_lines.append(f'My most recent reply was about: "{last_assistant[:220]}"')
            if facts["pending_follow_up"] and facts["last_follow_up_prompt"]:
                detail_lines.append(f'Pending follow-up: "{facts["last_follow_up_prompt"]}"')

            reply = "[Direct Answer]\n" + detail_lines[0] + "\n\n[Context Detail]\n" + "\n".join(detail_lines[1:])

            bullets = [f'Last substantive ask: "{primary_ask}"']
            if previous_ask:
                bullets.append(f'Previous ask: "{previous_ask}"')
            if facts["pending_follow_up"] and facts["last_follow_up_prompt"]:
                bullets.append(f'Pending follow-up: "{facts["last_follow_up_prompt"]}"')
            else:
                bullets.append("No pending follow-up detected")

        logger.info(
            "[OperationalAgent] History response built | bullets=%s reply_preview=%s",
            len(bullets),
            reply[:160],
        )
        return self.build_message_response(
            reply=reply,
            bullets=bullets,
            conversation_id=conversation_id,
            model_info=self.get_model_info(),
        )

    def _build_control_response(
        self,
        *,
        conversation_id: str,
        control_signal: str,
    ) -> AidaanMessageResponse:
        """
        Return an immediate operational acknowledgement for stop/clarify style turns.
        """
        logger.info(
            "[OperationalAgent] Control fast path engaged | conversation_id=%s control=%s",
            conversation_id,
            control_signal,
        )
        if control_signal == "stop":
            reply = (
                "[Direct Answer]\nAcknowledged. I have stopped the active follow-up flow.\n\n"
                "[Operational Detail]\nNo further operational lookup or continuation will be attempted until you start a new request."
            )
            bullets = ["Pending follow-up closed", "No tools were used"]
        else:
            reply = (
                "[Direct Answer]\nI can simplify or restate the recent context.\n\n"
                "[Operational Detail]\nAsk what you want clarified, and I will answer from session memory without opening a new workflow."
            )
            bullets = ["Clarification mode active", "No tools were used"]

        return self.build_message_response(
            reply=reply,
            bullets=bullets,
            conversation_id=conversation_id,
            model_info=self.get_model_info(),
        )

    async def _build_conversation_logic_response(
        self,
        *,
        text: str,
        conversation_id: str,
    ) -> AidaanMessageResponse:
        """
        Answer conversation-state questions from recent memory without using tools.
        """
        logger.info(
            "[OperationalAgent] Conversation-logic fast path engaged | conversation_id=%s",
            conversation_id,
        )
        history = runtime_context_service.get_recent_history(conversation_id)
        continuity = runtime_context_service.build_continuity_guidance(history)
        prompt = (
            f'User Query: "{text}"\n\n'
            "Answer using only recent conversation state.\n"
            "Explain whether the user is asking about a fresh query, a follow-up, stop ownership, or conversation interpretation.\n"
            "Do not provide market analysis, price commentary, or tool-backed data.\n"
            f"CONTINUITY_GUIDANCE_JSON: {continuity}\n"
            f"RECENT_HISTORY_JSON: {history}\n\n"
            "Return JSON: {\"reply\": \"...\", \"bullets\": [\"...\"]}"
        )
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=None,
                username=None,
                use_mcp_tools=False,
                response_schema=Prompts.OPERATIONAL_RESPONSE_SCHEMA,
                system_instruction=(
                    Prompts.TRADER_SYSTEM_INSTRUCTION
                    + "\nRole: Conversation-state specialist."
                    + " Answer only about routing, follow-up ownership, continuity, or interpretation of the recent chat."
                    + " Do not answer with market analysis or fresh trading commentary."
                    + " Keep the answer direct and operational."
                ),
            )
            logger.info(
                "[OperationalAgent] Conversation-logic response generated | bullets=%s reply_preview=%s",
                len(parsed.get("bullets", [])),
                str(parsed.get("reply", ""))[:160],
            )
            return self.build_message_response(
                reply=parsed.get("reply", "Conversation-state reasoning complete."),
                bullets=parsed.get("bullets", []),
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as exc:
            logger.warning("[OperationalAgent] Conversation-logic synthesis failed: %s", exc)
            return self.build_error_response(
                reply="I couldn't resolve the conversation-state question cleanly.",
                conversation_id=conversation_id,
                error=exc,
                bullets=["Conversation-state reasoning failed."],
                model_info=self.get_model_info(),
            )

    async def handle_message(
        self,
        text: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        context = context or {}
        routing = context.get("routing") or {}
        sub_intent = routing.get("sub_intent", "general")
        control_signal = routing.get("control_signal", "none")
        logger.info(
            "[OperationalAgent] Routing context received | sub_intent=%s control=%s confidence=%s reason=%s",
            sub_intent,
            control_signal,
            routing.get("confidence"),
            routing.get("reason"),
        )

        if sub_intent == "history_lookup":
            return self._build_history_response(conversation_id=conversation_id)
        if sub_intent == "conversation_logic":
            return await self._build_conversation_logic_response(
                text=text,
                conversation_id=conversation_id,
            )
        if sub_intent == "control" and control_signal in {"stop", "clarify"}:
            return self._build_control_response(
                conversation_id=conversation_id,
                control_signal=control_signal,
            )

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
            logger.info(
                "[OperationalAgent] Tool-backed response generated | bullets=%s reply_preview=%s",
                len(parsed.get("bullets", [])),
                str(parsed.get("reply", ""))[:160],
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
