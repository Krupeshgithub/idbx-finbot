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
    _MAX_HISTORY_QUESTIONS = 3

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

    @staticmethod
    def _is_preference_query(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        cues = (
            "saved preference",
            "preference check",
            "overnight risk",
            "maine",
            "i said",
            "did i say",
            "bola tha",
            "avoid holding positions overnight",
        )
        return any(cue in normalized for cue in cues)

    @staticmethod
    def _is_low_context_message(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return False
        tokens = re.findall(r"[A-Za-z0-9]+", normalized)
        if len(tokens) > 3:
            return False
        return "?" not in normalized and not any(ch.isdigit() for ch in normalized)

    def _extract_history_facts(self, conversation_id: str, max_substantive: int = _MAX_HISTORY_QUESTIONS) -> Dict[str, Any]:
        """
        Build a compact deterministic history summary from persisted memory.
        
        Args:
            conversation_id: The conversation to extract facts from
            max_substantive: Maximum number of substantive user messages to extract (default 2)
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

        # Extract up to max_substantive messages
        facts = {
            "history_count": len(history),
            "last_user_message": user_messages[-1] if user_messages else "",
            "substantive_user_messages": substantive_user_messages[-max_substantive:] if substantive_user_messages else [],
            "last_assistant_message": assistant_messages[-1] if assistant_messages else "",
            "pending_follow_up": bool(continuity.get("pending_follow_up")),
            "last_follow_up_prompt": self._clean_message_text(continuity.get("last_follow_up_prompt")),
        }
        logger.info(
            "[OperationalAgent] History facts extracted | history_count=%s user_messages=%s assistant_messages=%s substantive_count=%s pending_follow_up=%s",
            facts["history_count"],
            len(user_messages),
            len(assistant_messages),
            len(facts["substantive_user_messages"]),
            facts["pending_follow_up"],
        )
        return facts

    def _build_history_response(self, *, conversation_id: str, requested_count: int = _MAX_HISTORY_QUESTIONS) -> AidaanMessageResponse:
        """
        Return a strict no-tool conversation-memory summary.
        
        Args:
            conversation_id: The conversation to summarize
            requested_count: Number of substantive conversations to return (default 2)
        """
        logger.info("[OperationalAgent] History fast path engaged | conversation_id=%s requested_count=%s", conversation_id, requested_count)
        
        # Optimization: Use get_recent_history instead of get_conversation_bundle
        # This avoids fetching RFQ drafts, tool invocations, and audit events
        bounded_count = max(1, min(requested_count, self._MAX_HISTORY_QUESTIONS))
        facts = self._extract_history_facts(conversation_id, max_substantive=bounded_count)

        if facts["history_count"] == 0:
            reply = (
                "[Direct Answer]\nI do not see prior conversation history for this session.\n\n"
                "[Context Detail]\nAsk a few questions first, and I can summarize them back to you."
            )
            bullets = ["No persisted history found"]
        else:
            substantive_messages = facts["substantive_user_messages"]
            last_assistant = facts["last_assistant_message"] or "No prior assistant reply found."

            if not substantive_messages:
                primary_ask = facts["last_user_message"] or "No prior user ask found."
                detail_lines = [f'Your most recent query was: "{primary_ask}"']
            else:
                # Build detail lines for all substantive messages (most recent first in display)
                detail_lines = []
                for idx, msg in enumerate(reversed(substantive_messages)):
                    if idx == 0:
                        detail_lines.append(f'Your most recent substantive query was: "{msg}"')
                    else:
                        detail_lines.append(f'An earlier substantive query was: "{msg}"')
            
            detail_lines.append(f'My most recent reply was about: "{last_assistant[:220]}"')
            if facts["pending_follow_up"] and facts["last_follow_up_prompt"]:
                detail_lines.append(f'Pending follow-up: "{facts["last_follow_up_prompt"]}"')

            reply = "[Direct Answer]\n" + detail_lines[0] + "\n\n[Context Detail]\n" + "\n".join(detail_lines[1:])

            bullets = []
            for idx, msg in enumerate(reversed(substantive_messages)):
                if idx == 0:
                    bullets.append(f'Last substantive ask: "{msg}"')
                else:
                    bullets.append(f'Earlier ask #{idx+1}: "{msg}"')
            
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

    def _build_clarification_response(self, *, conversation_id: str) -> AidaanMessageResponse:
        reply = (
            "[Direct Answer]\nI need a little more context to route this correctly.\n\n"
            "[Operational Detail]\nPlease share a fresh query (instrument/topic + what you want: price, compare, outlook, or history)."
        )
        return self.build_message_response(
            reply=reply,
            bullets=["Low-context input detected", "Awaiting explicit query"],
            conversation_id=conversation_id,
            model_info=self.get_model_info(model_override="low-context-clarify"),
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

    async def _build_corporate_knowledge_response(
        self,
        *,
        text: str,
        conversation_id: str,
    ) -> AidaanMessageResponse:
        """
        Answer IDBX/AIDANN corporate knowledge queries using semantic search.
        
        This handles questions about:
        - Company information and mission
        - Leadership (e.g., "Who is the Chairman?")
        - AIDANN capabilities and technology
        - Security boundaries and execution policies
        - Data privacy and DLP features
        """
        logger.info(
            "[OperationalAgent] Corporate knowledge fast path engaged | conversation_id=%s query=%s",
            conversation_id,
            text[:100],
        )
        
        try:
            from app.db.session import get_db_session
            from app.db.operational.aidaan_store import AidaanStoreRepository
            from app.services.aidaan.response_formatter import (
                format_corporate_knowledge_response,
                DataSource
            )
            
            store = AidaanStoreRepository()
            
            with get_db_session() as session:
                # Search corporate knowledge base
                results = store.search_corporate_knowledge(
                    session=session,
                    query=text,
                    limit=2,
                    similarity_threshold=0.6
                )
                
                if not results:
                    logger.info("[OperationalAgent] No corporate knowledge results found | query=%s", text[:100])
                    return self.build_message_response(
                        reply=(
                            f"I could not find a direct entry for '{text}' right now. "
                            f"Please share a little more context and I will answer in detail.\n\n"
                            f"**IDBX Data Provenance:** {DataSource.IDBX_CORPORATE_KB}"
                        ),
                        bullets=[],
                        conversation_id=conversation_id,
                        model_info=self.get_model_info(model_override="corporate-knowledge-empty"),
                    )
                
                # Format response with data provenance
                formatted_response = format_corporate_knowledge_response(results, text)
                
                logger.info(
                    "[OperationalAgent] Corporate knowledge response built | results=%s top_similarity=%.2f",
                    len(results),
                    results[0].get('similarity', 0) if results else 0
                )
                
                return self.build_message_response(
                    reply=formatted_response,
                    bullets=[],
                    conversation_id=conversation_id,
                    model_info=self.get_model_info(model_override="corporate-knowledge-semantic"),
                )
                
        except Exception as exc:
            logger.error("[OperationalAgent] Corporate knowledge search failed: %s", exc, exc_info=True)
            return self.build_error_response(
                reply="I encountered an error searching the corporate knowledge base. Please try again or contact support.",
                conversation_id=conversation_id,
                error=exc,
                bullets=["Corporate knowledge search failed", "Database or embedding service may be unavailable"],
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

        if self._is_preference_query(text) and sub_intent != "preference_recall":
            logger.info("[OperationalAgent] Preference-query guard remapped sub_intent to preference_recall")
            sub_intent = "preference_recall"

        if self._is_low_context_message(text) and sub_intent in {"history_lookup", "general"}:
            return self._build_clarification_response(conversation_id=conversation_id)

        if sub_intent == "history_lookup":
            # Extract requested count from text if present
            requested_count = self._MAX_HISTORY_QUESTIONS  # default
            import re
            count_match = re.search(r'\b(\d+)\b', text.lower())
            if count_match:
                try:
                    requested_count = int(count_match.group(1))
                    # Hard cap to protect token usage
                    requested_count = min(requested_count, self._MAX_HISTORY_QUESTIONS)
                    logger.info("[OperationalAgent] Detected requested conversation count: %s", requested_count)
                except ValueError:
                    pass
            return self._build_history_response(conversation_id=conversation_id, requested_count=requested_count)
        
        if sub_intent == "corporate_knowledge":
            # Handle IDBX/AIDANN corporate knowledge queries
            return await self._build_corporate_knowledge_response(
                text=text,
                conversation_id=conversation_id
            )
        if sub_intent == "profile_recall":
            # Use hybrid memory (recent + Vertex semantic) to extract preferences deterministically.
            history = runtime_context_service.get_hybrid_history(conversation_id, query_text=text)
            joined = "\n".join(
                self._clean_message_text(item.get("content"))
                for item in history
                if item.get("role") == "user"
            )
            # Extract common patterns the user states.
            desk = None
            risk_limit = None
            tenor = None
            m = re.search(r"\bdesk\s+is\s+([A-Za-z0-9_-]+)", joined, flags=re.IGNORECASE)
            if m:
                desk = m.group(1)
            m = re.search(r"\brisk\s+limit\s+is\s+([0-9.]+)\s*([A-Za-z]{3})", joined, flags=re.IGNORECASE)
            if m:
                risk_limit = f"{m.group(1)} {m.group(2).upper()}"
            m = re.search(r"\bdefault\s+tenor\s+is\s+([0-9]+[A-Za-z])\b", joined, flags=re.IGNORECASE)
            if m:
                tenor = m.group(1).upper()

            if desk or risk_limit or tenor:
                parts = []
                if desk:
                    parts.append(f"desk={desk}")
                if risk_limit:
                    parts.append(f"risk_limit={risk_limit}")
                if tenor:
                    parts.append(f"default_tenor={tenor}")
                return self.build_message_response(
                    reply=" ".join(parts),
                    bullets=[],
                    conversation_id=conversation_id,
                    model_info=self.get_model_info(model_override="profile-recall-fastpath"),
                )
            return self.build_message_response(
                reply="I couldn't find your saved desk/risk/tenor in this conversation yet.",
                bullets=["Ask: `Remember: My desk is ...` to store it."],
                conversation_id=conversation_id,
                model_info=self.get_model_info(model_override="profile-recall-fastpath"),
            )
        if sub_intent == "preference_recall":
            # Answer preference questions directly from hybrid memory, not history-summary templates.
            history = runtime_context_service.get_hybrid_history(conversation_id, query_text=text)
            joined = "\n".join(
                self._clean_message_text(item.get("content"))
                for item in history
                if item.get("role") == "user"
            ).lower()

            # Detect explicit "avoid/refuse overnight risk" style statements.
            avoid_overnight = any(
                phrase in joined
                for phrase in [
                    "avoid holding positions overnight",
                    "avoid holding position overnight",
                    "refuse overnight risk",
                    "no overnight risk",
                    "don't hold overnight",
                    "do not hold overnight",
                    "no overnight",
                ]
            )

            if avoid_overnight:
                return self.build_message_response(
                    reply="No — you said you avoid holding positions overnight / refuse overnight risk.",
                    bullets=[],
                    conversation_id=conversation_id,
                    model_info=self.get_model_info(model_override="preference-recall-fastpath"),
                )

            # If we don't have explicit preference stored yet, be honest.
            return self.build_message_response(
                reply="I don't see a clear saved preference about overnight risk yet.",
                bullets=["Try: `Store this: I avoid holding positions overnight. Reply only Saved.`"],
                conversation_id=conversation_id,
                model_info=self.get_model_info(model_override="preference-recall-fastpath"),
            )
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
