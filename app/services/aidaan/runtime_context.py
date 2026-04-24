"""
Runtime context and memory assembly for AIDAAN agents.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.core.config.settings import settings
from app.db.operational.service import operational_data_service


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _clean_text(value: Any) -> str:
    if not value:
        return ""
    return str(value).strip()


class RuntimeContextService:
    """
    Collects compact context slices for prompts.
    """

    def get_recent_history(self, conversation_id: Optional[str]) -> List[Dict[str, Any]]:
        return operational_data_service.get_recent_history(
            conversation_id,
            limit=settings.AIDAAN_HISTORY_WINDOW,
        )

    def get_operational_context(
        self,
        *,
        username: Optional[str],
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "user": None,
            "desk": None,
            "desk_membership": None,
            "desk_limits": [],
            "counterparties": [],
            "recent_rfq_drafts": [],
            "recent_tool_invocations": [],
            "recent_audit_events": [],
            "source_schema": None,
        }
        if username:
            payload.update(operational_data_service.get_operational_snapshot(username))
        if conversation_id:
            payload.update(operational_data_service.get_conversation_bundle(conversation_id, limit=5))
        return payload

    def build_continuity_guidance(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Distill the recent persisted history into a compact continuity block.
        """
        last_user_message = ""
        last_assistant_message = ""
        last_specialist_agent = None
        last_follow_up_prompt = ""
        pending_follow_up = False
        pending_follow_up_owner = None

        for item in history:
            role = item.get("role")
            content = _clean_text(item.get("content"))
            agent_name = item.get("agent_name")

            if role == "user" and content:
                last_user_message = content

            if role == "assistant" and content:
                last_assistant_message = content
                if agent_name in {"market", "risk", "order", "operational", "context"}:
                    last_specialist_agent = "operational" if agent_name == "context" else agent_name

        if last_assistant_message:
            assistant_is_history_style = (
                "[Context Detail]" in last_assistant_message
                and "[Optional Follow-up]" not in last_assistant_message
            )
            optional_follow_up_match = re.search(
                r"\[Optional Follow-up\]\s*(.*)",
                last_assistant_message,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if optional_follow_up_match:
                last_follow_up_prompt = _clean_text(optional_follow_up_match.group(1))
            elif "?" in last_assistant_message and not assistant_is_history_style:
                question_parts = re.findall(r"([^?]*\?)", last_assistant_message, flags=re.DOTALL)
                if question_parts:
                    last_follow_up_prompt = _clean_text(question_parts[-1])

        if last_follow_up_prompt:
            pending_follow_up = True
            pending_follow_up_owner = last_specialist_agent

        return {
            "last_user_message": last_user_message,
            "last_assistant_message": last_assistant_message,
            "last_specialist_agent": last_specialist_agent,
            "last_follow_up_prompt": last_follow_up_prompt,
            "pending_follow_up": pending_follow_up,
            "pending_follow_up_owner": pending_follow_up_owner,
        }

    def build_prompt_context(
        self,
        *,
        base_prompt: str,
        conversation_id: Optional[str],
        username: Optional[str],
    ) -> str:
        # Optimization: Skip expensive context building for extremely simple/short queries
        # unless it's the very first message.
        # Optimization: Only skip full context if the query is extremely short AND not history-related.
        # This ensures 'Last question?' or 'Yesterday's tasks' still get memory.
        lowered = base_prompt.lower()
        is_history_query = any(k in lowered for k in ["last", "previous", "earlier", "history", "yesterday", "remember", "context"])
        
        if len(base_prompt.strip().split()) <= 2 and not is_history_query:
            return base_prompt

        history = self.get_recent_history(conversation_id)
        operational = self.get_operational_context(
            username=username,
            conversation_id=conversation_id,
        )
        if not history and not any(operational.values()):
            return base_prompt

        history_block = _json_dump(history)
        ops_block = _json_dump(operational)
        continuity_block = _json_dump(self.build_continuity_guidance(history))
        return (
            f"{base_prompt}\n\n"
            "Server-side conversation memory below is authoritative and should be used for continuity.\n"
            f"Only rely on the last {settings.AIDAAN_HISTORY_WINDOW} persisted messages for short-term memory.\n"
            f"RECENT_HISTORY_JSON: {history_block}\n"
            f"OPERATIONAL_CONTEXT_JSON: {ops_block}\n"
            f"CONTINUITY_GUIDANCE_JSON: {continuity_block}\n"
            "If the current user turn is brief, elliptical, confirmatory, or otherwise ambiguous, interpret it against CONTINUITY_GUIDANCE_JSON before answering.\n"
            "If pending_follow_up=true, treat the current turn as a likely response to the prior assistant follow-up question unless the new message clearly starts a different topic.\n"
            "Do not answer by analyzing the literal token alone when recent context makes the intended continuation clear.\n"
            "Do not mention hidden database internals unless the user explicitly asks."
        )


runtime_context_service = RuntimeContextService()
