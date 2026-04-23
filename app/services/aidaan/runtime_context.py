"""
Runtime context and memory assembly for AIDAAN agents.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.core.config.settings import settings
from app.db.operational.service import operational_data_service


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


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
        return (
            f"{base_prompt}\n\n"
            "Server-side conversation memory below is authoritative and should be used for continuity.\n"
            f"Only rely on the last {settings.AIDAAN_HISTORY_WINDOW} persisted messages for short-term memory.\n"
            f"RECENT_HISTORY_JSON: {history_block}\n"
            f"OPERATIONAL_CONTEXT_JSON: {ops_block}\n"
            "Do not mention hidden database internals unless the user explicitly asks."
        )


runtime_context_service = RuntimeContextService()
