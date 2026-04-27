"""
Higher-level persistence service for conversations, RFQs, and audit events.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.db.operational.service import operational_data_service


class PersistenceService:
    def resolve_user_context(self, username: Optional[str]) -> Dict[str, Optional[str]]:
        return operational_data_service.resolve_user_context(username)

    def persist_message_exchange(
        self,
        *,
        conversation_id: str,
        username: Optional[str],
        user_text: str,
        response_payload: Dict[str, Any],
        channel: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        operational_data_service.persist_message_exchange(
            conversation_id=conversation_id,
            username=username,
            user_text=user_text,
            response_payload=response_payload,
            channel=channel,
            context=context,
        )

    def persist_tool_invocation(
        self,
        *,
        username: Optional[str],
        conversation_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        operational_data_service.persist_tool_invocation(
            username=username,
            conversation_id=conversation_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )

    def persist_login_event(self, username: str, success: bool) -> None:
        operational_data_service.persist_login_event(username, success)


persistence_service = PersistenceService()
