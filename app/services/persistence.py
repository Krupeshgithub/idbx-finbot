"""
Higher-level persistence service for conversations, RFQs, and audit events.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.db.repositories import (
    ensure_conversation,
    get_primary_membership,
    get_user_by_username,
    store_audit_event,
    store_message,
    store_rfq_draft,
    store_tool_invocation,
)
from app.db.session import get_db_session

logger = logging.getLogger(__name__)


class PersistenceService:
    def resolve_user_context(self, username: Optional[str]) -> Dict[str, Optional[str]]:
        if not username:
            return {"user_id": None, "desk_id": None}

        with get_db_session() as session:
            user = get_user_by_username(session, username)
            if not user:
                return {"user_id": None, "desk_id": None}
            membership = get_primary_membership(session, user.id)
            return {
                "user_id": user.id,
                "desk_id": membership.desk_id if membership else None,
            }

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
        resolved = self.resolve_user_context(username)
        with get_db_session() as session:
            ensure_conversation(
                session,
                conversation_id=conversation_id,
                user_id=resolved["user_id"],
                desk_id=resolved["desk_id"],
                channel=channel,
                context=context,
                title=user_text[:120],
            )
            store_message(
                session,
                conversation_id=conversation_id,
                role="user",
                content=user_text,
                metadata={"channel": channel},
            )
            store_message(
                session,
                conversation_id=conversation_id,
                role="assistant",
                content=response_payload.get("reply", ""),
                agent_name=response_payload.get("model", {}).get("agent"),
                model_name=response_payload.get("model", {}).get("llm"),
                metadata={
                    "bullets": response_payload.get("bullets", []),
                    "actions": response_payload.get("actions", []),
                    "guardrails": response_payload.get("guardrails", []),
                },
            )
            store_audit_event(
                session,
                conversation_id=conversation_id,
                event_type="message_exchange",
                summary=f"{channel.upper()} message persisted for conversation {conversation_id}.",
                actor=username,
                payload={
                    "agent": response_payload.get("model", {}).get("agent"),
                    "channel": channel,
                },
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
        resolved = self.resolve_user_context(username)
        with get_db_session() as session:
            store_tool_invocation(
                session,
                conversation_id=conversation_id,
                user_id=resolved["user_id"],
                tool_name=tool_name,
                arguments=arguments,
                result=result,
            )
            if tool_name == "draft_rfq_ticket":
                store_rfq_draft(
                    session,
                    conversation_id=conversation_id,
                    user_id=resolved["user_id"],
                    instrument=arguments.get("instrument"),
                    notional=arguments.get("notional"),
                    tenor=arguments.get("tenor"),
                    settlement=arguments.get("settlement"),
                    payload=arguments,
                    status="draft_prepared",
                    requires_human_confirm=True,
                )
            store_audit_event(
                session,
                conversation_id=conversation_id,
                event_type="tool_invocation",
                summary=f"Tool '{tool_name}' executed.",
                actor=username,
                payload={"arguments": arguments, "result": result},
            )

    def persist_login_event(self, username: str, success: bool) -> None:
        with get_db_session() as session:
            store_audit_event(
                session,
                conversation_id=None,
                event_type="login_success" if success else "login_failure",
                summary=f"Authentication {'succeeded' if success else 'failed'} for {username}.",
                actor=username,
                severity="info" if success else "warning",
                payload={"username": username},
            )


persistence_service = PersistenceService()

