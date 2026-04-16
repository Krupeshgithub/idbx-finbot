"""
Repository helpers for operational persistence.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import (
    AuditEvent,
    Conversation,
    Counterparty,
    Desk,
    DeskLimit,
    DeskMembership,
    Message,
    RFQDraft,
    ToolInvocation,
    User,
)


def get_user_by_username(session: Session, username: str) -> Optional[User]:
    return session.execute(select(User).where(User.username == username)).scalar_one_or_none()


def get_primary_membership(session: Session, user_id: str) -> Optional[DeskMembership]:
    return session.execute(
        select(DeskMembership).where(
            DeskMembership.user_id == user_id,
            DeskMembership.is_primary.is_(True),
        )
    ).scalar_one_or_none()


def get_recent_messages(
    session: Session,
    *,
    conversation_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(desc(Message.created_at))
        .limit(limit)
    ).scalars().all()
    rows.reverse()
    return [
        {
            "role": row.role,
            "content": row.content,
            "agent_name": row.agent_name,
            "model_name": row.model_name,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


def get_recent_rfq_drafts(
    session: Session,
    *,
    conversation_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        select(RFQDraft)
        .where(RFQDraft.conversation_id == conversation_id)
        .order_by(desc(RFQDraft.created_at))
        .limit(limit)
    ).scalars().all()
    return [
        {
            "instrument": row.instrument,
            "notional": row.notional,
            "tenor": row.tenor,
            "settlement": row.settlement,
            "status": row.status,
            "requires_human_confirm": row.requires_human_confirm,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


def get_recent_tool_invocations(
    session: Session,
    *,
    conversation_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        select(ToolInvocation)
        .where(ToolInvocation.conversation_id == conversation_id)
        .order_by(desc(ToolInvocation.created_at))
        .limit(limit)
    ).scalars().all()
    return [
        {
            "tool_name": row.tool_name,
            "status": row.status,
            "arguments": row.arguments_json,
            "result": row.result_json,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


def get_recent_audit_events(
    session: Session,
    *,
    conversation_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        select(AuditEvent)
        .where(AuditEvent.conversation_id == conversation_id)
        .order_by(desc(AuditEvent.created_at))
        .limit(limit)
    ).scalars().all()
    return [
        {
            "event_type": row.event_type,
            "severity": row.severity,
            "summary": row.summary,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


def get_user_operational_bundle(session: Session, *, username: str) -> Dict[str, Any]:
    user = get_user_by_username(session, username)
    if user is None:
        return {
            "user": None,
            "desk": None,
            "desk_limits": [],
            "counterparties": [],
        }

    membership = get_primary_membership(session, user.id)
    desk = session.get(Desk, membership.desk_id) if membership else None
    desk_limits: List[DeskLimit] = []
    counterparties: List[Counterparty] = []
    if desk is not None:
        desk_limits = session.execute(
            select(DeskLimit).where(DeskLimit.desk_id == desk.id)
        ).scalars().all()
        counterparties = session.execute(
            select(Counterparty).where(Counterparty.desk_id == desk.id)
        ).scalars().all()

    return {
        "user": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "preferences": user.preferences,
        },
        "desk": {
            "desk_code": desk.desk_code,
            "name": desk.name,
            "location": desk.location,
            "asset_class": desk.asset_class,
            "timezone": desk.timezone,
            "language": desk.language,
            "default_context": desk.default_context,
        } if desk else None,
        "desk_limits": [
            {
                "instrument": row.instrument,
                "limit_type": row.limit_type,
                "soft_limit": row.soft_limit,
                "hard_limit": row.hard_limit,
                "currency": row.currency,
            }
            for row in desk_limits
        ],
        "counterparties": [
            {
                "code": row.code,
                "name": row.name,
                "region": row.region,
                "products": row.products,
            }
            for row in counterparties
        ],
    }


def ensure_conversation(
    session: Session,
    *,
    conversation_id: str,
    user_id: Optional[str],
    desk_id: Optional[str],
    channel: str,
    context: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            desk_id=desk_id,
            channel=channel,
            title=title,
            context_json=context or {},
        )
        session.add(conversation)
        session.flush()
        return conversation

    if user_id and not conversation.user_id:
        conversation.user_id = user_id
    if desk_id and not conversation.desk_id:
        conversation.desk_id = desk_id
    if context:
        merged = dict(conversation.context_json or {})
        merged.update(context)
        conversation.context_json = merged
    if title and not conversation.title:
        conversation.title = title
    conversation.updated_at = datetime.utcnow()
    session.flush()
    return conversation


def store_message(
    session: Session,
    *,
    conversation_id: str,
    role: str,
    content: str,
    agent_name: Optional[str] = None,
    model_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        agent_name=agent_name,
        model_name=model_name,
        metadata_json=metadata or {},
    )
    session.add(message)
    session.flush()
    return message


def store_audit_event(
    session: Session,
    *,
    conversation_id: Optional[str],
    event_type: str,
    summary: str,
    severity: str = "info",
    actor: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    event = AuditEvent(
        conversation_id=conversation_id,
        event_type=event_type,
        severity=severity,
        actor=actor,
        summary=summary,
        payload_json=payload or {},
    )
    session.add(event)
    session.flush()
    return event


def store_tool_invocation(
    session: Session,
    *,
    conversation_id: Optional[str],
    user_id: Optional[str],
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
    status: str = "completed",
) -> ToolInvocation:
    invocation = ToolInvocation(
        conversation_id=conversation_id,
        user_id=user_id,
        tool_name=tool_name,
        arguments_json=arguments,
        result_json=result,
        status=status,
    )
    session.add(invocation)
    session.flush()
    return invocation


def store_rfq_draft(
    session: Session,
    *,
    conversation_id: Optional[str],
    user_id: Optional[str],
    instrument: Optional[str],
    notional: Optional[float],
    tenor: Optional[str],
    settlement: Optional[str],
    payload: Dict[str, Any],
    status: str = "draft",
    requires_human_confirm: bool = True,
) -> RFQDraft:
    draft = RFQDraft(
        conversation_id=conversation_id,
        user_id=user_id,
        instrument=instrument,
        notional=notional,
        tenor=tenor,
        settlement=settlement,
        payload_json=payload,
        status=status,
        requires_human_confirm=requires_human_confirm,
        updated_at=datetime.utcnow(),
    )
    session.add(draft)
    session.flush()
    return draft
