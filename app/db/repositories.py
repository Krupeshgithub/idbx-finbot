"""
Repository helpers for operational persistence.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import MetaData, Table, desc, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.type_api import TypeEngine

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
)
from app.db.schema_config import schema_for_table


ANONYMOUS_IDENTITIES = {"anonymous", "anonymous-trader", "guest", "demo", "system", "trader", "user"}
USER_LOOKUP_COLUMNS = ("trader_id", "username", "user_name", "email", "id")
USER_FULL_NAME_COLUMNS = ("full_name", "name", "display_name")
USER_ROLE_COLUMNS = ("role", "user_role")
USER_PASSWORD_COLUMNS = ("hashed_password", "password_hash", "password")
USER_ACTIVE_COLUMNS = ("is_active", "active", "enabled", "status")
USER_PREFERENCES_COLUMNS = ("preferences", "preferences_json", "metadata", "meta")


@dataclass
class UserRecord:
    id: Any
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "trader"
    hashed_password: Optional[str] = None
    is_active: bool = True
    preferences: Dict[str, Any] | List[Any] | None = None


def _is_anonymous_identity(username: str | None) -> bool:
    if not username:
        return True
    return username.strip().lower() in ANONYMOUS_IDENTITIES


def _resolve_first_value(row: Any, columns: List[str], candidates: tuple[str, ...], default: Any = None) -> Any:
    for candidate in candidates:
        if candidate in columns:
            return row._mapping.get(candidate)
    return default


def normalize_username(username: str | None) -> str | None:
    if username is None:
        return None
    normalized = username.strip()
    if not normalized:
        return None
    if normalized.lower() in ANONYMOUS_IDENTITIES:
        return None
    return normalized


def _is_uuid_compatible(column_type: TypeEngine[Any]) -> bool:
    try:
        python_type = column_type.python_type
    except (AttributeError, NotImplementedError):
        python_type = None
    return python_type is uuid.UUID


def _coerce_lookup_value(raw_value: str, column_type: TypeEngine[Any]) -> Any:
    if _is_uuid_compatible(column_type):
        return uuid.UUID(raw_value)
    return raw_value


def _build_full_name(row: Any, columns: List[str]) -> Optional[str]:
    explicit_name = _resolve_first_value(row, columns, USER_FULL_NAME_COLUMNS)
    if explicit_name:
        return explicit_name

    first_name = row._mapping.get("first_name") if "first_name" in columns else None
    last_name = row._mapping.get("last_name") if "last_name" in columns else None
    combined = " ".join(part for part in [first_name, last_name] if part)
    return combined or None


def _coerce_active_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"active", "enabled", "true", "1"}
    if value is None:
        return True
    return bool(value)


def _load_users_table(session: Session) -> tuple[Table, List[str]]:
    bind = session.get_bind()
    metadata = MetaData()
    table = Table(
        "users",
        metadata,
        schema=schema_for_table("users"),
        autoload_with=bind,
    )
    columns = [column.name for column in table.columns]
    return table, columns


def _user_record_from_row(row: Any, columns: List[str]) -> UserRecord:
    user_id = row._mapping.get("id")
    lookup_value = _resolve_first_value(row, columns, USER_LOOKUP_COLUMNS)
    return UserRecord(
        id=user_id,
        username=str(lookup_value or user_id or ""),
        email=_resolve_first_value(row, columns, ("email",)),
        full_name=_build_full_name(row, columns),
        role=_resolve_first_value(row, columns, USER_ROLE_COLUMNS, "trader") or "trader",
        hashed_password=_resolve_first_value(row, columns, USER_PASSWORD_COLUMNS),
        is_active=_coerce_active_value(_resolve_first_value(row, columns, USER_ACTIVE_COLUMNS, True)),
        preferences=_resolve_first_value(row, columns, USER_PREFERENCES_COLUMNS, {}) or {},
    )


def get_user_by_username(session: Session, username: str) -> Optional[UserRecord]:
    normalized_username = normalize_username(username)
    if _is_anonymous_identity(normalized_username):
        return None

    users_table, columns = _load_users_table(session)

    for column_name in USER_LOOKUP_COLUMNS:
        if column_name not in columns:
            continue
        try:
            lookup_value = _coerce_lookup_value(normalized_username, users_table.c[column_name].type)
        except (ValueError, AttributeError):
            continue
        row = session.execute(
            select(users_table).where(users_table.c[column_name] == lookup_value)
        ).first()
        if row is not None:
            return _user_record_from_row(row, columns)

    return None


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
            "preferences": user.preferences or {},
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
