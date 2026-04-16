"""
Operational data models for AIDAAN.
These tables are compatible with local Postgres and AlloyDB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return str(uuid4())


class Desk(Base):
    __tablename__ = "desks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    desk_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    location: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    asset_class: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    language: Mapped[str] = mapped_column(String(16), default="en")
    default_context: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    memberships: Mapped[List["DeskMembership"]] = relationship(back_populates="desk")
    limits: Mapped[List["DeskLimit"]] = relationship(back_populates="desk")
    counterparties: Mapped[List["Counterparty"]] = relationship(back_populates="desk")
    conversations: Mapped[List["Conversation"]] = relationship(back_populates="desk")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    role: Mapped[str] = mapped_column(String(64), default="trader")
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    preferences: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    memberships: Mapped[List["DeskMembership"]] = relationship(back_populates="user")
    conversations: Mapped[List["Conversation"]] = relationship(back_populates="user")
    rfq_drafts: Mapped[List["RFQDraft"]] = relationship(back_populates="user")


class DeskMembership(Base):
    __tablename__ = "desk_memberships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    desk_id: Mapped[str] = mapped_column(ForeignKey("desks.id"), index=True)
    title: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="memberships")
    desk: Mapped[Desk] = relationship(back_populates="memberships")


class DeskLimit(Base):
    __tablename__ = "desk_limits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    desk_id: Mapped[str] = mapped_column(ForeignKey("desks.id"), index=True)
    instrument: Mapped[str] = mapped_column(String(128))
    limit_type: Mapped[str] = mapped_column(String(64), default="notional")
    soft_limit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hard_limit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    desk: Mapped[Desk] = relationship(back_populates="limits")


class Counterparty(Base):
    __tablename__ = "counterparties"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    desk_id: Mapped[Optional[str]] = mapped_column(ForeignKey("desks.id"), nullable=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    region: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    products: Mapped[List[str]] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    desk: Mapped[Optional[Desk]] = relationship(back_populates="counterparties")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    desk_id: Mapped[Optional[str]] = mapped_column(ForeignKey("desks.id"), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(32), default="rest")
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    context_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[Optional[User]] = relationship(back_populates="conversations")
    desk: Mapped[Optional[Desk]] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(back_populates="conversation")
    rfq_drafts: Mapped[List["RFQDraft"]] = relationship(back_populates="conversation")
    audit_events: Mapped[List["AuditEvent"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    agent_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class RFQDraft(Base):
    __tablename__ = "rfq_drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[Optional[str]] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    instrument: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    notional: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tenor: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    settlement: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    requires_human_confirm: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped[Optional[Conversation]] = relationship(back_populates="rfq_drafts")
    user: Mapped[Optional[User]] = relationship(back_populates="rfq_drafts")


class ToolInvocation(Base):
    __tablename__ = "tool_invocations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    arguments_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[Optional[str]] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    severity: Mapped[str] = mapped_column(String(32), default="info")
    actor: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped[Optional[Conversation]] = relationship(back_populates="audit_events")


class MarketSnapshotCacheMetadata(Base):
    __tablename__ = "market_snapshot_cache_metadata"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(64), default="alpha_vantage")
    freshness_seconds: Mapped[int] = mapped_column(Integer, default=30)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

