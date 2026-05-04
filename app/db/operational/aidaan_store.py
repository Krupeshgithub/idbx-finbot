"""
Read/write access layer for AIDAAN sidecar tables.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import desc, select, update
from sqlalchemy.orm import Session

from app.core.config.settings import settings
from app.db.operational.base import count_rows, fetch_all, insert_row
from app.db.models import KillSwitchEvent, RFQDraft


class AidaanStoreRepository:
    def __init__(self) -> None:
        self.schema = settings.DB_AIDAAN_SCHEMA

    def _new_id(self) -> str:
        return str(uuid4())

    def get_recent_messages(self, session: Session, conversation_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get recent messages for a conversation.
        Limit defaults to AIDAAN_HISTORY_WINDOW from settings (configurable).
        """
        if limit is None:
            limit = settings.AIDAAN_HISTORY_WINDOW
        rows = fetch_all(
            session,
            schema=self.schema,
            table_name="messages",
            filters={"conversation_id": conversation_id},
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )
        rows.reverse()
        return rows

    def get_recent_rfq_drafts(self, session: Session, conversation_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get recent RFQ drafts for a conversation.
        Limit defaults to AIDAAN_MAX_RFQ_DRAFTS from settings (configurable).
        """
        if limit is None:
            limit = settings.AIDAAN_MAX_RFQ_DRAFTS
        return fetch_all(
            session,
            schema=self.schema,
            table_name="rfq_drafts",
            filters={"conversation_id": conversation_id},
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )

    def get_recent_tool_invocations(self, session: Session, conversation_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get recent tool invocations for a conversation.
        Limit defaults to AIDAAN_MAX_TOOL_INVOCATIONS from settings (configurable).
        """
        if limit is None:
            limit = settings.AIDAAN_MAX_TOOL_INVOCATIONS
        return fetch_all(
            session,
            schema=self.schema,
            table_name="tool_invocations",
            filters={"conversation_id": conversation_id},
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )

    def get_recent_audit_events(self, session: Session, conversation_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get recent audit events for a conversation.
        Limit defaults to AIDAAN_MAX_AUDIT_EVENTS from settings (configurable).
        """
        if limit is None:
            limit = settings.AIDAAN_MAX_AUDIT_EVENTS
        return fetch_all(
            session,
            schema=self.schema,
            table_name="audit_events",
            filters={"conversation_id": conversation_id},
            order_by="created_at",
            order_desc=True,
            limit=limit,
        )

    def get_latest_kill_switch_event(
        self,
        session: Session,
        *,
        scope: str = "venue",
        venue: str = "global",
    ) -> Optional[Dict[str, Any]]:
        stmt = (
            select(KillSwitchEvent)
            .where(KillSwitchEvent.scope == scope, KillSwitchEvent.venue == venue)
            .order_by(desc(KillSwitchEvent.created_at))
            .limit(1)
        )
        event = session.execute(stmt).scalar_one_or_none()
        if event is None:
            return None
        return {
            "id": str(event.id),
            "scope": event.scope,
            "venue": event.venue,
            "is_active": event.is_active,
            "reason": event.reason,
            "actor": event.actor,
            "source": event.source,
            "conversation_id": event.conversation_id,
            "payload_json": event.payload_json,
            "created_at": event.created_at.isoformat(),
        }

    def ensure_conversation(
        self,
        session: Session,
        *,
        conversation_id: str,
        user_id: Optional[str],
        desk_id: Optional[str],
        channel: str,
        context: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = fetch_all(
            session,
            schema=self.schema,
            table_name="conversations",
            filters={"id": conversation_id},
            limit=1,
        )
        now = datetime.utcnow()
        if existing:
            return existing[0]

        return insert_row(
            session,
            schema=self.schema,
            table_name="conversations",
            values={
                "id": conversation_id,
                "user_id": user_id,
                "desk_id": desk_id,
                "channel": channel,
                "title": title,
                "status": "active",
                "context_json": context or {},
                "created_at": now,
                "updated_at": now,
            },
        )

    def store_message(
        self,
        session: Session,
        *,
        conversation_id: str,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a message and automatically generate its embedding vector using 
        the Cloud SQL Vertex AI integration.
        """
        from sqlalchemy import text
        
        # We use a raw insert to leverage the database-side embedding function
        # This is more efficient than calculating embeddings in Python.
        stmt = text(f"""
            INSERT INTO {self.schema}.messages 
            (id, conversation_id, role, content, agent_name, model_name, metadata_json, created_at, content_vector)
            VALUES 
            (:id, :conversation_id, :role, :content, :agent_name, :model_name, :metadata_json, :created_at, 
             aidaan.get_embedding(:content))
            RETURNING id, conversation_id, role, content, agent_name, model_name, metadata_json, created_at
        """)
        
        params = {
            "id": self._new_id(),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "agent_name": agent_name,
            "model_name": model_name,
            "metadata_json": metadata or {},
            "created_at": datetime.utcnow(),
        }
        
        result = session.execute(stmt, params).mappings().first()
        return dict(result) if result else {}

    def get_semantic_history(
        self, 
        session: Session, 
        conversation_id: str, 
        query_text: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for past messages that are semantically similar to the current query.
        Uses cosine similarity on the content_vector column.
        """
        from sqlalchemy import text
        
        # The <=> operator is for cosine distance in pgvector
        stmt = text(f"""
            SELECT id, role, content, agent_name, created_at,
                   (1 - (content_vector <=> aidaan.get_embedding(:query))) as similarity
            FROM {self.schema}.messages
            WHERE conversation_id = :conv_id
            AND content_vector IS NOT NULL
            ORDER BY content_vector <=> aidaan.get_embedding(:query)
            LIMIT :limit
        """)
        
        params = {
            "query": query_text,
            "conv_id": conversation_id,
            "limit": limit
        }
        
        return [dict(row) for row in session.execute(stmt, params).mappings().all()]

    def store_rfq_draft(
        self,
        session: Session,
        *,
        conversation_id: Optional[str],
        user_id: Optional[str],
        instrument: Optional[str],
        notional: Optional[float],
        tenor: Optional[str],
        settlement: Optional[str],
        payload: Dict[str, Any],
        status: str = "draft_prepared",
        requires_human_confirm: bool = True,
    ) -> Dict[str, Any]:
        now = datetime.utcnow()
        return insert_row(
            session,
            schema=self.schema,
            table_name="rfq_drafts",
            values={
                "id": self._new_id(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "status": status,
                "instrument": instrument,
                "notional": notional,
                "tenor": tenor,
                "settlement": settlement,
                "payload_json": payload,
                "requires_human_confirm": requires_human_confirm,
                "created_at": now,
                "updated_at": now,
            },
        )

    def store_tool_invocation(
        self,
        session: Session,
        *,
        conversation_id: Optional[str],
        user_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        status: str = "completed",
    ) -> Dict[str, Any]:
        return insert_row(
            session,
            schema=self.schema,
            table_name="tool_invocations",
            values={
                "id": self._new_id(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "tool_name": tool_name,
                "arguments_json": arguments,
                "result_json": result,
                "status": status,
                "created_at": datetime.utcnow(),
            },
        )

    def store_audit_event(
        self,
        session: Session,
        *,
        conversation_id: Optional[str],
        event_type: str,
        summary: str,
        severity: str = "info",
        actor: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return insert_row(
            session,
            schema=self.schema,
            table_name="audit_events",
            values={
                "id": self._new_id(),
                "conversation_id": conversation_id,
                "event_type": event_type,
                "severity": severity,
                "actor": actor,
                "summary": summary,
                "payload_json": payload or {},
                "created_at": datetime.utcnow(),
            },
        )

    def store_kill_switch_event(
        self,
        session: Session,
        *,
        is_active: bool,
        reason: str,
        actor: Optional[str],
        source: str = "api",
        scope: str = "venue",
        venue: str = "global",
        conversation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return insert_row(
            session,
            schema=self.schema,
            table_name="kill_switch_events",
            values={
                "id": self._new_id(),
                "scope": scope,
                "venue": venue,
                "is_active": is_active,
                "reason": reason,
                "actor": actor,
                "source": source,
                "conversation_id": conversation_id,
                "payload_json": payload or {},
                "created_at": datetime.utcnow(),
            },
        )

    def freeze_pending_rfq_drafts(
        self,
        session: Session,
        *,
        reason: str,
    ) -> int:
        stmt = (
            update(RFQDraft)
            .where(RFQDraft.status.in_(("draft", "draft_prepared")))
            .values(
                status="frozen_by_kill_switch",
                updated_at=datetime.utcnow(),
            )
        )
        result = session.execute(stmt)
        return int(result.rowcount or 0)

    def get_table_counts(self, session: Session) -> Dict[str, int]:
        return {
            table_name: count_rows(session, schema=self.schema, table_name=table_name)
            for table_name in [
                "desks",
                "desk_memberships",
                "desk_limits",
                "counterparties",
                "conversations",
                "messages",
                "rfq_drafts",
                "tool_invocations",
                "kill_switch_events",
                "audit_events",
            ]
        }
