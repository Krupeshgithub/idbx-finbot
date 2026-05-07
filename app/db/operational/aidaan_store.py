"""
Read/write access layer for AIDAAN sidecar tables.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import desc, select, update
from sqlalchemy.orm import Session

from app.core.config.settings import settings
from app.db.operational.base import count_rows, fetch_all, insert_row
from app.db.models import KillSwitchEvent, RFQDraft

logger = logging.getLogger(__name__)


class AidaanStoreRepository:
    def __init__(self) -> None:
        self.schema = settings.DB_AIDAAN_SCHEMA

    def _new_id(self) -> str:
        return str(uuid4())

    @staticmethod
    def _embedding_function_available(session: Session) -> bool:
        from sqlalchemy import text

        try:
            exists_stmt = text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.routines
                    WHERE routine_schema = 'aidaan'
                      AND routine_name = 'get_embedding'
                ) AS available
                """
            )
            return bool(session.execute(exists_stmt).scalar())
        except Exception:
            return False

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
        
        Falls back to storing without embeddings if Vertex AI integration is not available.
        """
        from sqlalchemy import text
        import json
        
        message_id = self._new_id()
        created_at = datetime.utcnow()
        
        params = {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "agent_name": agent_name,
            "model_name": model_name,
            "metadata_json": json.dumps(metadata or {}),
            "created_at": created_at,
        }
        
        # Try to insert with embedding first (Vertex AI integration)
        try:
            stmt = text(f"""
                INSERT INTO {self.schema}.messages 
                (id, conversation_id, role, content, agent_name, model_name, metadata_json, created_at, content_vector)
                VALUES 
                (:id, :conversation_id, :role, :content, :agent_name, :model_name, :metadata_json, :created_at, 
                 aidaan.get_embedding(:content))
                RETURNING id, conversation_id, role, content, agent_name, model_name, metadata_json, created_at
            """)
            
            result = session.execute(stmt, params).mappings().first()
            logger.debug(f"[Vertex AI] Message stored with embedding | id={message_id}")
            return dict(result) if result else {}
            
        except Exception as e:
            # Fallback: Store without embedding if Vertex AI integration is not available
            logger.warning(
                f"[Vertex AI] Failed to generate embedding, storing without vector | "
                f"error={str(e)[:100]} | id={message_id}"
            )
            
            try:
                stmt_fallback = text(f"""
                    INSERT INTO {self.schema}.messages 
                    (id, conversation_id, role, content, agent_name, model_name, metadata_json, created_at)
                    VALUES 
                    (:id, :conversation_id, :role, :content, :agent_name, :model_name, :metadata_json, :created_at)
                    RETURNING id, conversation_id, role, content, agent_name, model_name, metadata_json, created_at
                """)
                
                result = session.execute(stmt_fallback, params).mappings().first()
                logger.info(f"[Vertex AI] Message stored without embedding (fallback) | id={message_id}")
                return dict(result) if result else {}
                
            except Exception as fallback_error:
                logger.error(
                    f"[Vertex AI] Failed to store message even without embedding | "
                    f"error={str(fallback_error)} | id={message_id}"
                )
                raise

    def get_semantic_history(
        self, 
        session: Session, 
        conversation_id: str, 
        query_text: str, 
        limit: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for past messages that are semantically similar to the current query.
        Uses cosine similarity on the content_vector column with configurable threshold.
        
        OPTIMIZATION: Filter by similarity threshold in SQL for better performance.
        Falls back to empty results if Vertex AI integration is not available.
        """
        from sqlalchemy import text
        import time
        
        # Use config defaults if not specified
        if limit is None:
            limit = settings.AIDAAN_SEMANTIC_LIMIT
        if similarity_threshold is None:
            similarity_threshold = settings.AIDAAN_SEMANTIC_THRESHOLD
        
        start_time = time.monotonic()
        
        try:
            # The <=> operator is for cosine distance in pgvector
            # Filter by similarity threshold in SQL for better performance
            stmt = text(f"""
                WITH query_embedding AS (
                    SELECT aidaan.get_embedding(:query) as query_vec
                ),
                ranked_messages AS (
                    SELECT 
                        id, role, content, agent_name, created_at,
                        (1 - (content_vector <=> query_embedding.query_vec)) as similarity
                    FROM {self.schema}.messages, query_embedding
                    WHERE conversation_id = :conv_id
                    AND content_vector IS NOT NULL
                    AND (1 - (content_vector <=> query_embedding.query_vec)) >= :threshold
                    ORDER BY content_vector <=> query_embedding.query_vec
                    LIMIT :limit
                )
                SELECT * FROM ranked_messages
                ORDER BY created_at ASC
            """)
            
            params = {
                "query": query_text,
                "conv_id": conversation_id,
                "limit": limit,
                "threshold": similarity_threshold
            }
            
            result = [dict(row) for row in session.execute(stmt, params).mappings().all()]
            elapsed_ms = (time.monotonic() - start_time) * 1000
            
            # Enhanced logging
            logger.info(
                f"[TEST] Semantic history: results={len(result)} | latency_ms={elapsed_ms:.1f} | "
                f"threshold={similarity_threshold} | limit={limit} | optimization=threshold_filtered"
            )
            
            return result
            
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                f"[Vertex AI] Semantic search failed, returning empty results | "
                f"error={str(e)[:100]} | latency_ms={elapsed_ms:.1f}"
            )
            return []

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

    def resolve_ticker_alias(self, session: Session, alias: str) -> Optional[str]:
        """
        Database-driven ticker resolution. No hardcoding.
        Resolves company names and aliases to ticker symbols.
        
        Args:
            session: Database session
            alias: Company name or alias (e.g., "google", "apple")
        
        Returns:
            Ticker symbol (e.g., "GOOGL", "AAPL") or None if not found
        """
        from sqlalchemy import text
        
        stmt = text(f"""
            SELECT ticker 
            FROM {self.schema}.ticker_aliases 
            WHERE LOWER(alias) = LOWER(:alias)
            AND is_active = TRUE
            ORDER BY priority ASC
            LIMIT 1
        """)
        
        result = session.execute(stmt, {"alias": alias}).scalar_one_or_none()
        
        if result:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[TickerResolver] '{alias}' -> {result} (from database)")
        
        return result

    def search_corporate_knowledge(
        self,
        session: Session,
        query: str,
        limit: int = 3,
        similarity_threshold: float = 0.6,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search IDBX corporate knowledge base using semantic similarity (PostgreSQL)
        or full-text search (SQLite).
        
        This enables AIDANN to answer questions about:
        - IDBX company information and mission
        - Leadership (e.g., "Who is the Chairman?")
        - AIDANN capabilities and technology
        - Security boundaries and execution policies
        - Data privacy and DLP features
        
        Args:
            session: Database session
            query: User's question or search query
            limit: Maximum number of results to return (default: 3)
            similarity_threshold: Minimum similarity score 0-1 (default: 0.6)
            category: Optional category filter (e.g., 'leadership', 'security_boundaries')
        
        Returns:
            List of knowledge entries with similarity scores, sorted by relevance
            
        Example:
            results = store.search_corporate_knowledge(
                session, 
                "Who is the Chairman?",
                limit=1
            )
            # Returns: [{"content": "The Chairman of IDBX is Nicholas J Runcorn...", "similarity": 0.95}]
        """
        from sqlalchemy import text
        from app.db.config import get_database_config
        import time
        
        start_time = time.monotonic()
        db_config = get_database_config()
        
        # Check if we're using SQLite or PostgreSQL
        is_sqlite = db_config.url.startswith("sqlite:")
        
        try:
            if is_sqlite:
                # SQLite fallback: Use full-text search instead of vector similarity
                logger.info("[Corporate Knowledge] Using SQLite full-text search (no vector embeddings)")
                
                category_filter = ""
                if category:
                    category_filter = "AND ck.category = :category"
                
                # Use FTS5 for full-text search; fallback to LIKE if FTS is unavailable.
                try:
                    stmt = text(f"""
                        SELECT 
                            ck.id, ck.category, ck.question, ck.content, ck.metadata_json,
                            fts.rank as similarity
                        FROM corporate_knowledge_fts fts
                        JOIN corporate_knowledge ck ON ck.id = fts.id
                        WHERE corporate_knowledge_fts MATCH :query
                        {category_filter}
                        ORDER BY fts.rank
                        LIMIT :limit
                    """)

                    params = {
                        "query": query,
                        "limit": limit
                    }
                    if category:
                        params["category"] = category

                    results = [dict(row) for row in session.execute(stmt, params).mappings().all()]
                except Exception:
                    fallback_stmt = text(f"""
                        SELECT
                            id, category, question, content, metadata_json,
                            CASE
                                WHEN LOWER(question) LIKE LOWER(:exact_question) THEN 0.95
                                WHEN LOWER(content) LIKE LOWER(:contains_query) THEN 0.85
                                ELSE 0.70
                            END AS similarity
                        FROM corporate_knowledge
                        WHERE (
                            LOWER(question) LIKE LOWER(:contains_query)
                            OR LOWER(content) LIKE LOWER(:contains_query)
                        )
                        {category_filter}
                        ORDER BY similarity DESC
                        LIMIT :limit
                    """)
                    params = {
                        "exact_question": query.strip(),
                        "contains_query": f"%{query.strip()}%",
                        "limit": limit,
                    }
                    if category:
                        params["category"] = category
                    results = [dict(row) for row in session.execute(fallback_stmt, params).mappings().all()]
                
                # Normalize rank to similarity score (0-1 range)
                # FTS5 rank is negative, lower is better
                if results:
                    min_rank = min(r['similarity'] for r in results)
                    max_rank = max(r['similarity'] for r in results)
                    rank_range = max_rank - min_rank if max_rank != min_rank else 1
                    
                    for r in results:
                        # Normalize to 0.7-0.95 range for consistency with vector search
                        normalized = 0.95 - ((r['similarity'] - min_rank) / rank_range * 0.25)
                        r['similarity'] = normalized
                
            else:
                category_filter = ""
                if category:
                    category_filter = "AND category = :category"

                if self._embedding_function_available(session):
                    # PostgreSQL + pgvector ready: use vector similarity.
                    logger.info("[Corporate Knowledge] Using PostgreSQL vector similarity search")
                    stmt = text(f"""
                        WITH query_embedding AS (
                            SELECT aidaan.get_embedding(:query) as query_vec
                        ),
                        ranked_knowledge AS (
                            SELECT 
                                id, category, question, content, metadata_json,
                                (1 - (embedding_vector <=> query_embedding.query_vec)) as similarity
                            FROM {self.schema}.corporate_knowledge, query_embedding
                            WHERE embedding_vector IS NOT NULL
                            {category_filter}
                            AND (1 - (embedding_vector <=> query_embedding.query_vec)) >= :threshold
                            ORDER BY embedding_vector <=> query_embedding.query_vec
                            LIMIT :limit
                        )
                        SELECT * FROM ranked_knowledge
                        ORDER BY similarity DESC
                    """)

                    params = {
                        "query": query,
                        "threshold": similarity_threshold,
                        "limit": limit
                    }
                    if category:
                        params["category"] = category
                    results = [dict(row) for row in session.execute(stmt, params).mappings().all()]
                else:
                    # Fallback for early deployments before get_embedding is installed.
                    logger.info("[Corporate Knowledge] Embedding function unavailable, using lexical fallback")
                    stmt = text(f"""
                        SELECT
                            id,
                            category,
                            question,
                            content,
                            metadata_json,
                            CASE
                                WHEN LOWER(question) LIKE LOWER(:exact_question) THEN 0.95
                                WHEN LOWER(content) LIKE LOWER(:contains_query) THEN 0.85
                                ELSE 0.70
                            END AS similarity
                        FROM {self.schema}.corporate_knowledge
                        WHERE (
                            LOWER(question) LIKE LOWER(:contains_query)
                            OR LOWER(content) LIKE LOWER(:contains_query)
                        )
                        {category_filter}
                        ORDER BY similarity DESC, updated_at DESC
                        LIMIT :limit
                    """)
                    params = {
                        "exact_question": query.strip(),
                        "contains_query": f"%{query.strip()}%",
                        "limit": limit,
                    }
                    if category:
                        params["category"] = category
                    results = [dict(row) for row in session.execute(stmt, params).mappings().all()]
            
            elapsed_ms = (time.monotonic() - start_time) * 1000
            
            logger.info(
                f"[Corporate Knowledge] Search completed | "
                f"backend={'sqlite' if is_sqlite else 'postgresql'} | "
                f"query='{query[:50]}...' | results={len(results)} | "
                f"latency_ms={elapsed_ms:.1f}"
            )
            
            return results
            
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                f"[Corporate Knowledge] Search failed | "
                f"backend={'sqlite' if is_sqlite else 'postgresql'} | "
                f"query='{query[:50]}...' | error={str(e)} | latency_ms={elapsed_ms:.1f}"
            )
            return []
