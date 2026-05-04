"""
Read-only access layer for canonical public schema data.
"""
from __future__ import annotations

import json
from uuid import UUID
from typing import Any, Dict, List, Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config.settings import settings
from app.db.operational.base import fetch_all, fetch_one, registry, row_to_dict


class PublicReadRepository:
    def __init__(self) -> None:
        self.schema = settings.DB_PUBLIC_SCHEMA

    def get_user_by_identity(self, session: Session, identity: str) -> Optional[Dict[str, Any]]:
        users = registry.table(session, schema=self.schema, name="users")
        normalized_identity = str(identity or "").strip()
        if not normalized_identity:
            return None

        filters = []
        for column_name in ("trader_id", "username", "user_name", "email"):
            if column_name in users.c:
                filters.append(users.c[column_name] == normalized_identity)

        if "id" in users.c:
            id_column = users.c.id
            try:
                id_python_type = id_column.type.python_type
            except (AttributeError, NotImplementedError):
                id_python_type = None

            try:
                filters.append(id_column == UUID(normalized_identity))
            except (ValueError, TypeError):
                if id_python_type is not UUID:
                    filters.append(id_column == normalized_identity)

        if not filters:
            return None

        query = select(users).where(or_(*filters))
        row = session.execute(query).first()
        return row_to_dict(row) if row else None

    def get_user_profile(self, session: Session, identity: str) -> Optional[Dict[str, Any]]:
        user = self.get_user_by_identity(session, identity)
        if not user:
            return None

        full_name = " ".join(
            part
            for part in [user.get("first_name"), user.get("last_name")]
            if part
        ) or None
        return {
            "id": user.get("id"),
            "trader_id": user.get("trader_id"),
            "full_name": full_name,
            "risk_tier": user.get("risk_tier"),
            "status": user.get("status"),
            "last_login": user.get("last_login"),
        }

    def get_primary_membership(self, session: Session, user_id: str) -> Optional[Dict[str, Any]]:
        return fetch_one(
            session,
            schema=self.schema,
            table_name="desk_memberships",
            filters={"user_id": user_id, "is_primary": True},
        )

    def get_desk(self, session: Session, desk_id: str) -> Optional[Dict[str, Any]]:
        return fetch_one(
            session,
            schema=self.schema,
            table_name="desks",
            filters={"id": desk_id},
        )

    def get_desk_limits(self, session: Session, desk_id: str) -> List[Dict[str, Any]]:
        return fetch_all(
            session,
            schema=self.schema,
            table_name="desk_limits",
            filters={"desk_id": desk_id},
            order_by="created_at",
            order_desc=True,
        )

    def get_counterparties(self, session: Session, desk_id: str) -> List[Dict[str, Any]]:
        return fetch_all(
            session,
            schema=self.schema,
            table_name="counterparties",
            filters={"desk_id": desk_id},
            order_by="created_at",
            order_desc=True,
        )

    def search_counterparties(
        self,
        session: Session,
        *,
        term: Optional[str] = None,
        desk_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        table = registry.table(session, schema=self.schema, name="counterparties")
        query = select(table)
        if desk_id:
            query = query.where(table.c.desk_id == desk_id)
        if term:
            pattern = f"%{term}%"
            query = query.where(
                or_(
                    table.c.code.ilike(pattern),
                    table.c.name.ilike(pattern),
                    table.c.region.ilike(pattern),
                )
            )
        query = query.order_by(table.c.created_at.desc()).limit(limit)
        return [row_to_dict(row) for row in session.execute(query).fetchall()]

    def semantic_search_counterparties(
        self,
        session: Session,
        *,
        query_text: str,
        desk_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a high-performance semantic search on counterparties.
        Joins the READ-ONLY public table with the AI shadow table in aidaan schema.
        """
        from sqlalchemy import text
        
        # We join public.counterparties with aidaan.counterparty_embeddings
        # This is very fast because of the HNSW index on the vector column.
        stmt_str = f"""
            SELECT cp.*,
                   (1 - (ce.embedding_vector <=> aidaan.get_embedding(:query))) as similarity
            FROM {self.schema}.counterparties cp
            JOIN aidaan.counterparty_embeddings ce ON cp.id = ce.counterparty_id
            WHERE 1=1
        """
        if desk_id:
            stmt_str += " AND cp.desk_id = :desk_id"
        
        stmt_str += " ORDER BY ce.embedding_vector <=> aidaan.get_embedding(:query) LIMIT :limit"
        
        params = {"query": query_text, "limit": limit}
        if desk_id:
            params["desk_id"] = desk_id
            
        return [dict(row) for row in session.execute(text(stmt_str), params).mappings().all()]

    def sync_counterparty_embedding(self, session: Session, counterparty_id: str, text_content: str) -> None:
        """
        Manually sync/index a public counterparty into the aidaan shadow table.
        Writes to aidaan schema ONLY.
        """
        from sqlalchemy import text
        stmt = text("""
            INSERT INTO aidaan.counterparty_embeddings (counterparty_id, embedding_vector)
            VALUES (:cp_id, aidaan.get_embedding(:text))
            ON CONFLICT (counterparty_id) DO UPDATE 
            SET embedding_vector = EXCLUDED.embedding_vector,
                last_updated_at = CURRENT_TIMESTAMP
        """)
        session.execute(stmt, {"cp_id": counterparty_id, "text": text_content})

    def generate_ai_user_summary(self, session: Session, user_id: str) -> str:
        """
        Highly Integrated AI: Uses Cloud SQL Vertex AI to reason about a user's 
        profile directly inside the database.
        """
        from sqlalchemy import text
        user = self.get_user_profile(session, user_id)
        if not user:
            return "User not found."

        # We call the LLM directly via SQL to summarize the user's data
        # This keeps our app layer thin and fast.
        stmt = text("""
            SELECT google_ml.predict_row(
                'gemini-1.5-pro:generateContent'::varchar, 
                json_build_object(
                    'contents', json_build_array(
                        json_build_object(
                            'role', 'user',
                            'parts', json_build_array(
                                json_build_object(
                                    'text', 'Summarize this trader profile in one professional sentence: ' || :user_json
                                )
                            )
                        )
                    )
                )::json
            )
        """)
        
        result = session.execute(stmt, {"user_json": json.dumps(user)}).scalar()
        try:
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No summary available.")
        except (AttributeError, IndexError):
            return "Unable to generate AI summary."

    def generate_ai_desk_summary(self, session: Session, desk_id: str) -> str:
        """
        AI Reasoning for Desk: Summarizes desk limits and activity directly in SQL.
        """
        from sqlalchemy import text
        limits = self.get_desk_limits(session, desk_id)
        
        stmt = text("""
            SELECT google_ml.predict_row(
                'gemini-1.5-pro:generateContent'::varchar,
                json_build_object(
                    'contents', json_build_array(
                        json_build_object(
                            'role', 'user',
                            'parts', json_build_array(
                                json_build_object(
                                    'text', 'Based on these desk limits, describe the risk capacity in one concise sentence: ' || :limits_json
                                )
                            )
                        )
                    )
                )::json
            )
        """)
        
        result = session.execute(stmt, {"limits_json": json.dumps(limits[:5])}).scalar()
        try:
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Desk capacity analysis unavailable.")
        except (AttributeError, IndexError):
            return "Unable to analyze desk capacity."

    def get_operational_snapshot(self, session: Session, identity: str) -> Dict[str, Any]:
        user = self.get_user_profile(session, identity)
        if not user:
            return {
                "user": None,
                "desk": None,
                "desk_membership": None,
                "desk_limits": [],
                "counterparties": [],
                "source_schema": self.schema,
            }

        membership = self.get_primary_membership(session, user["id"])
        desk = self.get_desk(session, membership["desk_id"]) if membership else None
        desk_limits = self.get_desk_limits(session, desk["id"]) if desk else []
        counterparties = self.get_counterparties(session, desk["id"]) if desk else []

        return {
            "user": user,
            "desk": desk,
            "desk_membership": membership,
            "desk_limits": desk_limits,
            "counterparties": counterparties,
            "ai_user_summary": self.generate_ai_user_summary(session, user["id"]) if user else None,
            "ai_desk_summary": self.generate_ai_desk_summary(session, desk["id"]) if desk else None,
            "source_schema": self.schema,
        }
