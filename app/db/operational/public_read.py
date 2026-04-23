"""
Read-only access layer for canonical public schema data.
"""
from __future__ import annotations

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
        filters = [users.c.trader_id == identity]
        try:
            filters.append(users.c.id == UUID(str(identity)))
        except (ValueError, TypeError):
            pass

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
            "source_schema": self.schema,
        }
