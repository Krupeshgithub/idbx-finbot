"""
High-level operational data service for agents, persistence, and MCP tools.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from app.db.operational.aidaan_store import AidaanStoreRepository
from app.db.operational.public_read import PublicReadRepository
from app.db.session import get_db_session


class OperationalDataService:
    def __init__(self) -> None:
        self.public = PublicReadRepository()
        self.aidaan = AidaanStoreRepository()

    @staticmethod
    def _normalize_instrument(value: Optional[str]) -> str:
        """
        Normalize instrument identifiers so desk-limit matching stays deterministic.
        """
        return (value or "").strip().upper().replace(" ", "")

    @classmethod
    def _instrument_aliases(cls, instrument: Optional[str]) -> set[str]:
        """
        Expand instrument aliases used by trading desks and stored limit rows.
        """
        normalized = cls._normalize_instrument(instrument)
        aliases = {normalized} if normalized else set()
        alias_map = {
            "EUR/USD": {"EURUSD"},
            "EURUSD": {"EUR/USD"},
            "GBP/USD": {"GBPUSD", "CABLE"},
            "GBPUSD": {"GBP/USD", "CABLE"},
            "CABLE": {"GBP/USD", "GBPUSD"},
            "UKGILTS": {"GILT", "UK GILTS"},
            "UK GILTS": {"UKGILTS", "GILT"},
            "SONIA": {"GBP-SONIA"},
            "SOFR": {"USD-SOFR"},
        }
        for alias in list(aliases):
            aliases.update(alias_map.get(alias, set()))
        return aliases

    @classmethod
    def select_relevant_limit(
        cls,
        desk_limits: Iterable[Dict[str, Any]],
        instrument: Optional[str],
        *,
        limit_type: str = "notional",
    ) -> Optional[Dict[str, Any]]:
        """
        Select the most relevant desk limit for an instrument from the latest rows.
        """
        instrument_aliases = cls._instrument_aliases(instrument)
        normalized_limit_type = (limit_type or "").strip().lower()
        preferred_defaults = {"DEFAULT", "ALL", "GLOBAL", "ANY"}

        exact_matches: list[Dict[str, Any]] = []
        default_matches: list[Dict[str, Any]] = []
        typed_rows: list[Dict[str, Any]] = []

        for row in desk_limits:
            row_limit_type = str(row.get("limit_type") or "notional").strip().lower()
            if normalized_limit_type and row_limit_type != normalized_limit_type:
                continue

            typed_rows.append(row)
            row_instrument = cls._normalize_instrument(row.get("instrument"))
            if row_instrument in instrument_aliases:
                exact_matches.append(row)
            elif row_instrument in preferred_defaults:
                default_matches.append(row)

        if exact_matches:
            return exact_matches[0]
        if default_matches:
            return default_matches[0]
        return typed_rows[0] if typed_rows else None

    @staticmethod
    def _build_limit_usage(limit_row: Optional[Dict[str, Any]], position_size: Optional[float]) -> Dict[str, Any]:
        """
        Calculate simple utilization percentages for a proposed notional.
        """
        if not limit_row or position_size in (None, 0):
            return {}

        usage: Dict[str, Any] = {"requested_notional": position_size}
        for threshold_name in ("soft_limit", "hard_limit"):
            threshold = limit_row.get(threshold_name)
            if threshold in (None, 0):
                continue
            usage[f"{threshold_name}_utilization_pct"] = round((float(position_size) / float(threshold)) * 100, 2)
        return usage

    def resolve_user_context(self, identity: Optional[str]) -> Dict[str, Optional[str]]:
        if not identity:
            return {"user_id": None, "desk_id": None}

        with get_db_session() as session:
            user = self.public.get_user_by_identity(session, identity)
            if not user:
                return {"user_id": None, "desk_id": None}

            membership = self.public.get_primary_membership(session, str(user["id"]))
            return {
                "user_id": str(user["id"]),
                "desk_id": str(membership["desk_id"]) if membership else None,
            }

    def get_operational_snapshot(self, identity: Optional[str]) -> Dict[str, Any]:
        if not identity:
            return {
                "user": None,
                "desk": None,
                "desk_membership": None,
                "desk_limits": [],
                "counterparties": [],
                "source_schema": self.public.schema,
            }

        with get_db_session() as session:
            snapshot = self.public.get_operational_snapshot(session, identity)
            if snapshot.get("user"):
                return snapshot

            return {
                "user": None,
                "desk": None,
                "desk_membership": None,
                "desk_limits": [],
                "counterparties": [],
                "source_schema": self.public.schema,
            }

    def get_recent_history(self, conversation_id: Optional[str], limit: int) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        with get_db_session() as session:
            return self.aidaan.get_recent_messages(session, conversation_id, limit=limit)

    def get_conversation_bundle(self, conversation_id: Optional[str], limit: int = 5) -> Dict[str, Any]:
        if not conversation_id:
            return {
                "recent_messages": [],
                "recent_rfq_drafts": [],
                "recent_tool_invocations": [],
                "recent_audit_events": [],
            }
        with get_db_session() as session:
            return {
                "recent_messages": self.aidaan.get_recent_messages(session, conversation_id, limit=limit),
                "recent_rfq_drafts": self.aidaan.get_recent_rfq_drafts(session, conversation_id, limit=limit),
                "recent_tool_invocations": self.aidaan.get_recent_tool_invocations(session, conversation_id, limit=limit),
                "recent_audit_events": self.aidaan.get_recent_audit_events(session, conversation_id, limit=limit),
            }

    def get_greeting_context(self, identity: Optional[str], *, instrument: str = "EUR/USD") -> Dict[str, Any]:
        """
        Return a compact desk-aware context block for personalized greetings.
        """
        snapshot = self.get_operational_snapshot(identity)
        selected_limit = self.select_relevant_limit(snapshot.get("desk_limits", []), instrument)
        return {
            "user": snapshot.get("user"),
            "desk": snapshot.get("desk"),
            "desk_membership": snapshot.get("desk_membership"),
            "selected_limit": selected_limit,
            "source_schema": snapshot.get("source_schema"),
        }

    def get_risk_context(
        self,
        identity: Optional[str],
        *,
        instrument: Optional[str],
        position_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Return desk/user risk context sourced from operational tables.
        """
        snapshot = self.get_operational_snapshot(identity)
        selected_limit = self.select_relevant_limit(snapshot.get("desk_limits", []), instrument)
        return {
            "user": snapshot.get("user"),
            "desk": snapshot.get("desk"),
            "desk_membership": snapshot.get("desk_membership"),
            "selected_limit": selected_limit,
            "limit_usage": self._build_limit_usage(selected_limit, position_size),
            "source_schema": snapshot.get("source_schema"),
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
            self.aidaan.ensure_conversation(
                session,
                conversation_id=conversation_id,
                user_id=resolved["user_id"],
                desk_id=resolved["desk_id"],
                channel=channel,
                context=context,
                title=user_text[:120],
            )
            self.aidaan.store_message(
                session,
                conversation_id=conversation_id,
                role="user",
                content=user_text,
                metadata={"channel": channel},
            )
            self.aidaan.store_message(
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
            self.aidaan.store_audit_event(
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
            self.aidaan.store_tool_invocation(
                session,
                conversation_id=conversation_id,
                user_id=resolved["user_id"],
                tool_name=tool_name,
                arguments=arguments,
                result=result,
            )
            if tool_name == "draft_rfq_ticket":
                self.aidaan.store_rfq_draft(
                    session,
                    conversation_id=conversation_id,
                    user_id=resolved["user_id"],
                    instrument=arguments.get("instrument"),
                    notional=arguments.get("notional"),
                    tenor=arguments.get("tenor"),
                    settlement=arguments.get("settlement"),
                    payload=arguments,
                )
            self.aidaan.store_audit_event(
                session,
                conversation_id=conversation_id,
                event_type="tool_invocation",
                summary=f"Tool '{tool_name}' executed.",
                actor=username,
                payload={"arguments": arguments, "result": result},
            )

    def persist_login_event(self, username: str, success: bool) -> None:
        with get_db_session() as session:
            self.aidaan.store_audit_event(
                session,
                conversation_id=None,
                event_type="login_success" if success else "login_failure",
                summary=f"Authentication {'succeeded' if success else 'failed'} for {username}.",
                actor=username,
                severity="info" if success else "warning",
                payload={"username": username},
            )


operational_data_service = OperationalDataService()
