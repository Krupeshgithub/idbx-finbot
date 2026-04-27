"""
Operational database MCP tools for public read-only access and aidaan sidecar writes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.db.operational.base import count_rows
from app.db.operational.service import operational_data_service
from app.db.session import get_db_session
from app.services.aidaan.mcp.shared import mcp


@mcp.tool()
async def get_user_profile(identity: str) -> Dict[str, Any]:
    """
    Resolve a trader profile from public.users using id or trader_id.
    """
    snapshot = operational_data_service.get_operational_snapshot(identity)
    return {
        "user": snapshot.get("user"),
        "source_schema": snapshot.get("source_schema"),
    }


@mcp.tool()
async def get_desk_context(identity: str) -> Dict[str, Any]:
    """
    Return desk membership, limits, and counterparties for a trader identity.
    """
    return operational_data_service.get_operational_snapshot(identity)


@mcp.tool()
async def get_conversation_history(conversation_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    Return recent aidaan-side conversation state for a conversation id.
    """
    bundle = operational_data_service.get_conversation_bundle(conversation_id, limit=limit)
    return {
        "conversation_id": conversation_id,
        **bundle,
        "source_schema": "aidaan",
    }


@mcp.tool()
async def search_counterparties(term: str = "", desk_id: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Search canonical public counterparties by name, code, or region.
    """
    with get_db_session() as session:
        rows = operational_data_service.public.search_counterparties(
            session,
            term=term or None,
            desk_id=desk_id,
            limit=limit,
        )
    return {
        "rows": rows,
        "count": len(rows),
        "source_schema": operational_data_service.public.schema,
    }


@mcp.tool()
async def get_operational_table_counts() -> Dict[str, Any]:
    """
    Lightweight health snapshot showing public and aidaan table population.
    """
    with get_db_session() as session:
        return {
            "public": {
                "users": count_rows(session, schema=operational_data_service.public.schema, table_name="users"),
                "desks": count_rows(session, schema=operational_data_service.public.schema, table_name="desks"),
                "desk_memberships": count_rows(session, schema=operational_data_service.public.schema, table_name="desk_memberships"),
                "desk_limits": count_rows(session, schema=operational_data_service.public.schema, table_name="desk_limits"),
                "counterparties": count_rows(session, schema=operational_data_service.public.schema, table_name="counterparties"),
            },
            "aidaan": operational_data_service.aidaan.get_table_counts(session),
        }


@mcp.tool()
async def draft_rfq_ticket(
    conversation_id: str,
    instrument: str,
    notional: Optional[float] = None,
    tenor: Optional[str] = None,
    settlement: Optional[str] = None,
    user_identity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a draft RFQ record in the aidaan sidecar schema.
    """
    resolved = operational_data_service.resolve_user_context(user_identity)
    arguments = {
        "conversation_id": conversation_id,
        "instrument": instrument,
        "notional": notional,
        "tenor": tenor,
        "settlement": settlement,
        "user_identity": user_identity,
    }
    result = {
        "status": "ticket_prepared",
        "instrument": instrument,
        "notional": notional,
        "tenor": tenor,
        "settlement": settlement,
        "requires_human_confirm": True,
        "storage_schema": "aidaan",
    }
    with get_db_session() as session:
        operational_data_service.aidaan.store_tool_invocation(
            session,
            conversation_id=conversation_id,
            user_id=resolved["user_id"],
            tool_name="draft_rfq_ticket",
            arguments=arguments,
            result=result,
        )
        operational_data_service.aidaan.store_rfq_draft(
            session,
            conversation_id=conversation_id,
            user_id=resolved["user_id"],
            instrument=instrument,
            notional=notional,
            tenor=tenor,
            settlement=settlement,
            payload=arguments,
        )
        operational_data_service.aidaan.store_audit_event(
            session,
            conversation_id=conversation_id,
            event_type="tool_invocation",
            summary="RFQ draft ticket prepared via MCP tool.",
            actor=user_identity,
            payload={"arguments": arguments, "result": result},
        )
    return result
