"""
AIDAAN Tool Invocation Routes.
Handles the formal execution of actions drafted by agents.
"""
from fastapi import APIRouter

from app.db.repositories import ANONYMOUS_IDENTITIES
from app.core.llm_client import llm_client
from app.schemas.aidaan import (
    ToolInvokeRequest,
    ToolInvokeResponse
)
from app.services.persistence import persistence_service
from app.services.kill_switch import KillSwitchBlockedError, kill_switch_service

router = APIRouter()


@router.post(
    "/invoke",
    response_model=ToolInvokeResponse
)
async def invoke_tool(payload: ToolInvokeRequest):
    """
    Explicit tool invocation endpoint.
    Called only after a human trader clicks a "Draft" or "Confirm" button.
    This is the core of the "Zero Auto-Execution" rule.
    """
    # Hardcoded safety check: Log the intent and return success with draft details
    # In Phase 2, this will connect to the MCP executor.
    
    resolved_username = (
        None
        if not payload.user_id or payload.user_id.strip().lower() in ANONYMOUS_IDENTITIES
        else payload.user_id
    )

    try:
        kill_switch_service.assert_tool_allowed(
            tool_name=payload.tool_name,
            actor=resolved_username,
            conversation_id=payload.arguments.get("conversation_id"),
            arguments=payload.arguments,
        )
    except KillSwitchBlockedError as exc:
        return ToolInvokeResponse(
            ok=False,
            result={
                "blocked": True,
                "reason": "venue_kill_switch_active",
                "kill_switch_status": exc.status,
            },
            error=str(exc),
        )

    if payload.tool_name == "draft_rfq_ticket":
        mcp_result = await llm_client.call_mcp_tool(
            "draft_rfq_ticket",
            {
                **payload.arguments,
                "user_identity": resolved_username,
            },
        )
        response = ToolInvokeResponse(
            ok="error" not in (mcp_result or {}),
            result=mcp_result,
            error=(mcp_result or {}).get("error") if isinstance(mcp_result, dict) else None,
        )
        return response
    
    response = ToolInvokeResponse(
        ok=True,
        result={
            "message": f"Successfully invoked tool '{payload.tool_name}'. Phase 1 result: Simulation OK."
        },
        error=None
    )
    persistence_service.persist_tool_invocation(
        username=resolved_username,
        conversation_id=payload.arguments.get("conversation_id"),
        tool_name=payload.tool_name,
        arguments=payload.arguments,
        result=response.result or {},
    )
    return response
