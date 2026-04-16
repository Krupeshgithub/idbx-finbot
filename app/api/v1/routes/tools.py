"""
AIDAAN Tool Invocation Routes.
Handles the formal execution of actions drafted by agents.
"""
from fastapi import APIRouter
from app.schemas.aidaan import (
    ToolInvokeRequest,
    ToolInvokeResponse
)
from app.services.persistence import persistence_service

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
    
    if payload.tool_name == "draft_rfq_ticket":
        response = ToolInvokeResponse(
            ok=True,
            result={
                "status": "ticket_prepared",
                "instrument": payload.arguments.get("instrument"),
                "notional": payload.arguments.get("notional"),
                "requires_human_confirm": True
            }
        )
        persistence_service.persist_tool_invocation(
            username=payload.user_id,
            conversation_id=payload.arguments.get("conversation_id"),
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            result=response.result or {},
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
        username=payload.user_id,
        conversation_id=payload.arguments.get("conversation_id"),
        tool_name=payload.tool_name,
        arguments=payload.arguments,
        result=response.result or {},
    )
    return response
