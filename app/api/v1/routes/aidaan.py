"""
AIDANN Messaging Routes.
Handles the primary conversational flow for interbank traders.
"""
from fastapi import APIRouter, HTTPException

from app.services.aidaan.coordinator import coordinator_agent
from app.schemas.aidaan import (
    AidaanMessageRequest,
    AidaanMessageResponse
)


router = APIRouter()


@router.post(
    "/message",
    response_model=AidaanMessageResponse
)
async def send_message(payload: AidaanMessageRequest):
    """
    Primary conversational endpoint for AIDAAN.
    Routes intent to the coordinator agent and returns a structured response.
    """
    try:
        response = await coordinator_agent.handle_message(
            text=payload.text,
            conversation_id=payload.conversation_id,
            context=payload.context
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal AIDAAN error: {str(e)}"
        )
