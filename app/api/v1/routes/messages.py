"""
REST conversation route for AIDAAN.
"""
from fastapi import APIRouter, Depends

from app.schemas.aidaan import AidaanMessageRequest, AidaanMessageResponse
from app.services.aidaan.agents.coordinator.coordinator_agent import coordinator_agent
from app.schemas.auth import UserContext
from app.api.deps import get_current_user
from app.services.persistence import persistence_service

router = APIRouter()


@router.post("/aidaan/message", response_model=AidaanMessageResponse)
async def create_message(
    payload: AidaanMessageRequest,
    current_user: UserContext = Depends(get_current_user)
) -> AidaanMessageResponse:
    context = dict(payload.context or {})
    context.setdefault("username", payload.user_id)
    response = await coordinator_agent.handle_message(
        text=payload.text,
        conversation_id=payload.conversation_id,
        context=context,
    )
    persistence_service.persist_message_exchange(
        conversation_id=response.conversation_id,
        username=payload.user_id,
        user_text=payload.text,
        response_payload=response.model_dump(),
        channel="rest",
        context=context,
    )
    return response
