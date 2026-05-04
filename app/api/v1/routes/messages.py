"""
REST conversation route for AIDAAN.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.db.operational.service import operational_data_service
from app.schemas.aidaan import (
    AidaanMessageRequest,
    AidaanMessageResponse,
    UserValidationRequest,
    UserValidationResponse,
)
from app.services.aidaan.agents.coordinator.coordinator_agent import coordinator_agent
from app.schemas.auth import UserContext
from app.api.deps import get_current_user
from app.services.persistence import persistence_service

router = APIRouter()


@router.post("/validate-user", response_model=UserValidationResponse)
async def validate_user(payload: UserValidationRequest) -> UserValidationResponse:
    user_session = operational_data_service.validate_user_session(payload.user_id)
    if not user_session["allowed"]:
        persistence_service.persist_login_event(payload.user_id, success=False)
        return UserValidationResponse(
            valid=False,
            reason=user_session["reason"],
            user_id=user_session.get("user_id"),
            source_schema=operational_data_service.public.schema,
            storage_schema=operational_data_service.aidaan.schema,
        )

    snapshot = operational_data_service.get_operational_snapshot(payload.user_id)
    user = snapshot.get("user") or {}
    desk = snapshot.get("desk") or {}
    persistence_service.persist_login_event(payload.user_id, success=True)
    return UserValidationResponse(
        valid=True,
        reason="ok",
        user_id=user_session["user_id"],
        trader_id=user.get("trader_id"),
        full_name=user.get("full_name"),
        risk_tier=user.get("risk_tier"),
        status=user.get("status"),
        desk=desk.get("name") or desk.get("desk_code"),
        desk_id=user_session["desk_id"],
        conversation_id=user_session["conversation_id"],
        source_schema=operational_data_service.public.schema,
        storage_schema=operational_data_service.aidaan.schema,
    )


@router.post("/aidaan/message", response_model=AidaanMessageResponse)
async def create_message(
    payload: AidaanMessageRequest,
    current_user: UserContext = Depends(get_current_user)
) -> AidaanMessageResponse:
    user_session = operational_data_service.validate_user_session(payload.user_id)
    if not user_session["allowed"]:
        persistence_service.persist_login_event(payload.user_id, success=False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=user_session["reason"],
        )

    context = dict(payload.context or {})
    context.update(
        {
            "username": payload.user_id,
            "canonical_user_id": user_session["user_id"],
            "desk_id": user_session["desk_id"],
            "frontend_conversation_id": payload.conversation_id,
        }
    )
    response = await coordinator_agent.handle_message(
        text=payload.text,
        conversation_id=user_session["conversation_id"],
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
