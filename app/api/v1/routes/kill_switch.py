"""
Kill switch routes for emergency venue-wide trading freeze.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.schemas.aidaan import KillSwitchStatusResponse, KillSwitchUpdateRequest
from app.schemas.auth import UserContext
from app.services.kill_switch import kill_switch_service

router = APIRouter()


def _require_kill_switch_operator(current_user: UserContext) -> None:
    if kill_switch_service.is_authorized_role(current_user.role):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Kill switch access is restricted to risk/admin operators.",
    )


@router.get("/status", response_model=KillSwitchStatusResponse)
async def get_kill_switch_status(
    current_user: UserContext = Depends(get_current_user),
) -> KillSwitchStatusResponse:
    _ = current_user
    return KillSwitchStatusResponse(**kill_switch_service.get_status())


@router.post("/activate", response_model=KillSwitchStatusResponse)
async def activate_kill_switch(
    payload: KillSwitchUpdateRequest,
    current_user: UserContext = Depends(get_current_user),
) -> KillSwitchStatusResponse:
    _ = payload
    _require_kill_switch_operator(current_user)
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Kill switch state is managed from .env. Update KILL_SWITCH_ACTIVE and restart the app.",
    )


@router.post("/deactivate", response_model=KillSwitchStatusResponse)
async def deactivate_kill_switch(
    payload: KillSwitchUpdateRequest,
    current_user: UserContext = Depends(get_current_user),
) -> KillSwitchStatusResponse:
    _ = payload
    _require_kill_switch_operator(current_user)
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Kill switch state is managed from .env. Update KILL_SWITCH_ACTIVE and restart the app.",
    )
