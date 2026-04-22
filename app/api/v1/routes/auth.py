"""
Authentication Routes for AIDAAN.
Handles institutional login and identity verification
"""
from fastapi import (APIRouter, HTTPException, Depends, status)

from app.schemas.auth import Token, LoginRequest, UserContext
from app.api.deps import get_current_user
from app.core.security import (
    create_access_token,
    verify_password,
)
from app.db.repositories import get_user_by_username
from app.db.session import get_db_session
from app.services.persistence import persistence_service


router = APIRouter()


@router.post(
    "/login",
    response_model=Token,
)
async def login(payload: LoginRequest):
    """
    Standard institutional login endpoint.
    Verifies credentials and returns a JWT access token.
    """
    with get_db_session() as session:
        user = get_user_by_username(session, payload.username)
        if not user or not user.hashed_password or not verify_password(
            payload.password,
            user.hashed_password,
        ):
            persistence_service.persist_login_event(payload.username, success=False)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

    persistence_service.persist_login_event(payload.username, success=True)
    access_token = create_access_token(data={"sub": payload.username})
    return Token(
        access_token=access_token,
        token_type="bearer"
    )


@router.get("/me", response_model=UserContext)
async def read_users_me(
    current_user: UserContext = Depends(get_current_user)
):
    """
    Returns context for the currently authenticated user.
    """
    return current_user
