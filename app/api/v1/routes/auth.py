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
    get_password_hash
)


router = APIRouter()


# Mock User Database for Phase 1 (Replace with Real DB in Phase 2)
MOCK_USERS = {
    "trader-001": {
        "username": "trader-001",
        "email": "trader-001@example.com",
        "hashed_password": get_password_hash("Password123")
    }
}


@router.post(
    "/login",
    response_model=Token,
)
async def login(payload: LoginRequest):
    """
    Standard institutional login endpoint.
    Verifies credentials and returns a JWT access token.
    """
    user = MOCK_USERS.get(payload.username)
    if not user or not verify_password(
        payload.password, 
        user["hashed_password"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
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
