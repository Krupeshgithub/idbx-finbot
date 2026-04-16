"""
Authentication Dependencies for AIDAAN.
Provides a standard way to retrieve the current user from the JWT token.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.security import decode_access_token
from app.db.repositories import get_user_by_username
from app.db.session import get_db_session
from app.schemas.auth import UserContext


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> UserContext:
    """
    Dependency that extracts the user from the JWT token.
    Raises 401 if the token is invalid or the user doesn't exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
        
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
        
    with get_db_session() as session:
        user = get_user_by_username(session, username)
        if user is None:
            raise credentials_exception
        return UserContext(
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
        )
