"""
Authentication Pydantic Schemas.
Standardized models for institutional login and identity.
"""
from typing import Optional
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    """
    Access token and metadata returned on successful login.
    """
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None


class TokenPayload(BaseModel):
    """
    Decoded internal payload of a JWT token.
    """
    sub: Optional[str] = None


class LoginRequest(BaseModel):
    """
    Credentials provided by the user for login.
    """
    username: str
    password: str


class UserContext(BaseModel):
    """
    Public user information used throughout the application.
    """
    username: str
    email: Optional[EmailStr] = None
    role: str = "trader"
    is_active: bool = True
