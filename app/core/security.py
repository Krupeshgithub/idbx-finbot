"""
Core Security Utilities for AIDAAN.
Handles JWT signing, verification, and password hashing for institutional auth.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from jose import jwt

from passlib.context import CryptContext


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# Configuration
SECRET_KEY = "secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 


def _truncate_password(password: str) -> str:
    """
    Bcrypt has a 72-byte limit. We truncate the password to ensure
    it stays within this limit before hashing or verification.
    """
    return password.encode("utf-8")[:72].decode("utf-8", errors="ignore")


def verify_password(
    plain_password: str, 
    hashed_password: str
) -> bool:
    """
    Verifies a plain password against its hashed version.
    """
    return pwd_context.verify(_truncate_password(plain_password), hashed_password)


def get_password_hash(
    password: str
) -> str:
    """
    Generates a bcrypt hash for the provided password.
    """
    return pwd_context.hash(_truncate_password(password))


def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Creates a JWT access token with the provided subject and expiration.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        SECRET_KEY, 
        algorithm=ALGORITHM
    )
    return encoded_jwt


def decode_access_token(
    token: str
) -> Optional[Dict[str, Any]]:
    """
    Decodes and validates a JWT token.
    """
    try:
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        return payload
    except Exception:
        return None
