"""
Authentication routes for AIDANN.
"""
from fastapi import APIRouter
from app.utils.security import create_access_token


router = APIRouter()


@router.post("/login")
async def login():
    """
    This endpoint allows users to log in and receive an access token.
    """
    return {
        "access_token": create_access_token({"sub": "trader"}), 
        "token_type": "bearer"
    }
