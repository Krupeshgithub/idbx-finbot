"""
V1 API Router for AIDAAN.
Harmonizes all Phase 1 endpoints under the /v1 prefix.
"""
from fastapi import APIRouter
from app.api.v1.routes import (
    auth,
    messages,
    tools,
    websocket
)

api_router = APIRouter()

# Authentication
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])

# Conversational REST
api_router.include_router(messages.router, tags=["aidaan"])

# Tool Invocations
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])

# Real-time WebSockets
api_router.include_router(websocket.router, prefix="/ws", tags=["ws"])
