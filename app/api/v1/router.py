"""
V1 API Router for AIDAAN.
Harmonizes all Phase 1 endpoints under the /v1 prefix.
"""
from fastapi import APIRouter
from app.api.v1.routes import (
    aidaan,
    tools,
    websocket
)

api_router = APIRouter()

# Conversational & Messaging
api_router.include_router(aidaan.router, prefix="/aidaan", tags=["aidaan"])

# Tool Invocations
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])

# Real-time WebSockets
api_router.include_router(websocket.router, prefix="/ws", tags=["ws"])
