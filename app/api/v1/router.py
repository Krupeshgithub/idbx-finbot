"""
V1 API Router for AIDAAN.
Harmonizes all Phase 1 endpoints under the /v1 prefix.
"""
from fastapi import APIRouter
from app.api.v1.routes import (
    auth,
    diagnostics,
    guardrails,
    kill_switch,
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

# Guardrails (institutional compliance)
api_router.include_router(guardrails.router, tags=["guardrails"])

# Kill switch control plane (legacy, kept for backward compatibility)
api_router.include_router(kill_switch.router, prefix="/kill-switch", tags=["kill-switch"])

# Diagnostics (safe introspection)
api_router.include_router(diagnostics.router, tags=["diagnostics"])

# Real-time WebSockets
api_router.include_router(websocket.router, prefix="/ws", tags=["ws"])
