"""
Guardrails API Endpoints

Provides API endpoints for kill switch control, audit log queries,
and guardrail configuration management.
"""
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.services.kill_switch import kill_switch_service, KillSwitchBlockedError
from app.db.operational.service import operational_data_service
from app.core.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/guardrails", tags=["guardrails"])


# --- Request/Response Models ---

class KillSwitchActivateRequest(BaseModel):
    """Request model for kill switch activation."""
    reason: str = Field(..., min_length=10, max_length=500, description="Reason for activation")
    scope: str = Field(default="venue", description="Scope of kill switch (venue, desk, user)")
    venue: str = Field(default="global", description="Venue identifier")


class KillSwitchDeactivateRequest(BaseModel):
    """Request model for kill switch deactivation."""
    reason: str = Field(..., min_length=10, max_length=500, description="Reason for deactivation")
    scope: str = Field(default="venue", description="Scope of kill switch (venue, desk, user)")
    venue: str = Field(default="global", description="Venue identifier")


class KillSwitchStatusResponse(BaseModel):
    """Response model for kill switch status."""
    scope: str
    venue: str
    is_active: bool
    reason: str
    actor: Optional[str]
    source: Optional[str]
    conversation_id: Optional[str]
    last_updated_at: Optional[datetime]
    frozen_rfq_count: int


class AuditEventResponse(BaseModel):
    """Response model for audit event."""
    id: str
    conversation_id: Optional[str]
    event_type: str
    severity: str
    actor: Optional[str]
    summary: str
    payload: dict
    created_at: datetime


class AuditEventsResponse(BaseModel):
    """Response model for audit events list."""
    events: List[AuditEventResponse]
    total: int
    limit: int
    offset: int


class GuardrailConfigResponse(BaseModel):
    """Response model for guardrail configuration."""
    kill_switch_enabled: bool
    advisory_detection_enabled: bool
    pii_redaction_enabled: bool
    strategic_data_redaction_enabled: bool
    advisory_confidence_threshold: float
    authorized_roles: List[str]


class GuardrailConfigUpdateRequest(BaseModel):
    """Request model for guardrail configuration update."""
    advisory_confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


# --- Kill Switch Endpoints ---

@router.post("/kill-switch/activate", response_model=KillSwitchStatusResponse)
async def activate_kill_switch(
    request: KillSwitchActivateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Activate venue-level kill switch.
    
    Requires authorized role: admin, super_admin, risk_manager, risk_officer, desk_head, supervisor
    """
    username = current_user.get("username")
    role = current_user.get("role", "").lower()
    
    try:
        status = kill_switch_service.activate(
            actor=username,
            reason=request.reason,
            role=role,
            source="api",
            scope=request.scope,
            venue=request.venue
        )
        
        return KillSwitchStatusResponse(**status)
    
    except RuntimeError as exc:
        logger.error(
            "[API] Kill switch activation failed | user=%s role=%s error=%s",
            username,
            role,
            exc
        )
        raise HTTPException(status_code=403, detail=str(exc))
    
    except Exception as exc:
        logger.error(
            "[API] Kill switch activation error | user=%s error=%s",
            username,
            exc
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/kill-switch/deactivate", response_model=KillSwitchStatusResponse)
async def deactivate_kill_switch(
    request: KillSwitchDeactivateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Deactivate venue-level kill switch.
    
    Requires authorized role: admin, super_admin, risk_manager, risk_officer, desk_head, supervisor
    """
    username = current_user.get("username")
    role = current_user.get("role", "").lower()
    
    try:
        status = kill_switch_service.deactivate(
            actor=username,
            reason=request.reason,
            role=role,
            source="api",
            scope=request.scope,
            venue=request.venue
        )
        
        return KillSwitchStatusResponse(**status)
    
    except RuntimeError as exc:
        logger.error(
            "[API] Kill switch deactivation failed | user=%s role=%s error=%s",
            username,
            role,
            exc
        )
        raise HTTPException(status_code=403, detail=str(exc))
    
    except Exception as exc:
        logger.error(
            "[API] Kill switch deactivation error | user=%s error=%s",
            username,
            exc
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/kill-switch/status", response_model=KillSwitchStatusResponse)
async def get_kill_switch_status(
    scope: str = Query(default="venue", description="Scope of kill switch"),
    venue: str = Query(default="global", description="Venue identifier"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get current kill switch status.
    
    Available to all authenticated users.
    """
    try:
        status = kill_switch_service.get_status(scope=scope, venue=venue)
        return KillSwitchStatusResponse(**status)
    
    except Exception as exc:
        logger.error(
            "[API] Kill switch status query error | user=%s error=%s",
            current_user.get("username"),
            exc
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Audit Log Endpoints ---

@router.get("/audit/events", response_model=AuditEventsResponse)
async def get_audit_events(
    conversation_id: Optional[str] = Query(None, description="Filter by conversation ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    actor: Optional[str] = Query(None, description="Filter by actor"),
    start_time: Optional[datetime] = Query(None, description="Filter by start timestamp"),
    end_time: Optional[datetime] = Query(None, description="Filter by end timestamp"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    current_user: dict = Depends(get_current_user)
):
    """
    Query audit events with filters.
    
    Available to all authenticated users (filtered by their access level).
    """
    try:
        # TODO: Implement audit event query with filters
        # For now, return empty result
        return AuditEventsResponse(
            events=[],
            total=0,
            limit=limit,
            offset=offset
        )
    
    except Exception as exc:
        logger.error(
            "[API] Audit events query error | user=%s error=%s",
            current_user.get("username"),
            exc
        )
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Configuration Endpoints ---

@router.get("/config", response_model=GuardrailConfigResponse)
async def get_guardrail_config(
    current_user: dict = Depends(get_current_user)
):
    """
    Get current guardrail configuration.
    
    Available to all authenticated users.
    """
    return GuardrailConfigResponse(
        kill_switch_enabled=settings.ENABLE_KILL_SWITCH,
        advisory_detection_enabled=settings.ENABLE_ADVISORY_DETECTION,
        pii_redaction_enabled=settings.ENABLE_PII_REDACTION,
        strategic_data_redaction_enabled=settings.ENABLE_STRATEGIC_DATA_REDACTION,
        advisory_confidence_threshold=settings.ADVISORY_CONFIDENCE_THRESHOLD,
        authorized_roles=list(kill_switch_service._AUTHORIZED_ROLES)
    )


@router.put("/config", response_model=GuardrailConfigResponse)
async def update_guardrail_config(
    request: GuardrailConfigUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update guardrail configuration.
    
    Requires admin or super_admin role.
    """
    username = current_user.get("username")
    role = current_user.get("role", "").lower()
    
    # Authorization check
    if role not in {"admin", "super_admin"}:
        logger.error(
            "[API] Unauthorized config update attempt | user=%s role=%s",
            username,
            role
        )
        raise HTTPException(
            status_code=403,
            detail="Unauthorized. Required roles: admin, super_admin"
        )
    
    # Update configuration
    if request.advisory_confidence_threshold is not None:
        settings.ADVISORY_CONFIDENCE_THRESHOLD = request.advisory_confidence_threshold
        logger.info(
            "[API] Advisory confidence threshold updated | user=%s value=%.2f",
            username,
            request.advisory_confidence_threshold
        )
    
    return GuardrailConfigResponse(
        kill_switch_enabled=settings.ENABLE_KILL_SWITCH,
        advisory_detection_enabled=settings.ENABLE_ADVISORY_DETECTION,
        pii_redaction_enabled=settings.ENABLE_PII_REDACTION,
        strategic_data_redaction_enabled=settings.ENABLE_STRATEGIC_DATA_REDACTION,
        advisory_confidence_threshold=settings.ADVISORY_CONFIDENCE_THRESHOLD,
        authorized_roles=list(kill_switch_service._AUTHORIZED_ROLES)
    )
