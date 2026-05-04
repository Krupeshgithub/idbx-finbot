"""
AIDAAN API Pydantic Schemas.
Standardized models for institutional messaging and tool orchestration.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """
    Represents a suggested action returned by AIDAAN
    """
    kind: str = Field(..., description="Type of action (e.g., draft_rfq, apply_filter)")
    title: str = Field(..., description="Human-readable title for the action button")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Data required to execute the action")
    requires_human_confirm: bool = Field(True, description="Hardcoded safety flag")


class ModelInfo(BaseModel):
    """
    Metadata about the agent/LLM that generated the response.
    """
    agent: str = Field(..., description="The name of the agent (e.g., coordinator)")
    llm: str = Field("gemini-2.5-flash", description="The underlying LLM")


class AidaanMessageRequest(BaseModel):
    """
    Request payload for AIDAAN conversational messages.
    """
    user_id: str
    text: str
    language: str = "en"
    conversation_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class AidaanMessageResponse(BaseModel):
    """
    Response payload for AIDAAN conversational messages.
    """
    reply: str
    bullets: List[str] = Field(default_factory=list)
    actions: List[ActionItem] = Field(default_factory=list)
    conversation_id: str
    guardrails: List[str] = Field(default_factory=lambda: ["zero_auto_execution"])
    model: ModelInfo
    latency_ms: float = 0.0


class UserValidationRequest(BaseModel):
    """
    Frontend user-id validation payload.
    """
    user_id: str


class UserValidationResponse(BaseModel):
    """
    Public user validation result for login/session gating.
    """
    valid: bool
    reason: str
    user_id: Optional[str] = None
    trader_id: Optional[str] = None
    full_name: Optional[str] = None
    risk_tier: Optional[str] = None
    status: Optional[str] = None
    desk: Optional[str] = None
    desk_id: Optional[str] = None
    conversation_id: Optional[str] = None
    source_schema: str = "public"
    storage_schema: str = "aidaan"


class ToolInvokeRequest(BaseModel):
    """
    Request payload for explicit tool execution.
    """
    user_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolInvokeResponse(BaseModel):
    """
    Response payload for tool execution.
    """
    ok: bool
    result: Optional[Any] = None
    error: Optional[str] = None


class KillSwitchStatusResponse(BaseModel):
    scope: str = "venue"
    venue: str = "global"
    is_active: bool
    reason: str
    actor: Optional[str] = None
    source: Optional[str] = None
    conversation_id: Optional[str] = None
    last_updated_at: Optional[str] = None
    frozen_rfq_count: int = 0


class KillSwitchUpdateRequest(BaseModel):
    reason: str = Field(..., min_length=3, description="Operator reason for toggling the venue kill switch")
    conversation_id: Optional[str] = None
    source: str = "api"
    payload: Dict[str, Any] = Field(default_factory=dict)
