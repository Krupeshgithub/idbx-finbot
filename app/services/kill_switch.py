"""
Venue-level kill switch service for blocking trade-affecting actions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.core.config.settings import settings
from app.db.operational.service import operational_data_service

logger = logging.getLogger(__name__)


class KillSwitchBlockedError(RuntimeError):
    def __init__(self, message: str, *, status: Dict[str, Any]) -> None:
        super().__init__(message)
        self.status = status


class KillSwitchService:
    _AUTHORIZED_ROLES = {
        "admin",
        "super_admin",
        "risk_manager",
        "risk_officer",
        "desk_head",
        "supervisor",
    }
    _BLOCKED_KEYWORDS = (
        "trade",
        "order",
        "rfq",
        "quote",
        "stage",
        "draft",
        "execute",
        "hedge",
        "cancel",
    )

    def get_status(self) -> Dict[str, Any]:
        active = bool(settings.ENABLE_KILL_SWITCH and settings.KILL_SWITCH_ACTIVE)
        reason = settings.KILL_SWITCH_REASON.strip() or "Kill switch state controlled from environment."
        return {
            "scope": "venue",
            "venue": "global",
            "is_active": active,
            "reason": reason if active else "Kill switch is currently off via environment configuration.",
            "actor": "env-config",
            "source": "env",
            "conversation_id": None,
            "last_updated_at": None,
            "frozen_rfq_count": 0,
        }

    def is_authorized_role(self, role: Optional[str]) -> bool:
        normalized = (role or "").strip().lower()
        return normalized in self._AUTHORIZED_ROLES

    def activate(
        self,
        *,
        actor: Optional[str],
        reason: str,
        source: str = "api",
        conversation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise RuntimeError("Kill switch state is env-managed. Update .env and restart the app.")

    def deactivate(
        self,
        *,
        actor: Optional[str],
        reason: str,
        source: str = "api",
        conversation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise RuntimeError("Kill switch state is env-managed. Update .env and restart the app.")

    def is_active(self) -> bool:
        if not settings.ENABLE_KILL_SWITCH:
            return False
        return bool(self.get_status().get("is_active"))

    def blocks_tool(self, tool_name: str) -> bool:
        normalized = (tool_name or "").strip().lower()
        if not normalized:
            return False
        return any(keyword in normalized for keyword in self._BLOCKED_KEYWORDS)

    def request_has_trade_intent(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(keyword in normalized for keyword in self._BLOCKED_KEYWORDS)

    def assert_tool_allowed(
        self,
        *,
        tool_name: str,
        actor: Optional[str],
        conversation_id: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not settings.ENABLE_KILL_SWITCH or not self.blocks_tool(tool_name):
            return

        status = self.get_status()
        if not status.get("is_active"):
            return

        logger.warning(
            "[KillSwitch] Blocked trade-affecting action | tool=%s actor=%s conversation_id=%s reason=%s arguments=%s",
            tool_name,
            actor,
            conversation_id,
            status.get("reason"),
            arguments or {},
        )
        operational_data_service.persist_tool_invocation(
            username=actor,
            conversation_id=conversation_id,
            tool_name=tool_name,
            arguments=arguments or {},
            result={
                "blocked": True,
                "reason": "venue_kill_switch_active",
                "kill_switch_status": status,
            },
            status="blocked",
        )
        raise KillSwitchBlockedError(
            "Venue kill switch is active. Trade-affecting actions are blocked.",
            status=status,
        )


kill_switch_service = KillSwitchService()
