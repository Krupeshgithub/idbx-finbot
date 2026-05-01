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

    def get_status(self, *, scope: str = "venue", venue: str = "global") -> Dict[str, Any]:
        """
        Get current kill switch status from database.
        
        Args:
            scope: Scope of kill switch (venue, desk, user)
            venue: Venue identifier (global, specific venue)
        
        Returns:
            Kill switch status dictionary
        """
        # Check if database-backed state is enabled
        if settings.ENABLE_KILL_SWITCH:
            try:
                # Get status from database
                return operational_data_service.get_kill_switch_status(
                    scope=scope,
                    venue=venue
                )
            except Exception as exc:
                logger.error(
                    "[KillSwitch] Failed to get status from database: %s",
                    exc
                )
                # Fallback to environment-based status
        
        # Fallback: environment-based status
        active = bool(settings.ENABLE_KILL_SWITCH and settings.KILL_SWITCH_ACTIVE)
        reason = settings.KILL_SWITCH_REASON.strip() or "Kill switch state controlled from environment."
        return {
            "scope": scope,
            "venue": venue,
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
        actor: str,
        reason: str,
        role: str,
        source: str = "api",
        scope: str = "venue",
        venue: str = "global",
        conversation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Activate kill switch with authorization check.
        
        Args:
            actor: Username activating the kill switch
            reason: Reason for activation
            role: User role for authorization
            source: Source of activation (api, manual, automated)
            scope: Scope of kill switch (venue, desk, user)
            venue: Venue identifier (global, specific venue)
            conversation_id: Optional conversation ID
            payload: Optional additional data
        
        Returns:
            Kill switch status dictionary
        
        Raises:
            RuntimeError: If actor role is not authorized
        """
        # Authorization check
        if not self.is_authorized_role(role):
            logger.error(
                "[KillSwitch] Unauthorized activation attempt | actor=%s role=%s",
                actor,
                role
            )
            raise RuntimeError(
                f"Unauthorized. Required roles: {self._AUTHORIZED_ROLES}"
            )
        
        # Record activation event
        event = operational_data_service.record_kill_switch_event(
            is_active=True,
            reason=reason,
            actor=actor,
            source=source,
            scope=scope,
            venue=venue,
            conversation_id=conversation_id,
            payload=payload
        )
        
        logger.critical(
            "[KillSwitch] ACTIVATED | actor=%s role=%s reason=%s frozen_rfq_count=%d",
            actor,
            role,
            reason,
            event.get("payload_json", {}).get("frozen_rfq_count", 0)
        )
        
        return self.get_status(scope=scope, venue=venue)

    def deactivate(
        self,
        *,
        actor: str,
        reason: str,
        role: str,
        source: str = "api",
        scope: str = "venue",
        venue: str = "global",
        conversation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deactivate kill switch with authorization check.
        
        Args:
            actor: Username deactivating the kill switch
            reason: Reason for deactivation
            role: User role for authorization
            source: Source of deactivation (api, manual, automated)
            scope: Scope of kill switch (venue, desk, user)
            venue: Venue identifier (global, specific venue)
            conversation_id: Optional conversation ID
            payload: Optional additional data
        
        Returns:
            Kill switch status dictionary
        
        Raises:
            RuntimeError: If actor role is not authorized
        """
        # Authorization check
        if not self.is_authorized_role(role):
            logger.error(
                "[KillSwitch] Unauthorized deactivation attempt | actor=%s role=%s",
                actor,
                role
            )
            raise RuntimeError(
                f"Unauthorized. Required roles: {self._AUTHORIZED_ROLES}"
            )
        
        # Record deactivation event
        operational_data_service.record_kill_switch_event(
            is_active=False,
            reason=reason,
            actor=actor,
            source=source,
            scope=scope,
            venue=venue,
            conversation_id=conversation_id,
            payload=payload
        )
        
        logger.warning(
            "[KillSwitch] DEACTIVATED | actor=%s role=%s reason=%s",
            actor,
            role,
            reason
        )
        
        return self.get_status(scope=scope, venue=venue)

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
