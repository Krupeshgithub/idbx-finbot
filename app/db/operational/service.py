"""
High-level operational data service for agents, persistence, and MCP tools.
"""
from __future__ import annotations

import logging
import time
import random
from uuid import UUID
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar
from functools import lru_cache

from sqlalchemy.exc import OperationalError

from app.db.operational.aidaan_store import AidaanStoreRepository
from app.db.operational.public_read import PublicReadRepository
from app.db.session import engine, get_db_session
from app.core.config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OperationalDataService:
    def __init__(self) -> None:
        self.public = PublicReadRepository()
        self.aidaan = AidaanStoreRepository()
        
        # Phase 2 Optimization: Query result caching (config-driven)
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._cache_ttl = settings.CONTEXT_CACHE_TTL_SECONDS  # From config
        self._cache_max_size = settings.CACHE_MAX_SIZE  # From config

    @staticmethod
    def _normalize_instrument(value: Optional[str]) -> str:
        """
        Normalize instrument identifiers so desk-limit matching stays deterministic.
        """
        return (value or "").strip().upper().replace(" ", "")

    @classmethod
    def _instrument_aliases(cls, instrument: Optional[str]) -> set[str]:
        """
        Expand instrument aliases used by trading desks and stored limit rows.
        """
        normalized = cls._normalize_instrument(instrument)
        aliases = {normalized} if normalized else set()
        alias_map = {
            "EUR/USD": {"EURUSD"},
            "EURUSD": {"EUR/USD"},
            "GBP/USD": {"GBPUSD", "CABLE"},
            "GBPUSD": {"GBP/USD", "CABLE"},
            "CABLE": {"GBP/USD", "GBPUSD"},
            "UKGILTS": {"GILT", "UK GILTS"},
            "UK GILTS": {"UKGILTS", "GILT"},
            "SONIA": {"GBP-SONIA"},
            "SOFR": {"USD-SOFR"},
        }
        for alias in list(aliases):
            aliases.update(alias_map.get(alias, set()))
        return aliases

    @classmethod
    def select_relevant_limit(
        cls,
        desk_limits: Iterable[Dict[str, Any]],
        instrument: Optional[str],
        *,
        limit_type: str = "notional",
    ) -> Optional[Dict[str, Any]]:
        """
        Select the most relevant desk limit for an instrument from the latest rows.
        """
        instrument_aliases = cls._instrument_aliases(instrument)
        normalized_limit_type = (limit_type or "").strip().lower()
        preferred_defaults = {"DEFAULT", "ALL", "GLOBAL", "ANY"}

        exact_matches: list[Dict[str, Any]] = []
        default_matches: list[Dict[str, Any]] = []
        typed_rows: list[Dict[str, Any]] = []

        for row in desk_limits:
            row_limit_type = str(row.get("limit_type") or "notional").strip().lower()
            if normalized_limit_type and row_limit_type != normalized_limit_type:
                continue

            typed_rows.append(row)
            row_instrument = cls._normalize_instrument(row.get("instrument"))
            if row_instrument in instrument_aliases:
                exact_matches.append(row)
            elif row_instrument in preferred_defaults:
                default_matches.append(row)

        if exact_matches:
            return exact_matches[0]
        if default_matches:
            return default_matches[0]
        return typed_rows[0] if typed_rows else None

    @staticmethod
    def _build_limit_usage(limit_row: Optional[Dict[str, Any]], position_size: Optional[float]) -> Dict[str, Any]:
        """
        Calculate simple utilization percentages for a proposed notional.
        """
        if not limit_row or position_size in (None, 0):
            return {}

        usage: Dict[str, Any] = {"requested_notional": position_size}
        for threshold_name in ("soft_limit", "hard_limit"):
            threshold = limit_row.get(threshold_name)
            if threshold in (None, 0):
                continue
            usage[f"{threshold_name}_utilization_pct"] = round((float(position_size) / float(threshold)) * 100, 2)
        return usage

    @staticmethod
    def _empty_conversation_bundle() -> Dict[str, Any]:
        return {
            "recent_messages": [],
            "recent_rfq_drafts": [],
            "recent_tool_invocations": [],
            "kill_switch_status": {
                "scope": "venue",
                "venue": "global",
                "is_active": False,
                "reason": "Kill switch status unavailable.",
                "actor": None,
                "source": None,
                "conversation_id": None,
                "last_updated_at": None,
                "frozen_rfq_count": 0,
            },
            "recent_audit_events": [],
        }

    @staticmethod
    def _transient_db_sleep(attempt: int) -> None:
        """
        Sleep with exponential backoff and jitter.
        """
        base_delay = 0.1
        max_delay = 2.0
        delay = min(max_delay, base_delay * (2**attempt))
        jitter = delay * 0.2 * random.random()
        time.sleep(delay + jitter)

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.
        Phase 2 Optimization: Query result caching.
        """
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            age = time.time() - timestamp
            if age < self._cache_ttl:
                logger.info(
                    "[Cache] HIT | key=%s | age=%.1fs | ttl=%ds",
                    cache_key,
                    age,
                    self._cache_ttl
                )
                return cached_data
            else:
                # Expired, remove it
                del self._cache[cache_key]
                logger.info(
                    "[Cache] EXPIRED | key=%s | age=%.1fs | ttl=%ds",
                    cache_key,
                    age,
                    self._cache_ttl
                )
        return None

    def _put_in_cache(self, cache_key: str, data: Any) -> None:
        """
        Store value in cache with timestamp.
        Phase 2 Optimization: Query result caching.
        """
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._cache_max_size:
            # Remove 20% oldest entries
            sorted_keys = sorted(self._cache.items(), key=lambda x: x[1][1])
            evict_count = max(1, self._cache_max_size // 5)
            for key, _ in sorted_keys[:evict_count]:
                del self._cache[key]
            logger.info(
                "[Cache] EVICTED | count=%d | reason=max_size_reached",
                evict_count
            )
        
        self._cache[cache_key] = (data, time.time())
        logger.info(
            "[Cache] STORED | key=%s | cache_size=%d/%d",
            cache_key,
            len(self._cache),
            self._cache_max_size
        )

    def _invalidate_cache(self, conversation_id: str) -> None:
        """
        Invalidate all cache entries for a conversation.
        Phase 2 Optimization: Smart cache invalidation.
        """
        keys_to_remove = [
            key for key in self._cache.keys()
            if conversation_id in key
        ]
        for key in keys_to_remove:
            del self._cache[key]
        
        if keys_to_remove:
            logger.info(
                "[Cache] INVALIDATED | conv_id=%s | keys_removed=%d",
                conversation_id,
                len(keys_to_remove)
            )

    def _execute_with_retry(
        self,
        operation: Callable[[], T],
        *,
        label: str,
        max_retries: int = 3,
        fallback_factory: Optional[Callable[[], T]] = None,
    ) -> T:
        """
        Execute a database operation with retries on OperationalError.
        Enhanced with slow query detection.
        """
        last_exc = None
        for attempt in range(max_retries):
            db_start = time.monotonic()
            try:
                result = operation()
                db_elapsed = (time.monotonic() - db_start) * 1000
                logger.info(f"[TIMING] DB operation | label={label} | elapsed={db_elapsed:.1f}ms | attempt={attempt+1}")
                
                # Warn on slow queries (configurable threshold)
                if db_elapsed > settings.SLOW_QUERY_THRESHOLD_MS:
                    logger.warning(
                        f"[PERF] SLOW QUERY | {label} | elapsed={db_elapsed:.1f}ms | "
                        f"threshold={settings.SLOW_QUERY_THRESHOLD_MS}ms"
                    )
                
                return result
            except OperationalError as exc:
                last_exc = exc
                if getattr(exc, "connection_invalidated", False):
                    engine.dispose()
                    logger.warning(
                        "[OperationalData] %s invalidated the DB connection pool after disconnect.",
                        label,
                    )
                if attempt < max_retries - 1:
                    logger.warning(
                        "[OperationalData] %s hit transient DB error; retrying (%d/%d) | error=%s",
                        label,
                        attempt + 1,
                        max_retries,
                        exc,
                    )
                    self._transient_db_sleep(attempt)
                else:
                    logger.error(
                        "[OperationalData] %s failed after %d attempts | error=%s",
                        label,
                        max_retries,
                        exc,
                    )

        if fallback_factory:
            return fallback_factory()
        raise last_exc or RuntimeError(f"Retry loop for {label} exited unexpectedly")

    def resolve_user_context(self, identity: Optional[str]) -> Dict[str, Optional[str]]:
        if not identity:
            return {"user_id": None, "desk_id": None}

        with get_db_session() as session:
            user = self.public.get_user_by_identity(session, identity)
            if not user:
                return {"user_id": None, "desk_id": None}

            membership = self.public.get_primary_membership(session, str(user["id"]))
            return {
                "user_id": str(user["id"]),
                "desk_id": str(membership["desk_id"]) if membership else None,
            }

    @staticmethod
    def _stable_conversation_id(user_id: str) -> str:
        """
        One durable chat thread per canonical public user id.
        """
        try:
            return f"conv-user-{UUID(str(user_id)).hex}"
        except (ValueError, TypeError):
            safe = "".join(ch for ch in str(user_id).lower() if ch.isalnum())
            return f"conv-user-{safe[:40]}"

    @staticmethod
    def _is_user_active(user: Dict[str, Any]) -> bool:
        status = user.get("status")
        if isinstance(status, str) and status.strip().lower() in {"disabled", "inactive", "blocked", "suspended"}:
            return False
        active_value = user.get("is_active", user.get("active", user.get("enabled", True)))
        if isinstance(active_value, str):
            return active_value.strip().lower() in {"active", "enabled", "true", "1", "yes"}
        return bool(active_value)

    def validate_user_session(self, identity: Optional[str]) -> Dict[str, Any]:
        """
        Validate a frontend identity against public.users and return the canonical
        aidaan trace identifiers. Does not write to the public schema.
        """
        raw_identity = str(identity or "").strip()
        if not raw_identity:
            return {
                "allowed": False,
                "reason": "user_id is required",
                "identity": raw_identity,
                "user_id": None,
                "desk_id": None,
                "conversation_id": None,
                "user": None,
            }

        with get_db_session() as session:
            user = self.public.get_user_by_identity(session, raw_identity)
            if not user:
                return {
                    "allowed": False,
                    "reason": "user_id not found in public.users",
                    "identity": raw_identity,
                    "user_id": None,
                    "desk_id": None,
                    "conversation_id": None,
                    "user": None,
                }
            if not self._is_user_active(user):
                return {
                    "allowed": False,
                    "reason": "user_id is inactive in public.users",
                    "identity": raw_identity,
                    "user_id": str(user.get("id")),
                    "desk_id": None,
                    "conversation_id": None,
                    "user": user,
                }

            membership = self.public.get_primary_membership(session, str(user["id"]))
            canonical_user_id = str(user["id"])
            return {
                "allowed": True,
                "reason": "ok",
                "identity": raw_identity,
                "user_id": canonical_user_id,
                "desk_id": str(membership["desk_id"]) if membership else None,
                "conversation_id": self._stable_conversation_id(canonical_user_id),
                "user": user,
            }

    def get_operational_snapshot(self, identity: Optional[str]) -> Dict[str, Any]:
        if not identity:
            return {
                "user": None,
                "desk": None,
                "desk_membership": None,
                "desk_limits": [],
                "counterparties": [],
                "source_schema": self.public.schema,
            }

        with get_db_session() as session:
            snapshot = self.public.get_operational_snapshot(session, identity)
            if snapshot.get("user"):
                return snapshot

            return {
                "user": None,
                "desk": None,
                "desk_membership": None,
                "desk_limits": [],
                "counterparties": [],
                "source_schema": self.public.schema,
            }

    def get_recent_history(self, conversation_id: Optional[str], limit: int) -> list[dict[str, Any]]:
        if not conversation_id:
            return []

        # Phase 2 Optimization: Check cache first
        cache_key = f"history:{conversation_id}:{limit}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        def _fetch():
            with get_db_session() as session:
                return self.aidaan.get_recent_messages(session, conversation_id, limit=limit)

        result = self._execute_with_retry(
            _fetch,
            label=f"Recent history lookup ({conversation_id})",
            fallback_factory=list,
        )
        
        # Store in cache
        self._put_in_cache(cache_key, result)
        return result

    def get_hybrid_history(
        self, 
        conversation_id: Optional[str], 
        query_text: str, 
        temporal_limit: int = 10, 
        semantic_limit: int = None,  # Use config default
        similarity_threshold: float = None  # Use config default
    ) -> list[dict[str, Any]]:
        """
        The 'Gold Standard' for history retrieval:
        Combines the latest messages (for continuity) with semantically relevant 
        past messages (for long-term memory), powered by Cloud SQL Vertex AI.
        
        OPTIMIZATION: Skip semantic search for very short queries (< 5 words)
        to avoid expensive embedding computation.
        """
        if not conversation_id:
            return []
        
        # Use config defaults if not specified
        if semantic_limit is None:
            semantic_limit = settings.AIDAAN_SEMANTIC_LIMIT
        if similarity_threshold is None:
            similarity_threshold = settings.AIDAAN_SEMANTIC_THRESHOLD

        def _fetch():
            with get_db_session() as session:
                # 1. Get recent messages for immediate context
                recent = self.aidaan.get_recent_messages(session, conversation_id, limit=temporal_limit)
                
                # OPTIMIZATION: Skip semantic search for short queries
                query_word_count = len(query_text.strip().split())
                logger.info(f"[TEST] Semantic search decision: word_count={query_word_count} | threshold=5 | skip={query_word_count < 5}")
                
                if query_word_count < 5:
                    # For short queries, just return recent messages
                    combined = list(recent)
                else:
                    # 2. Get semantic messages for deep memory (only for longer queries)
                    semantic = self.aidaan.get_semantic_history(
                        session, 
                        conversation_id, 
                        query_text, 
                        limit=semantic_limit,
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Merge and de-duplicate by ID
                    seen_ids = {msg["id"] for msg in recent}
                    combined = list(recent)
                    for msg in semantic:
                        if msg["id"] not in seen_ids:
                            msg["is_semantic_memory"] = True # Flag for the agent to know this is retrieved memory
                            msg["similarity_score"] = msg.get("similarity", 0)
                            combined.append(msg)
                
                # Sort by creation time to keep the conversation logical
                # Convert datetime to ISO string for consistent sorting
                def _sort_key(msg):
                    created_at = msg.get("created_at")
                    if created_at is None:
                        return ""
                    # If it's a datetime object, convert to ISO string
                    if hasattr(created_at, 'isoformat'):
                        return created_at.isoformat()
                    # If it's already a string, return as-is
                    return str(created_at)
                
                combined.sort(key=_sort_key)
                return combined

        return self._execute_with_retry(
            _fetch,
            label=f"Hybrid history lookup ({conversation_id})",
            fallback_factory=list,
        )

    def get_conversation_bundle(self, conversation_id: Optional[str], limit: int = None) -> Dict[str, Any]:
        """
        Get conversation bundle with recent messages, drafts, tool invocations, and audit events.
        Phase 2 Optimization: Added caching layer.
        
        Args:
            conversation_id: The conversation ID to fetch data for
            limit: Maximum messages to fetch (defaults to AIDAAN_HISTORY_WINDOW from settings)
        
        Returns:
            Dictionary containing recent conversation data, limited by settings
        """
        if not conversation_id:
            return self._empty_conversation_bundle()

        # Use settings default if limit not specified
        if limit is None:
            limit = settings.AIDAAN_HISTORY_WINDOW

        # Phase 2 Optimization: Check cache first
        cache_key = f"bundle:{conversation_id}:{limit}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        def _fetch():
            with get_db_session() as session:
                return {
                    "recent_messages": self.aidaan.get_recent_messages(session, conversation_id, limit=limit),
                    "recent_rfq_drafts": self.aidaan.get_recent_rfq_drafts(session, conversation_id),  # Uses AIDAAN_MAX_RFQ_DRAFTS
                    "recent_tool_invocations": self.aidaan.get_recent_tool_invocations(session, conversation_id),  # Uses AIDAAN_MAX_TOOL_INVOCATIONS
                    "kill_switch_status": self.get_kill_switch_status(),
                    "recent_audit_events": self.aidaan.get_recent_audit_events(session, conversation_id),  # Uses AIDAAN_MAX_AUDIT_EVENTS
                }

        result = self._execute_with_retry(
            _fetch,
            label=f"Conversation bundle lookup ({conversation_id})",
            fallback_factory=self._empty_conversation_bundle,
        )
        
        # Store in cache
        self._put_in_cache(cache_key, result)
        return result

    def get_kill_switch_status(self, *, scope: str = "venue", venue: str = "global") -> Dict[str, Any]:
        with get_db_session() as session:
            latest_event = self.aidaan.get_latest_kill_switch_event(
                session,
                scope=scope,
                venue=venue,
            )
            if latest_event is None:
                return {
                    "scope": scope,
                    "venue": venue,
                    "is_active": False,
                    "reason": "Kill switch has not been activated.",
                    "actor": None,
                    "source": None,
                    "conversation_id": None,
                    "last_updated_at": None,
                    "frozen_rfq_count": 0,
                }
            return {
                "scope": latest_event["scope"],
                "venue": latest_event["venue"],
                "is_active": bool(latest_event["is_active"]),
                "reason": latest_event["reason"],
                "actor": latest_event["actor"],
                "source": latest_event["source"],
                "conversation_id": latest_event["conversation_id"],
                "last_updated_at": latest_event["created_at"],
                "frozen_rfq_count": int((latest_event.get("payload_json") or {}).get("frozen_rfq_count") or 0),
            }

    def record_kill_switch_event(
        self,
        *,
        is_active: bool,
        reason: str,
        actor: Optional[str],
        source: str = "api",
        scope: str = "venue",
        venue: str = "global",
        conversation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with get_db_session() as session:
            payload_data = dict(payload or {})
            if is_active:
                frozen_count = self.aidaan.freeze_pending_rfq_drafts(
                    session,
                    reason=reason,
                )
                payload_data["frozen_rfq_count"] = frozen_count
            event = self.aidaan.store_kill_switch_event(
                session,
                is_active=is_active,
                reason=reason,
                actor=actor,
                source=source,
                scope=scope,
                venue=venue,
                conversation_id=conversation_id,
                payload=payload_data,
            )
            self.aidaan.store_audit_event(
                session,
                conversation_id=conversation_id,
                event_type="kill_switch_activated" if is_active else "kill_switch_released",
                severity="critical" if is_active else "warning",
                actor=actor,
                summary=(
                    f"Venue kill switch {'activated' if is_active else 'released'} "
                    f"for {venue}."
                ),
                payload={
                    "scope": scope,
                    "venue": venue,
                    "reason": reason,
                    "source": source,
                    **payload_data,
                },
            )
            return event

    def get_greeting_context(self, identity: Optional[str], *, instrument: str = "EUR/USD") -> Dict[str, Any]:
        """
        Return a compact desk-aware context block for personalized greetings.
        """
        snapshot = self.get_operational_snapshot(identity)
        selected_limit = self.select_relevant_limit(snapshot.get("desk_limits", []), instrument)
        return {
            "user": snapshot.get("user"),
            "desk": snapshot.get("desk"),
            "desk_membership": snapshot.get("desk_membership"),
            "selected_limit": selected_limit,
            "source_schema": snapshot.get("source_schema"),
        }

    def get_risk_context(
        self,
        identity: Optional[str],
        *,
        instrument: Optional[str],
        position_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Return desk/user risk context sourced from operational tables.
        """
        snapshot = self.get_operational_snapshot(identity)
        selected_limit = self.select_relevant_limit(snapshot.get("desk_limits", []), instrument)
        return {
            "user": snapshot.get("user"),
            "desk": snapshot.get("desk"),
            "desk_membership": snapshot.get("desk_membership"),
            "selected_limit": selected_limit,
            "limit_usage": self._build_limit_usage(selected_limit, position_size),
            "source_schema": snapshot.get("source_schema"),
        }

    def persist_message_exchange(
        self,
        *,
        conversation_id: str,
        username: Optional[str],
        user_text: str,
        response_payload: Dict[str, Any],
        channel: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _persist():
            resolved = self.resolve_user_context(username)
            with get_db_session() as session:
                self.aidaan.ensure_conversation(
                    session,
                    conversation_id=conversation_id,
                    user_id=resolved["user_id"],
                    desk_id=resolved["desk_id"],
                    channel=channel,
                    context=context,
                    title=user_text[:120],
                )
                self.aidaan.store_message(
                    session,
                    conversation_id=conversation_id,
                    role="user",
                    content=user_text,
                    metadata={"channel": channel},
                )
                self.aidaan.store_message(
                    session,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response_payload.get("reply", ""),
                    agent_name=response_payload.get("model", {}).get("agent"),
                    model_name=response_payload.get("model", {}).get("llm"),
                    metadata={
                        "bullets": response_payload.get("bullets", []),
                        "actions": response_payload.get("actions", []),
                        "guardrails": response_payload.get("guardrails", []),
                    },
                )
                self.aidaan.store_audit_event(
                    session,
                    conversation_id=conversation_id,
                    event_type="message_exchange",
                    summary=f"{channel.upper()} message persisted for conversation {conversation_id}.",
                    actor=username,
                    payload={
                        "agent": response_payload.get("model", {}).get("agent"),
                        "channel": channel,
                    },
                )

        self._execute_with_retry(_persist, label=f"Message persistence ({conversation_id})")
        
        # Phase 2 Optimization: Invalidate cache after new message
        self._invalidate_cache(conversation_id)

    def persist_tool_invocation(
        self,
        *,
        username: Optional[str],
        conversation_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        status: str = "completed",
    ) -> None:
        resolved = self.resolve_user_context(username)
        with get_db_session() as session:
            self.aidaan.store_tool_invocation(
                session,
                conversation_id=conversation_id,
                user_id=resolved["user_id"],
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                status=status,
            )
            if tool_name == "draft_rfq_ticket" and status == "completed":
                self.aidaan.store_rfq_draft(
                    session,
                    conversation_id=conversation_id,
                    user_id=resolved["user_id"],
                    instrument=arguments.get("instrument"),
                    notional=arguments.get("notional"),
                    tenor=arguments.get("tenor"),
                    settlement=arguments.get("settlement"),
                    payload=arguments,
                )
            self.aidaan.store_audit_event(
                session,
                conversation_id=conversation_id,
                event_type="tool_invocation",
                severity="warning" if status != "completed" else "info",
                summary=f"Tool '{tool_name}' executed with status '{status}'.",
                actor=username,
                payload={"arguments": arguments, "result": result, "status": status},
            )

    def persist_login_event(self, username: str, success: bool) -> None:
        with get_db_session() as session:
            self.aidaan.store_audit_event(
                session,
                conversation_id=None,
                event_type="login_success" if success else "login_failure",
                summary=f"Authentication {'succeeded' if success else 'failed'} for {username}.",
                actor=username,
                severity="info" if success else "warning",
                payload={"username": username},
            )


operational_data_service = OperationalDataService()
