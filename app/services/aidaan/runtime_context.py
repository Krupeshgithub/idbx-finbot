"""
Runtime context and memory assembly for AIDAAN agents.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import contextvars

logger = logging.getLogger(__name__)

# Context variables for Secure A2A session propagation (Zero Leakage)
current_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_conversation_id", default="")
current_username: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_username", default=None)

from app.core.config.settings import settings
from app.db.operational.service import operational_data_service

def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _clean_text(value: Any) -> str:
    if not value:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# In-process session-level context cache
# Avoids repeated DB hits for the same conversation within a short window.
# TTL: 5 minutes — long enough to cover multi-turn conversations,
# short enough to stay reasonably fresh.
# ---------------------------------------------------------------------------
_SESSION_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes (was 30 seconds)

class _SessionContextCache:
    """
    Lightweight in-memory TTL cache keyed by (conversation_id, username).
    Thread-safe for asyncio single-threaded event loop usage.
    """
    def __init__(self) -> None:
        # key → (payload, expiry_monotonic)
        self._store: Dict[str, Tuple[Any, float]] = {}

    def _key(self, conversation_id: Optional[str], username: Optional[str]) -> str:
        return f"{conversation_id or ''}::{username or ''}"

    def get(self, conversation_id: Optional[str], username: Optional[str]) -> Optional[Any]:
        k = self._key(conversation_id, username)
        entry = self._store.get(k)
        if entry is None:
            return None
        payload, expiry = entry
        if time.monotonic() > expiry:
            del self._store[k]
            return None
        return payload

    def set(self, conversation_id: Optional[str], username: Optional[str], payload: Any) -> None:
        k = self._key(conversation_id, username)
        self._store[k] = (payload, time.monotonic() + _SESSION_CACHE_TTL_SECONDS)

    def invalidate(self, conversation_id: Optional[str], username: Optional[str] = None) -> None:
        """Call this after a new message is persisted to force a fresh DB read."""
        k = self._key(conversation_id, username)
        self._store.pop(k, None)


_session_ctx_cache = _SessionContextCache()


class RuntimeContextService:
    """
    Collects compact context slices for prompts.
    """

    def get_hybrid_history(self, conversation_id: Optional[str], query_text: str) -> List[Dict[str, Any]]:
        return operational_data_service.get_hybrid_history(
            conversation_id,
            query_text=query_text,
            temporal_limit=settings.AIDAAN_HISTORY_WINDOW // 2,
            semantic_limit=settings.AIDAAN_HISTORY_WINDOW // 2,
        )

    def _truncate_payload(self, data: Any, max_len: int = 500) -> Any:
        """
        Recursively truncate large strings or lists in a JSON-like object.
        """
        if isinstance(data, str):
            if len(data) > max_len:
                return data[:max_len] + "... [TRUNCATED]"
            return data
        if isinstance(data, list):
            return [self._truncate_payload(item, max_len) for item in data[:10]]
        if isinstance(data, dict):
            return {k: self._truncate_payload(v, max_len) for k, v in data.items()}
        return data

    def get_operational_context(
        self,
        *,
        username: Optional[str],
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "user": None,
            "desk": None,
            "desk_membership": None,
            "desk_limits": [],
            "counterparties": [],
            "recent_rfq_drafts": [],
            "recent_tool_invocations": [],
            "recent_audit_events": [],
            "source_schema": None,
        }
        if username:
            payload.update(operational_data_service.get_operational_snapshot(username))
        if conversation_id:
            bundle = operational_data_service.get_conversation_bundle(conversation_id, limit=5)
            # REDACTION: Tool results and audit payloads can be huge and cause context leakage/hallucination.
            # We truncate them here to keep the prompt focused on intent and state, not raw data dumps.
            if "recent_tool_invocations" in bundle:
                bundle["recent_tool_invocations"] = [
                    {**item, "result_json": "[TRUNCATED for relevance]"}
                    for item in bundle["recent_tool_invocations"]
                ]
            if "recent_audit_events" in bundle:
                bundle["recent_audit_events"] = [
                    {**item, "payload_json": "[TRUNCATED for relevance]"}
                    for item in bundle["recent_audit_events"]
                ]
            payload.update(bundle)
        return payload

    def build_continuity_guidance(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Distill the recent persisted history into a compact continuity block.
        """
        last_user_message = ""
        last_assistant_message = ""
        last_specialist_agent = None
        last_follow_up_prompt = ""
        pending_follow_up = False
        pending_follow_up_owner = None

        for item in history:
            role = item.get("role")
            content = _clean_text(item.get("content"))
            agent_name = item.get("agent_name")

            if role == "user" and content:
                last_user_message = content

            if role == "assistant" and content:
                last_assistant_message = content
                if agent_name in {"market", "risk", "order", "operational", "context"}:
                    last_specialist_agent = "operational" if agent_name == "context" else agent_name

        if last_assistant_message:
            assistant_is_history_style = (
                "[Context Detail]" in last_assistant_message
                and "[Optional Follow-up]" not in last_assistant_message
            )
            optional_follow_up_match = re.search(
                r"\[Optional Follow-up\]\s*(.*)",
                last_assistant_message,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if optional_follow_up_match:
                last_follow_up_prompt = _clean_text(optional_follow_up_match.group(1))
            elif "?" in last_assistant_message and not assistant_is_history_style:
                question_parts = re.findall(r"([^?]*\?)", last_assistant_message, flags=re.DOTALL)
                if question_parts:
                    last_follow_up_prompt = _clean_text(question_parts[-1])

        if last_follow_up_prompt:
            pending_follow_up = True
            pending_follow_up_owner = last_specialist_agent

        return {
            "last_user_message": last_user_message,
            "last_assistant_message": last_assistant_message,
            "last_specialist_agent": last_specialist_agent,
            "last_follow_up_prompt": last_follow_up_prompt,
            "pending_follow_up": pending_follow_up,
            "pending_follow_up_owner": pending_follow_up_owner,
        }

    def build_prompt_context(
        self,
        *,
        base_prompt: str,
        conversation_id: Optional[str],
        username: Optional[str],
    ) -> str:
        # Optimization: Skip expensive context building for extremely simple/short queries
        # unless it's the very first message.
        # Optimization: Only skip full context if the query is extremely short AND not history-related.
        # This ensures 'Last question?' or 'Yesterday's tasks' still get memory.
        lowered = base_prompt.lower()
        is_history_query = any(k in lowered for k in ["last", "previous", "earlier", "history", "yesterday", "remember", "context"])
        
        if len(base_prompt.strip().split()) <= 2 and not is_history_query:
            return base_prompt

        # --- Session-level cache: avoid repeated DB hits within the same burst ---
        ctx_start = time.monotonic()
        cached_ctx = _session_ctx_cache.get(conversation_id, username)
        if cached_ctx is not None:
            ctx_elapsed = (time.monotonic() - ctx_start) * 1000
            logger.info(f"[TIMING] Context cache HIT | conv_id={conversation_id} | elapsed={ctx_elapsed:.1f}ms")
            history_block, ops_block, continuity_block = cached_ctx
        else:
            logger.info(f"[TIMING] Context cache MISS | conv_id={conversation_id} | fetching from DB (Hybrid Mode)")
            # Use Hybrid History to get both recent and semantically relevant messages
            history = self.get_hybrid_history(conversation_id, query_text=base_prompt)
            operational = self.get_operational_context(
                username=username,
                conversation_id=conversation_id,
            )
            if not history and not any(operational.values()):
                return base_prompt

            history_block = _json_dump(history)
            ops_block = _json_dump(operational)
            continuity_block = _json_dump(self.build_continuity_guidance(history))
            _session_ctx_cache.set(conversation_id, username, (history_block, ops_block, continuity_block))
            ctx_elapsed = (time.monotonic() - ctx_start) * 1000
            logger.info(f"[TIMING] Hybrid context built and cached | conv_id={conversation_id} | elapsed={ctx_elapsed:.1f}ms")

        return (
            f"{base_prompt}\n\n"
            "Server-side conversation memory below is authoritative and should be used for continuity.\n"
            "HISTORY_SOURCE: Cloud SQL + Vertex AI Hybrid Semantic Memory.\n"
            f"RECENT_HISTORY_JSON: {history_block}\n"
            f"OPERATIONAL_CONTEXT_JSON: {ops_block}\n"
            f"CONTINUITY_GUIDANCE_JSON: {continuity_block}\n"
            f"AI_GENERATED_USER_ANALYSIS: {operational.get('ai_user_summary', 'N/A')}\n"
            f"AI_GENERATED_DESK_ANALYSIS: {operational.get('ai_desk_summary', 'N/A')}\n"
            "If the current user turn is brief, elliptical, confirmatory, or otherwise ambiguous, interpret it against CONTINUITY_GUIDANCE_JSON and hybrid memory before answering.\n"
            "Messages marked with 'is_semantic_memory': true are retrieved from long-term memory because they are semantically relevant to the current user query.\n"
            "If pending_follow_up=true, treat the current turn as a likely response to the prior assistant follow-up question unless the new message clearly starts a different topic.\n"
            "Do not answer by analyzing the literal token alone when recent context makes the intended continuation clear.\n"
            "Do not mention hidden database internals unless the user explicitly asks."
        )

    def invalidate_session_cache(self, conversation_id: Optional[str], username: Optional[str] = None) -> None:
        """
        Invalidate the session context cache for a conversation.
        Call this after persisting a new message so the next turn gets fresh DB data.
        """
        _session_ctx_cache.invalidate(conversation_id, username)


runtime_context_service = RuntimeContextService()
