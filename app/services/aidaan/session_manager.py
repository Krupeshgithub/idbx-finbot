"""
Institutional Session Management for AIDAAN.
Handles conversation states and contexts using the Caching Service.
"""
from typing import Any, Dict
from app.core.cache import cache


class SessionManager:
    """
    Manages user sessions and conversation contexts.
    """
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieves context for a given conversation ID.
        """
        context = cache.get(f"session:{conversation_id}")
        return context if context else {}

    def save_context(self, conversation_id: str, context: Dict[str, Any]):
        """
        Saves or updates context for a given conversation ID.
        """
        # Expire session after 4 hours of inactivity (Institutional standard)
        cache.set(f"session:{conversation_id}", context, expire=14400)

    def update_context(self, conversation_id: str, updates: Dict[str, Any]):
        """
        Merges new data into the existing conversation context.
        """
        context = self.get_context(conversation_id)
        context.update(updates)
        self.save_context(conversation_id, context)


# Global session manager instance
session_manager = SessionManager()
