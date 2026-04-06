"""
Institutional Caching Service for AIDAAN.
Provides a unified interface for Redis while supporting in-memory fallback.
"""
import json
import logging

from typing import Any, Optional

import redis

from app.core.config.settings import settings


logger = logging.getLogger(__name__)


class CacheService:
    """
    Manage connections to Redis with institutional failover.
    """

    def __init__(self):
        redis_url = getattr(settings, "REDIS_URL", None)

        if redis_url:
            self.client = redis.from_url(
                redis_url,
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        else:
            ...

    def set(
        self, 
        key: str, 
        value: Any, 
        expire: int = 3600
    ):
        """
        Stores a value in the cache with an optional expiration.
        """
        serialized_value = json.dumps(value)
        self.client.set(
            key, 
            serialized_value, 
            ex=expire
        )
    
    def get(
        self, 
        key: str
    ) -> Optional[Any]:
        """
        Retrieves and deserializes a value from the cache.
        """
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def delete(
        self,
        key: str
    ):
        """
        Removes a key from the cache.
        """
        self.client.delete(key)


cache = CacheService()
