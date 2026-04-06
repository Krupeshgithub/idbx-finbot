"""
Institutional Caching Service for AIDAAN.
Provides a unified interface for Redis while supporting in-memory fallback.
"""
import json
import logging

from typing import Any, Optional

import redis
import fakeredis

from app.core.config.settings import settings


logger = logging.getLogger(__name__)


class CacheService:
    """
    Manage connections to Redis with institutional failover.
    """

    def __init__(self):
        """
        Initialize the Redis client using the configured REDIS_URL.
        Falls back to fakeredis for local development context if needed.
        """
        self.redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")

        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Could not connect to Redis at {self.redis_url}: {e}")
            logger.info("Falling back to FakeStrictRedis for local development compatibility.")
            self.client = fakeredis.FakeStrictRedis(decode_responses=True)

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

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves and deserializes a value from the cache.
        """
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def delete(self, key: str):
        """
        Removes a key from the cache.
        """
        self.client.delete(key)


cache = CacheService()
