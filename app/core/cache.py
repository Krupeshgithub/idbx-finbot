"""
Institutional Caching Service for AIDAAN.
Optimized for high-concurrency Scaling with connection pooling.
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
    Manage connections to Redis with institutional failover and connection pooling.
    Highly efficient for scaling to millions of users.
    """

    def __init__(self):
        """
        Initialize the Redis client using a connection pool.
        """
        # Ensure we read the latest settings
        self.redis_url = settings.REDIS_URL
        logger.info(f"Attempting to initialize CacheService with URL: {self.redis_url}")
        
        try:
            # Use a ConnectionPool for high-concurrency scaling
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=100
            )
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis Connection Failed at {self.redis_url}: {e}")
            logger.info("Falling back to FakeStrictRedis for resilience.")
            self.client = fakeredis.FakeStrictRedis(decode_responses=True)

    def set(self, key: str, value: Any, expire: int = 3600):
        """
        Stores a value in the cache. Safe against connection drops.
        """
        try:
            serialized_value = json.dumps(value)
            self.client.set(key, serialized_value, ex=expire)
        except Exception as e:
            logger.error(f"Cache SET Error for key {key}: {e}")

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the cache. Safe against connection drops.
        """
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache GET Error for key {key}: {e}")
        return None
    
    def delete(self, key: str):
        """
        Removes a key from the cache. Safe against connection drops.
        """
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Cache DELETE Error for key {key}: {e}")


cache = CacheService()
