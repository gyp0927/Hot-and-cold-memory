"""Redis cache implementation."""

import json
from typing import Any

import redis.asyncio as redis

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.exceptions import CacheError

from .base import BaseCache


class RedisCache(BaseCache):
    """Redis-based cache."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: redis.Redis | None = None

    async def initialize(self) -> None:
        """Connect to Redis."""
        if not self.settings.CACHE_URL:
            raise CacheError("CACHE_URL not configured")

        self.client = redis.from_url(
            self.settings.CACHE_URL,
            decode_responses=True,
        )

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            value = await self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except redis.RedisError as e:
            raise CacheError(f"Redis get failed: {e}") from e

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            ttl = ttl or self.settings.CACHE_TTL_SECONDS
            await self.client.setex(
                key,
                ttl,
                json.dumps(value),
            )
        except redis.RedisError as e:
            raise CacheError(f"Redis set failed: {e}") from e

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            result = await self.client.delete(key)
            return result > 0
        except redis.RedisError as e:
            raise CacheError(f"Redis delete failed: {e}") from e

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            result = await self.client.exists(key)
            return result > 0
        except redis.RedisError as e:
            raise CacheError(f"Redis exists failed: {e}") from e

    async def flush(self) -> None:
        """Clear all cached data."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            await self.client.flushdb()
        except redis.RedisError as e:
            raise CacheError(f"Redis flush failed: {e}") from e
