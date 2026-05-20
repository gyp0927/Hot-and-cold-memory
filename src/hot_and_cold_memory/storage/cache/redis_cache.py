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
        self._prefix = "amem:"

    async def initialize(self) -> None:
        """Connect to Redis."""
        if not self.settings.CACHE_URL:
            raise CacheError("CACHE_URL not configured")

        self.client = redis.from_url(
            self.settings.CACHE_URL,
            decode_responses=True,
        )

    def _prefixed(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.client = None

    async def mget(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values in one round-trip."""
        if not self.client:
            raise CacheError("Redis not initialized")
        if not keys:
            return []
        try:
            pkeys = [self._prefixed(k) for k in keys]
            values = await self.client.mget(pkeys)
            return [json.loads(v) if v is not None else None for v in values]
        except redis.RedisError as e:
            raise CacheError(f"Redis mget failed: {e}") from e

    async def mset(self, items: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple values in one round-trip."""
        if not self.client:
            raise CacheError("Redis not initialized")
        if not items:
            return
        try:
            ttl = ttl or self.settings.CACHE_TTL_SECONDS
            prefixed = {self._prefixed(k): json.dumps(v) for k, v in items.items()}
            pipe = self.client.pipeline()
            pipe.mset(prefixed)
            for key in prefixed:
                pipe.expire(key, ttl)
            await pipe.execute()
        except redis.RedisError as e:
            raise CacheError(f"Redis mset failed: {e}") from e

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            value = await self.client.get(self._prefixed(key))
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
                self._prefixed(key),
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
            result = await self.client.delete(self._prefixed(key))
            return result > 0
        except redis.RedisError as e:
            raise CacheError(f"Redis delete failed: {e}") from e

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            result = await self.client.exists(self._prefixed(key))
            return result > 0
        except redis.RedisError as e:
            raise CacheError(f"Redis exists failed: {e}") from e

    async def flush(self) -> None:
        """Clear only keys with our prefix (safer than FLUSHDB)."""
        if not self.client:
            raise CacheError("Redis not initialized")

        try:
            cursor = 0
            while True:
                cursor, keys = await self.client.scan(
                    cursor, match=f"{self._prefix}*", count=100
                )
                if keys:
                    await self.client.delete(*keys)
                if cursor == 0:
                    break
        except redis.RedisError as e:
            raise CacheError(f"Redis flush failed: {e}") from e
