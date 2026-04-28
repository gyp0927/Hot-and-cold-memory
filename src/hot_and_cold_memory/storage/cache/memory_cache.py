"""In-memory cache implementation."""

import time
from typing import Any

from hot_and_cold_memory.core.config import get_settings

from .base import BaseCache


class MemoryCache(BaseCache):
    """Simple in-memory cache with TTL support."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._expires: dict[str, float] = {}
        self.settings = get_settings()

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key in self._expires and time.time() > self._expires[key]:
            self._data.pop(key, None)
            self._expires.pop(key, None)
            return None
        return self._data.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        self._data[key] = value
        if ttl is not None:
            self._expires[key] = time.time() + ttl
        else:
            self._expires[key] = time.time() + self.settings.CACHE_TTL_SECONDS

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        existed = key in self._data
        self._data.pop(key, None)
        self._expires.pop(key, None)
        return existed

    async def exists(self, key: str) -> bool:
        """Check if key exists and not expired."""
        if key in self._expires and time.time() > self._expires[key]:
            return False
        return key in self._data

    async def flush(self) -> None:
        """Clear all cached data."""
        self._data.clear()
        self._expires.clear()
