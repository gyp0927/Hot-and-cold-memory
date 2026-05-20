"""Abstract base class for caches."""

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Clear all cached data."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize cache connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close cache connection."""
        pass

    @abstractmethod
    async def mget(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values in a single round-trip."""
        pass

    @abstractmethod
    async def mset(self, items: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple values in a single round-trip."""
        pass
