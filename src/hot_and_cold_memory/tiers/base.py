"""Abstract tier interface."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from hot_and_cold_memory.core.config import Tier


@dataclass(frozen=True)
class RetrievedMemory:
    """A memory retrieved from a tier."""

    memory_id: uuid.UUID
    content: str
    score: float
    tier: Tier
    is_decompressed: bool
    access_count: int = 0
    frequency_score: float = 0.0
    memory_type: str = "observation"
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MemoryEntry:
    """A memory entry for storage."""

    memory_id: uuid.UUID
    content: str
    tags: list[str] | None = None


class BaseTier(ABC):
    """Abstract base for hot (short-term) and cold (long-term) tier implementations."""

    @property
    @abstractmethod
    def tier_type(self) -> Tier:
        """Return the tier type."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve memories by vector similarity."""
        pass

    @abstractmethod
    async def get_by_id(self, memory_id: uuid.UUID) -> RetrievedMemory | None:
        """Get a specific memory by ID."""
        pass

    @abstractmethod
    async def delete(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete memories. Returns number deleted."""
        pass

    @abstractmethod
    async def exists(self, memory_id: uuid.UUID) -> bool:
        """Check if memory exists in this tier."""
        pass
