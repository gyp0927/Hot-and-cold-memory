"""Abstract tier interface."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from adaptive_rag.core.config import Tier


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk retrieved from a tier."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    tier: Tier
    is_decompressed: bool
    access_count: int = 0
    frequency_score: float = 0.0
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class Chunk:
    """A document chunk for storage."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    index: int = 0
    tags: list[str] | None = None


class BaseTier(ABC):
    """Abstract base for hot and cold tier implementations."""

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
    ) -> list[RetrievedChunk]:
        """Retrieve chunks by vector similarity."""
        pass

    @abstractmethod
    async def get_by_id(self, chunk_id: uuid.UUID) -> RetrievedChunk | None:
        """Get a specific chunk by ID."""
        pass

    @abstractmethod
    async def delete(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete chunks. Returns number deleted."""
        pass

    @abstractmethod
    async def exists(self, chunk_id: uuid.UUID) -> bool:
        """Check if chunk exists in this tier."""
        pass
