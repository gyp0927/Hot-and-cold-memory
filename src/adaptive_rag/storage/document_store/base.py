"""Abstract base class for document stores."""

import uuid
from abc import ABC, abstractmethod


class BaseDocumentStore(ABC):
    """Abstract interface for storing original document content."""

    @abstractmethod
    async def store(self, chunk_id: uuid.UUID, content: str) -> None:
        """Store document content."""
        pass

    @abstractmethod
    async def store_batch(self, items: list[tuple[uuid.UUID, str]]) -> None:
        """Store multiple documents."""
        pass

    @abstractmethod
    async def get(self, chunk_id: uuid.UUID) -> str | None:
        """Retrieve document content."""
        pass

    @abstractmethod
    async def delete(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete documents. Returns count deleted."""
        pass

    @abstractmethod
    async def exists(self, chunk_id: uuid.UUID) -> bool:
        """Check if document exists."""
        pass
