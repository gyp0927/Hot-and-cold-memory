"""Local filesystem document store implementation."""

import asyncio
import uuid
from pathlib import Path

import aiofiles

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.exceptions import DocumentStoreError

from .base import BaseDocumentStore


class LocalDocumentStore(BaseDocumentStore):
    """Store documents on local filesystem."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_path = Path(self.settings.DOCUMENT_STORE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, chunk_id: uuid.UUID) -> Path:
        """Get filesystem path for a chunk."""
        id_str = str(chunk_id)
        # Shard by first 2 chars to avoid too many files in one dir
        return self.base_path / id_str[:2] / f"{id_str}.txt"

    async def store(self, chunk_id: uuid.UUID, content: str) -> None:
        """Store document content."""
        path = self._path(chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(content)
        except OSError as e:
            raise DocumentStoreError(f"Failed to store {chunk_id}: {e}") from e

    async def store_batch(self, items: list[tuple[uuid.UUID, str]]) -> None:
        """Store multiple documents concurrently."""
        await asyncio.gather(*[
            self.store(chunk_id, content)
            for chunk_id, content in items
        ])

    async def get(self, chunk_id: uuid.UUID) -> str | None:
        """Retrieve document content."""
        path = self._path(chunk_id)

        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, encoding="utf-8") as f:
                return await f.read()
        except OSError:
            return None

    async def delete(self, chunk_ids: list[uuid.UUID]) -> int:
        """Delete documents."""
        count = 0
        for chunk_id in chunk_ids:
            path = self._path(chunk_id)
            if path.exists():
                path.unlink()
                count += 1
        return count

    async def exists(self, chunk_id: uuid.UUID) -> bool:
        """Check if document exists."""
        return self._path(chunk_id).exists()
