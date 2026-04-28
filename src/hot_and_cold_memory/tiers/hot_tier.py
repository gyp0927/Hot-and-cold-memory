"""Hot tier (short-term memory): stores full original content with embeddings."""

import uuid
from datetime import datetime
from typing import Any

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.core.exceptions import TierError
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.storage.cache.base import BaseCache
from hot_and_cold_memory.storage.document_store.base import BaseDocumentStore
from hot_and_cold_memory.storage.metadata_store.base import (
    BaseMetadataStore,
    MemoryItem,
)
from hot_and_cold_memory.storage.vector_store.base import BaseVectorStore

from .base import BaseTier, MemoryEntry, RetrievedMemory

logger = get_logger(__name__)


class HotTier(BaseTier):
    """Hot tier stores full original content with dense embeddings.

    Optimized for low-latency retrieval of frequently accessed memories.
    Memories with high topic frequency are routed here.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
        document_store: BaseDocumentStore,
        cache: BaseCache | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.document_store = document_store
        self.cache = cache
        self.collection = self.settings.VECTOR_DB_COLLECTION

    @property
    def tier_type(self) -> Tier:
        return Tier.HOT

    async def store_memories(
        self,
        memories: list[MemoryEntry],
        embeddings: list[list[float]],
        memory_type: str = "observation",
        source: str | None = None,
    ) -> list[MemoryItem]:
        """Store memories in the hot tier.

        Args:
            memories: Memory entries.
            embeddings: Embedding vectors for each memory.
            memory_type: Type of memory (observation/fact/reflection/summary).
            source: Source identifier (e.g., conversation ID).

        Returns:
            List of memory metadata.
        """
        if len(memories) != len(embeddings):
            raise TierError("Memories and embeddings count mismatch")

        memory_ids = [m.memory_id for m in memories]

        # 1. Store original content in document store
        await self.document_store.store_batch([
            (m.memory_id, m.content) for m in memories
        ])

        # 2. Store vectors
        payloads = [{
            "memory_id": str(m.memory_id),
            "tier": Tier.HOT.value,
            "tags": m.tags or [],
        } for m in memories]

        await self.vector_store.upsert(
            collection=self.collection,
            ids=memory_ids,
            vectors=embeddings,
            payloads=payloads,
        )

        # 3. Store metadata (new memories start with max frequency)
        now = datetime.utcnow()
        metadata_list = [
            MemoryItem(
                memory_id=memory.memory_id,
                tier=Tier.HOT,
                content=memory.content,
                original_length=len(memory.content),
                memory_type=memory_type,
                source=source,
                access_count=0,
                frequency_score=1.0,  # New memories start hot
                created_at=now,
                updated_at=now,
                tags=memory.tags or [],
            )
            for memory in memories
        ]
        await self.metadata_store.create_memories_batch(metadata_list)

        logger.info(
            "hot_tier_stored",
            memory_count=len(memories),
        )
        return metadata_list

    async def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve memories by vector similarity from hot tier."""
        # Search vectors
        results = await self.vector_store.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=top_k,
            filters=filters,
        )

        # Batch fetch metadata to avoid N+1 queries
        memory_ids = [r.memory_id for r in results]
        meta_map = {
            m.memory_id: m
            for m in await self.metadata_store.get_memories_batch(memory_ids)
        }

        memories = []
        for result in results:
            memory_id = result.memory_id

            # Try cache first
            text = None
            if self.cache:
                text = await self.cache.get(f"memory:{memory_id}")

            # Fetch from document store
            if text is None:
                text = await self.document_store.get(memory_id)
                if text and self.cache:
                    await self.cache.set(f"memory:{memory_id}", text)

            if text is None:
                logger.warning("memory_not_found", memory_id=str(memory_id))
                continue

            meta = meta_map.get(memory_id)
            memories.append(RetrievedMemory(
                memory_id=memory_id,
                content=text,
                score=result.score,
                tier=Tier.HOT,
                is_decompressed=False,
                access_count=meta.access_count if meta else 0,
                frequency_score=meta.frequency_score if meta else 0.0,
                memory_type=meta.memory_type if meta else "observation",
                metadata=result.payload or {},
            ))

        return memories

    async def get_by_id(self, memory_id: uuid.UUID) -> RetrievedMemory | None:
        """Get a specific memory by ID."""
        # Get from vector store for payload
        vec_result = await self.vector_store.get_by_id(self.collection, memory_id)
        if not vec_result:
            return None

        # Get text
        text = await self.document_store.get(memory_id)
        if text is None:
            return None

        # Get metadata
        meta = await self.metadata_store.get_memory(memory_id)

        return RetrievedMemory(
            memory_id=memory_id,
            content=text,
            score=1.0,
            tier=Tier.HOT,
            is_decompressed=False,
            access_count=meta.access_count if meta else 0,
            frequency_score=meta.frequency_score if meta else 0.0,
            memory_type=meta.memory_type if meta else "observation",
            metadata=vec_result.payload or {},
        )

    async def delete(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete memories from hot tier."""
        # Delete from all stores
        await self.vector_store.delete(self.collection, memory_ids)
        await self.document_store.delete(memory_ids)
        count = await self.metadata_store.delete_memories(memory_ids)

        # Clear cache
        if self.cache:
            for memory_id in memory_ids:
                await self.cache.delete(f"memory:{memory_id}")

        return count

    async def exists(self, memory_id: uuid.UUID) -> bool:
        """Check if memory exists in hot tier."""
        result = await self.vector_store.get_by_id(self.collection, memory_id)
        return result is not None
