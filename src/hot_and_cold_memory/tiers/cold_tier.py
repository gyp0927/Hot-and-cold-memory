"""Cold tier (long-term memory): stores compressed summaries with summary embeddings."""

import uuid
from datetime import datetime
from typing import Any

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.core.exceptions import TierError
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.ingestion.embedder import Embedder
from hot_and_cold_memory.storage.cache.base import BaseCache
from hot_and_cold_memory.storage.document_store.base import BaseDocumentStore
from hot_and_cold_memory.storage.metadata_store.base import (
    BaseMetadataStore,
    MemoryItem,
)
from hot_and_cold_memory.storage.vector_store.base import BaseVectorStore

from .base import BaseTier, MemoryEntry, RetrievedMemory
from .compression import CompressionEngine

logger = get_logger(__name__)


class ColdTier(BaseTier):
    """Cold tier stores LLM-compressed summaries with summary embeddings.

    Storage-efficient for long-term memories that are less frequently accessed.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
        document_store: BaseDocumentStore,
        compression_engine: CompressionEngine,
        cache: BaseCache | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.document_store = document_store
        self.compression_engine = compression_engine
        self.cache = cache
        self.embedder = embedder or Embedder()
        self.collection = f"{self.settings.VECTOR_DB_COLLECTION}_cold"

    @property
    def tier_type(self) -> Tier:
        return Tier.COLD

    async def store_memories(
        self,
        memories: list[MemoryEntry],
        original_embeddings: list[list[float]] | None = None,
        memory_type: str = "observation",
        source: str | None = None,
    ) -> list[MemoryItem]:
        """Compress and store memories in the cold tier.

        Args:
            memories: Memory entries with full content.
            original_embeddings: Optional original embeddings (unused for cold tier).
            memory_type: Type of memory.
            source: Source identifier.

        Returns:
            List of memory metadata.
        """
        # 1. Compress memories via LLM
        compressed = await self.compression_engine.compress_batch(memories)

        # 2. Generate embeddings for summaries
        summary_texts = [c.summary_text for c in compressed]
        summary_embeddings = await self.embedder.embed_batch(summary_texts)

        # 3. Store compressed summaries in document store
        await self.document_store.store_batch([
            (m.memory_id, comp.summary_text)
            for m, comp in zip(memories, compressed)
        ])

        # 4. Store summary vectors
        memory_ids = [m.memory_id for m in memories]
        payloads = [{
            "memory_id": str(m.memory_id),
            "tier": Tier.COLD.value,
            "tags": m.tags or [],
            "compressed": True,
        } for m in memories]

        await self.vector_store.upsert(
            collection=self.collection,
            ids=memory_ids,
            vectors=summary_embeddings,
            payloads=payloads,
        )

        # 5. Store metadata
        now = datetime.utcnow()
        metadata_list = [
            MemoryItem(
                memory_id=memory.memory_id,
                tier=Tier.COLD,
                content=comp.summary_text,
                original_length=len(memory.content),
                memory_type=memory_type,
                source=source,
                access_count=0,
                frequency_score=0.0,
                created_at=now,
                updated_at=now,
                tags=memory.tags or [],
            )
            for memory, comp in zip(memories, compressed)
        ]
        await self.metadata_store.create_memories_batch(metadata_list)

        logger.info(
            "cold_tier_stored",
            memory_count=len(memories),
            avg_compression=sum(len(c.summary_text) for c in compressed) / max(sum(len(m.content) for m in memories), 1),
        )
        return metadata_list

    async def store_raw_memories(
        self,
        memories: list[MemoryEntry],
        embeddings: list[list[float]],
        memory_type: str = "observation",
        source: str | None = None,
        initial_score: float = 0.1,
    ) -> list[MemoryItem]:
        """Store raw (uncompressed) memories directly into the cold tier.

        This skips LLM compression, saving costs for newly created memories
        that can be later compressed during scheduled consolidation cycles.

        Args:
            memories: Memory entries with full content.
            embeddings: Pre-computed embedding vectors for each memory.
            memory_type: Type of memory.
            source: Source identifier.
            initial_score: Starting frequency score (default 0.1 for cold memories).

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

        # 2. Store vectors (original text embeddings for similarity search)
        payloads = [{
            "memory_id": str(m.memory_id),
            "tier": Tier.COLD.value,
            "tags": m.tags or [],
            "compressed": False,
        } for m in memories]

        await self.vector_store.upsert(
            collection=self.collection,
            ids=memory_ids,
            vectors=embeddings,
            payloads=payloads,
        )

        # 3. Store metadata
        now = datetime.utcnow()
        metadata_list = [
            MemoryItem(
                memory_id=memory.memory_id,
                tier=Tier.COLD,
                content=memory.content,
                original_length=len(memory.content),
                memory_type=memory_type,
                source=source,
                access_count=0,
                frequency_score=initial_score,
                created_at=now,
                updated_at=now,
                tags=memory.tags or [],
            )
            for memory in memories
        ]
        await self.metadata_store.create_memories_batch(metadata_list)

        logger.info(
            "cold_tier_raw_stored",
            memory_count=len(memories),
            initial_score=initial_score,
        )
        return metadata_list

    async def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve memories from cold tier.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.
            filters: Optional metadata filters.
        """
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

            # Try cache
            text = None
            if self.cache:
                text = await self.cache.get(f"memory:{memory_id}")

            # Fetch summary from document store
            if text is None:
                text = await self.document_store.get(memory_id)
                if text and self.cache:
                    await self.cache.set(f"memory:{memory_id}", text)

            if text is None:
                continue

            meta = meta_map.get(memory_id)

            memories.append(RetrievedMemory(
                memory_id=memory_id,
                content=text,
                score=result.score,
                tier=Tier.COLD,
                is_decompressed=False,
                access_count=meta.access_count if meta else 0,
                frequency_score=meta.frequency_score if meta else 0.0,
                memory_type=meta.memory_type if meta else "observation",
                metadata=result.payload or {},
            ))

        return memories

    async def get_by_id(self, memory_id: uuid.UUID) -> RetrievedMemory | None:
        """Get a specific memory by ID."""
        vec_result = await self.vector_store.get_by_id(self.collection, memory_id)
        if not vec_result:
            return None

        text = await self.document_store.get(memory_id)
        if text is None:
            return None

        meta = await self.metadata_store.get_memory(memory_id)

        return RetrievedMemory(
            memory_id=memory_id,
            content=text,
            score=1.0,
            tier=Tier.COLD,
            is_decompressed=False,
            access_count=meta.access_count if meta else 0,
            frequency_score=meta.frequency_score if meta else 0.0,
            memory_type=meta.memory_type if meta else "observation",
            metadata=vec_result.payload or {},
        )

    async def delete(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete memories from cold tier."""
        await self.vector_store.delete(self.collection, memory_ids)
        await self.document_store.delete(memory_ids)
        count = await self.metadata_store.delete_memories(memory_ids)

        if self.cache:
            for memory_id in memory_ids:
                await self.cache.delete(f"memory:{memory_id}")

        return count

    async def exists(self, memory_id: uuid.UUID) -> bool:
        """Check if memory exists in cold tier."""
        result = await self.vector_store.get_by_id(self.collection, memory_id)
        return result is not None
