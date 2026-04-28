"""Memory ingestion pipeline for agent memory system."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.frequency.tracker import FrequencyTracker
from hot_and_cold_memory.monitoring.metrics import MEMORIES_TOTAL
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore
from hot_and_cold_memory.tiers.cold_tier import ColdTier
from hot_and_cold_memory.tiers.hot_tier import HotTier

from .embedder import Embedder

logger = get_logger(__name__)


@dataclass
class MemoryWriteResult:
    """Result of writing a memory."""

    memory_id: uuid.UUID
    status: str = "pending"
    tier: str = ""
    error: str | None = None
    message: str | None = None
    processing_time_ms: float = 0.0


class MemoryPipeline:
    """Orchestrates memory ingestion into the system.

    Instead of blindly placing every new memory into the hot tier, the pipeline
    estimates the topic's historical popularity via the frequency tracker. Hot
    topics go to hot tier; new / cold topics skip compression and go directly to
    cold tier as raw text, saving LLM costs. After ingestion the hot tier
    capacity is checked and the coldest memories evicted if needed.
    """

    # Threshold above which a topic is considered "hot" at ingestion time.
    HOT_TOPIC_THRESHOLD: float = 0.5

    # Hard cap on hot tier memories. When exceeded, the coldest evict percent
    # of hot memories are pushed to cold tier.
    hot_tier_capacity: int = 10000
    evict_percent: float = 0.1

    def __init__(
        self,
        metadata_store: BaseMetadataStore,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        embedder: Embedder,
        frequency_tracker: FrequencyTracker,
        migration_engine=None,
    ) -> None:
        self.settings = get_settings()
        self.metadata_store = metadata_store
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.embedder = embedder
        self.frequency_tracker = frequency_tracker
        self.hot_tier_capacity = self.settings.HOT_TIER_CAPACITY
        self.evict_percent = self.settings.HOT_TIER_EVICT_PERCENT
        self.migration_engine = migration_engine

    async def write_memory(
        self,
        content: str,
        memory_type: str = "observation",
        source: str | None = None,
        importance: float = 0.5,
        tags: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Write a memory into the system.

        Args:
            content: Memory content text.
            memory_type: Type of memory (observation/fact/reflection/summary).
            source: Source identifier (e.g., conversation ID).
            importance: Initial importance score (0-1).
            tags: Optional tags.
            attributes: Optional additional attributes.

        Returns:
            Memory write result.
        """
        start_time = datetime.utcnow()
        memory_id = uuid.uuid4()

        try:
            if not content.strip():
                return MemoryWriteResult(
                    memory_id=memory_id,
                    status="failed",
                    error="Memory content is empty",
                )

            # 1. Generate embedding
            embeddings = await self.embedder.embed_batch([content])
            embedding = embeddings[0]

            # 2. Check topic frequency to decide tier
            topic_info = await self.frequency_tracker.get_topic_frequency(embedding)
            is_hot = (
                topic_info.frequency >= self.HOT_TOPIC_THRESHOLD
                or topic_info.access_count >= self.settings.HOT_ACCESS_COUNT_THRESHOLD
            )

            from hot_and_cold_memory.tiers.base import MemoryEntry
            entry = MemoryEntry(
                memory_id=memory_id,
                content=content,
                tags=tags or [],
            )

            if is_hot:
                # Store in hot tier (short-term memory)
                await self.hot_tier.store_memories(
                    memories=[entry],
                    embeddings=[embedding],
                    memory_type=memory_type,
                    source=source,
                )
                tier = "hot"
            else:
                # Store in cold tier as raw (long-term memory, uncompressed)
                await self.cold_tier.store_raw_memories(
                    memories=[entry],
                    embeddings=[embedding],
                    memory_type=memory_type,
                    source=source,
                    initial_score=0.1,
                )
                tier = "cold"

            # Update importance if specified
            if importance != 0.5:
                await self.metadata_store.update_memory(
                    memory_id=memory_id,
                    updates={"importance": importance, "attributes": attributes or {}},
                )

            # Hot tier capacity check
            await self._enforce_hot_tier_capacity()

            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Update Prometheus gauges
            hot_total = await self.metadata_store.count_memories_by_tier(Tier.HOT)
            cold_total = await self.metadata_store.count_memories_by_tier(Tier.COLD)
            MEMORIES_TOTAL.labels(tier="hot").set(hot_total)
            MEMORIES_TOTAL.labels(tier="cold").set(cold_total)

            logger.info(
                "memory_written",
                memory_id=str(memory_id),
                tier=tier,
                memory_type=memory_type,
                elapsed_ms=elapsed,
            )

            return MemoryWriteResult(
                memory_id=memory_id,
                status="success",
                tier=tier,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error("memory_write_failed", memory_id=str(memory_id), error=str(e))
            return MemoryWriteResult(
                memory_id=memory_id,
                status="failed",
                error=str(e),
            )

    async def write_memories_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[MemoryWriteResult]:
        """Write multiple memories in batch.

        Args:
            items: List of memory dicts with keys: content, memory_type, source,
                   importance, tags, attributes.

        Returns:
            List of write results.
        """
        results = []
        for item in items:
            result = await self.write_memory(
                content=item["content"],
                memory_type=item.get("memory_type", "observation"),
                source=item.get("source"),
                importance=item.get("importance", 0.5),
                tags=item.get("tags"),
                attributes=item.get("attributes"),
            )
            results.append(result)
        return results

    async def delete_memory(self, memory_id: uuid.UUID) -> bool:
        """Delete a memory from all stores.

        Args:
            memory_id: Memory to delete.

        Returns:
            True if deleted.
        """
        meta = await self.metadata_store.get_memory(memory_id)
        if not meta:
            return False

        if meta.tier == Tier.HOT:
            await self.hot_tier.delete([memory_id])
        else:
            await self.cold_tier.delete([memory_id])

        logger.info("memory_deleted", memory_id=str(memory_id))
        return True

    async def _enforce_hot_tier_capacity(self) -> None:
        """If hot tier exceeds capacity, evict the coldest memories to cold tier."""
        try:
            hot_count = await self.metadata_store.count_memories_by_tier(tier=Tier.HOT)
            if hot_count > self.hot_tier_capacity:
                if self.migration_engine is not None:
                    evicted = await self.migration_engine.evict_coldest(
                        percent=self.evict_percent
                    )
                    logger.warning(
                        "hot_tier_capacity_exceeded",
                        hot_count=hot_count,
                        capacity=self.hot_tier_capacity,
                        evicted=len(evicted),
                    )
                else:
                    logger.warning(
                        "hot_tier_capacity_exceeded_no_migration_engine",
                        hot_count=hot_count,
                        capacity=self.hot_tier_capacity,
                    )
        except Exception as e:
            logger.error("hot_tier_capacity_check_failed", error=str(e))
