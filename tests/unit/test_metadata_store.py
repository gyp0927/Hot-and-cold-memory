"""Unit tests for metadata store."""

import uuid
from datetime import datetime

import pytest

from adaptive_memory.core.config import Tier
from adaptive_memory.storage.metadata_store.base import (
    MemoryItem,
    TopicCluster,
)


class TestPostgresMetadataStore:
    """Test metadata store operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_memory(self, metadata_store):
        """Test creating and retrieving a memory."""
        memory_id = uuid.uuid4()
        meta = MemoryItem(
            memory_id=memory_id,
            tier=Tier.HOT,
            content="Test memory content",
            original_length=100,
            memory_type="fact",
            access_count=0,
            frequency_score=1.0,
        )
        await metadata_store.create_memory(meta)

        retrieved = await metadata_store.get_memory(memory_id)
        assert retrieved is not None
        assert retrieved.memory_id == memory_id
        assert retrieved.tier == Tier.HOT
        assert retrieved.frequency_score == 1.0
        assert retrieved.memory_type == "fact"

    @pytest.mark.asyncio
    async def test_get_memories_batch(self, metadata_store):
        """Test batch memory retrieval."""
        memory_ids = [uuid.uuid4() for _ in range(5)]
        for mid in memory_ids:
            await metadata_store.create_memory(
                MemoryItem(
                    memory_id=mid,
                    tier=Tier.HOT,
                    content="test",
                    original_length=50,
                )
            )

        results = await metadata_store.get_memories_batch(memory_ids)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_update_memories_batch(self, metadata_store):
        """Test batch memory update."""
        memory_ids = [uuid.uuid4() for _ in range(3)]
        for mid in memory_ids:
            await metadata_store.create_memory(
                MemoryItem(
                    memory_id=mid,
                    tier=Tier.HOT,
                    content="test",
                    original_length=50,
                    frequency_score=0.0,
                )
            )

        updates = {mid: {"frequency_score": 0.8} for mid in memory_ids}
        await metadata_store.update_memories_batch(updates)

        for mid in memory_ids:
            retrieved = await metadata_store.get_memory(mid)
            assert retrieved.frequency_score == 0.8

    @pytest.mark.asyncio
    async def test_increment_access_batch(self, metadata_store):
        """Test batch access increment."""
        memory_ids = [uuid.uuid4() for _ in range(3)]
        for mid in memory_ids:
            await metadata_store.create_memory(
                MemoryItem(
                    memory_id=mid,
                    tier=Tier.HOT,
                    content="test",
                    original_length=50,
                    access_count=0,
                )
            )

        await metadata_store.increment_access(
            memory_ids=memory_ids,
            cluster_id=None,
            timestamp=datetime.utcnow(),
        )

        for mid in memory_ids:
            retrieved = await metadata_store.get_memory(mid)
            assert retrieved.access_count == 1

    @pytest.mark.asyncio
    async def test_count_memories_by_tier(self, metadata_store):
        """Test counting memories by tier."""
        for _ in range(3):
            await metadata_store.create_memory(
                MemoryItem(
                    memory_id=uuid.uuid4(),
                    tier=Tier.HOT,
                    content="test",
                    original_length=50,
                )
            )
        for _ in range(2):
            await metadata_store.create_memory(
                MemoryItem(
                    memory_id=uuid.uuid4(),
                    tier=Tier.COLD,
                    content="test",
                    original_length=50,
                )
            )

        hot_count = await metadata_store.count_memories_by_tier(Tier.HOT)
        cold_count = await metadata_store.count_memories_by_tier(Tier.COLD)
        assert hot_count == 3
        assert cold_count == 2

    @pytest.mark.asyncio
    async def test_get_clusters_batch(self, metadata_store):
        """Test batch cluster retrieval."""
        cluster_ids = [uuid.uuid4() for _ in range(3)]
        for cid in cluster_ids:
            await metadata_store.create_cluster(
                TopicCluster(
                    cluster_id=cid,
                    centroid=[0.1, 0.2, 0.3],
                    representative_query="test query",
                )
            )

        results = await metadata_store.get_clusters_batch(cluster_ids)
        assert len(results) == 3
