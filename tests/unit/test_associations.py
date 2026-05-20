"""Tests for memory association graph."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hot_and_cold_memory.core.config import Tier
from hot_and_cold_memory.frequency.tracker import FrequencyTracker
from hot_and_cold_memory.storage.metadata_store.base import MemoryItem, MemoryLink


class TestMemoryLinks:
    """Test PostgresMetadataStore link operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_related(self, metadata_store):
        """Create links and retrieve related memories."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory one",
            original_length=10,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory two",
            original_length=10,
        )
        await metadata_store.create_memory(m1)
        await metadata_store.create_memory(m2)

        link = MemoryLink(
            source_memory_id=m1.memory_id,
            target_memory_id=m2.memory_id,
            link_type="coaccess",
            strength=1.0,
        )
        await metadata_store.create_link(link)

        related = await metadata_store.get_related_memories(m1.memory_id)
        assert len(related) == 1
        assert related[0][1].memory_id == m2.memory_id
        assert related[0][0].strength == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_link_upsert_increases_strength(self, metadata_store):
        """Creating the same link twice should increase strength."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory one",
            original_length=10,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory two",
            original_length=10,
        )
        await metadata_store.create_memory(m1)
        await metadata_store.create_memory(m2)

        link = MemoryLink(
            source_memory_id=m1.memory_id,
            target_memory_id=m2.memory_id,
            link_type="coaccess",
            strength=1.0,
        )
        await metadata_store.create_link(link)
        await metadata_store.create_link(link)

        related = await metadata_store.get_related_memories(m1.memory_id)
        assert len(related) == 1
        assert related[0][0].strength > 1.0

    @pytest.mark.asyncio
    async def test_reverse_link_detected(self, metadata_store):
        """Reverse link (b->a) should be detected as existing link."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory one",
            original_length=10,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory two",
            original_length=10,
        )
        await metadata_store.create_memory(m1)
        await metadata_store.create_memory(m2)

        link = MemoryLink(
            source_memory_id=m1.memory_id,
            target_memory_id=m2.memory_id,
            link_type="coaccess",
            strength=1.0,
        )
        await metadata_store.create_link(link)

        reverse = MemoryLink(
            source_memory_id=m2.memory_id,
            target_memory_id=m1.memory_id,
            link_type="coaccess",
            strength=1.0,
        )
        await metadata_store.create_link(reverse)

        related = await metadata_store.get_related_memories(m1.memory_id)
        assert len(related) == 1

    @pytest.mark.asyncio
    async def test_delete_links_for_memories(self, metadata_store):
        """Deleting memories should remove their links."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory one",
            original_length=10,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory two",
            original_length=10,
        )
        await metadata_store.create_memory(m1)
        await metadata_store.create_memory(m2)

        await metadata_store.create_link(
            MemoryLink(
                source_memory_id=m1.memory_id,
                target_memory_id=m2.memory_id,
                link_type="coaccess",
            )
        )

        deleted = await metadata_store.delete_links_for_memories([m1.memory_id])
        assert deleted == 1

        related = await metadata_store.get_related_memories(m1.memory_id)
        assert len(related) == 0

    @pytest.mark.asyncio
    async def test_get_related_filter_by_link_type(self, metadata_store):
        """Link type filter should restrict results."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory one",
            original_length=10,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory two",
            original_length=10,
        )
        m3 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="memory three",
            original_length=10,
        )
        for m in [m1, m2, m3]:
            await metadata_store.create_memory(m)

        await metadata_store.create_link(
            MemoryLink(
                source_memory_id=m1.memory_id,
                target_memory_id=m2.memory_id,
                link_type="coaccess",
            )
        )
        await metadata_store.create_link(
            MemoryLink(
                source_memory_id=m1.memory_id,
                target_memory_id=m3.memory_id,
                link_type="semantic",
            )
        )

        coaccess = await metadata_store.get_related_memories(m1.memory_id, link_type="coaccess")
        assert len(coaccess) == 1
        assert coaccess[0][0].link_type == "coaccess"


class TestFrequencyTrackerCoaccess:
    """Test coaccess link creation in FrequencyTracker."""

    @pytest.mark.asyncio
    async def test_record_access_creates_links(self, monkeypatch):
        """When multiple memories are accessed together, coaccess links are created."""
        monkeypatch.setenv("ENABLE_ASSOCIATIONS", "true")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        store = MagicMock()
        store.increment_access = AsyncMock()
        store.create_access_logs_batch = AsyncMock()
        store.create_link = AsyncMock()
        store.get_cluster = AsyncMock(return_value=None)
        store.update_cluster = AsyncMock(return_value=None)

        ft = FrequencyTracker(
            metadata_store=store,
            vector_store=MagicMock(),
        )
        # Mock cluster lookup to avoid vector store calls
        ft._get_or_create_cluster = AsyncMock(return_value=uuid.uuid4())
        ft._recalculate_scores = AsyncMock()

        mids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
        await ft.record_access(mids, "test query")

        # C(3,2) = 3 pairs
        assert store.create_link.await_count == 3

    @pytest.mark.asyncio
    async def test_single_memory_no_links(self, monkeypatch):
        """Only one memory accessed means no links."""
        monkeypatch.setenv("ENABLE_ASSOCIATIONS", "true")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        store = MagicMock()
        store.increment_access = AsyncMock()
        store.create_access_logs_batch = AsyncMock()
        store.create_link = AsyncMock()
        store.get_cluster = AsyncMock(return_value=None)
        store.update_cluster = AsyncMock(return_value=None)

        ft = FrequencyTracker(
            metadata_store=store,
            vector_store=MagicMock(),
        )
        ft._get_or_create_cluster = AsyncMock(return_value=uuid.uuid4())
        ft._recalculate_scores = AsyncMock()

        await ft.record_access([uuid.uuid4()], "test query")
        store.create_link.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_associations_skip_links(self, monkeypatch):
        """When ENABLE_ASSOCIATIONS=false, no links are created."""
        monkeypatch.setenv("ENABLE_ASSOCIATIONS", "false")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        store = MagicMock()
        store.increment_access = AsyncMock()
        store.create_access_logs_batch = AsyncMock()
        store.create_link = AsyncMock()
        store.get_cluster = AsyncMock(return_value=None)
        store.update_cluster = AsyncMock(return_value=None)

        ft = FrequencyTracker(
            metadata_store=store,
            vector_store=MagicMock(),
        )
        ft._get_or_create_cluster = AsyncMock(return_value=uuid.uuid4())
        ft._recalculate_scores = AsyncMock()

        mids = [uuid.uuid4(), uuid.uuid4()]
        await ft.record_access(mids, "test query")
        store.create_link.assert_not_awaited()
