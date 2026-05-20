"""Tests for true forgetting mechanism (TTL + active deletion)."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.migration.engine import MigrationEngine
from hot_and_cold_memory.storage.metadata_store.base import MemoryItem


class TestQueryForgettableMemories:
    """Test PostgresMetadataStore.query_forgettable_memories."""

    @pytest.mark.asyncio
    async def test_returns_old_compressed_low_importance(self, metadata_store):
        """Eligible cold memories should be returned."""
        old = datetime.now(timezone.utc) - timedelta(days=40)
        meta = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.COLD,
            content="old cold memory",
            original_length=100,
            importance=0.1,
            compressed=True,
            last_accessed_at=old,
        )
        await metadata_store.create_memory(meta)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        results = await metadata_store.query_forgettable_memories(
            tier=Tier.COLD, max_importance=0.2, cutoff=cutoff, limit=10
        )
        assert len(results) == 1
        assert results[0].memory_id == meta.memory_id

    @pytest.mark.asyncio
    async def test_skips_recent_access(self, metadata_store):
        """Recently accessed memories should not be returned."""
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        meta = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.COLD,
            content="recent cold memory",
            original_length=100,
            importance=0.1,
            compressed=True,
            last_accessed_at=recent,
        )
        await metadata_store.create_memory(meta)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        results = await metadata_store.query_forgettable_memories(
            tier=Tier.COLD, max_importance=0.2, cutoff=cutoff, limit=10
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_skips_uncompressed(self, metadata_store):
        """Uncompressed memories should not be returned."""
        old = datetime.now(timezone.utc) - timedelta(days=40)
        meta = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.COLD,
            content="uncompressed cold memory",
            original_length=100,
            importance=0.1,
            compressed=False,
            last_accessed_at=old,
        )
        await metadata_store.create_memory(meta)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        results = await metadata_store.query_forgettable_memories(
            tier=Tier.COLD, max_importance=0.2, cutoff=cutoff, limit=10
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_skips_high_importance(self, metadata_store):
        """High-importance memories should not be returned."""
        old = datetime.now(timezone.utc) - timedelta(days=40)
        meta = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.COLD,
            content="important cold memory",
            original_length=100,
            importance=0.5,
            compressed=True,
            last_accessed_at=old,
        )
        await metadata_store.create_memory(meta)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        results = await metadata_store.query_forgettable_memories(
            tier=Tier.COLD, max_importance=0.2, cutoff=cutoff, limit=10
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_skips_wrong_tier(self, metadata_store):
        """Hot-tier memories should not be returned."""
        old = datetime.now(timezone.utc) - timedelta(days=40)
        meta = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="hot memory",
            original_length=100,
            importance=0.1,
            compressed=True,
            last_accessed_at=old,
        )
        await metadata_store.create_memory(meta)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        results = await metadata_store.query_forgettable_memories(
            tier=Tier.COLD, max_importance=0.2, cutoff=cutoff, limit=10
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_never_accessed_uses_created_at(self, metadata_store):
        """Memories never accessed use created_at for age check."""
        old = datetime.now(timezone.utc) - timedelta(days=40)
        meta = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.COLD,
            content="never accessed",
            original_length=100,
            importance=0.1,
            compressed=True,
            last_accessed_at=None,
            created_at=old,
        )
        await metadata_store.create_memory(meta)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        results = await metadata_store.query_forgettable_memories(
            tier=Tier.COLD, max_importance=0.2, cutoff=cutoff, limit=10
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_never_accessed_recent_created_at_skipped(self, metadata_store):
        """Never accessed but recently created should be skipped."""
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        meta = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.COLD,
            content="never accessed recent",
            original_length=100,
            importance=0.1,
            compressed=True,
            last_accessed_at=None,
            created_at=recent,
        )
        await metadata_store.create_memory(meta)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        results = await metadata_store.query_forgettable_memories(
            tier=Tier.COLD, max_importance=0.2, cutoff=cutoff, limit=10
        )
        assert len(results) == 0


class TestMigrationEngineForget:
    """Test MigrationEngine._forget_cold_memories."""

    @pytest.fixture
    def engine(self, monkeypatch):
        """Create engine with mocked tiers and store."""
        monkeypatch.setenv("ENABLE_FORGETTING", "true")
        monkeypatch.setenv("FORGET_MIN_IMPORTANCE", "0.2")
        monkeypatch.setenv("FORGET_MIN_DAYS_SINCE_ACCESS", "30")
        monkeypatch.setenv("FORGET_BATCH_SIZE", "100")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        hot_tier = MagicMock()
        cold_tier = MagicMock()
        cold_tier.delete = AsyncMock()
        store = MagicMock()
        store.delete_memories = AsyncMock(return_value=2)

        engine = MigrationEngine(
            hot_tier=hot_tier,
            cold_tier=cold_tier,
            metadata_store=store,
        )
        return engine

    @pytest.mark.asyncio
    async def test_forget_deletes_from_both_stores(self, engine):
        """Eligible memories are deleted from cold tier and metadata store."""
        old = datetime.now(timezone.utc) - timedelta(days=40)
        forgettable = [
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.COLD,
                content="forget me",
                original_length=10,
                importance=0.1,
                compressed=True,
                last_accessed_at=old,
            ),
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.COLD,
                content="forget me too",
                original_length=10,
                importance=0.15,
                compressed=True,
                last_accessed_at=old,
            ),
        ]
        engine.metadata_store.query_forgettable_memories = AsyncMock(return_value=forgettable)

        result = await engine._forget_cold_memories()

        assert len(result) == 2
        engine.cold_tier.delete.assert_awaited_once_with([m.memory_id for m in forgettable])
        engine.metadata_store.delete_memories.assert_awaited_once_with([m.memory_id for m in forgettable])

    @pytest.mark.asyncio
    async def test_forget_disabled_returns_empty(self, monkeypatch):
        """When forgetting is disabled, nothing is deleted."""
        monkeypatch.setenv("ENABLE_FORGETTING", "false")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        store = MagicMock()
        store.query_forgettable_memories = AsyncMock()
        store.delete_memories = AsyncMock()
        cold_tier = MagicMock()
        cold_tier.delete = AsyncMock()

        engine = MigrationEngine(
            hot_tier=MagicMock(),
            cold_tier=cold_tier,
            metadata_store=store,
        )

        result = await engine._forget_cold_memories()
        assert result == []
        store.query_forgettable_memories.assert_not_awaited()
        cold_tier.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forget_no_candidates(self, engine):
        """When no candidates match, nothing is deleted."""
        engine.metadata_store.query_forgettable_memories = AsyncMock(return_value=[])

        result = await engine._forget_cold_memories()
        assert result == []
        engine.cold_tier.delete.assert_not_awaited()
        engine.metadata_store.delete_memories.assert_not_awaited()


class TestPersistHotToColdExpiresAt:
    """Test that _persist_hot_to_cold sets compressed=True and expires_at correctly."""

    @pytest.fixture
    def engine(self, monkeypatch):
        """Create engine with mocked dependencies."""
        monkeypatch.setenv("FORGET_MIN_IMPORTANCE", "0.2")
        monkeypatch.setenv("FORGET_MIN_DAYS_SINCE_ACCESS", "30")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        hot_tier = MagicMock()
        cold_tier = MagicMock()
        cold_tier.document_store = MagicMock()
        cold_tier.document_store.store_batch = AsyncMock()
        cold_tier.vector_store = MagicMock()
        cold_tier.vector_store.upsert = AsyncMock()

        store = MagicMock()
        store.create_memory = AsyncMock()
        store.create_migration_log = AsyncMock()

        engine = MigrationEngine(
            hot_tier=hot_tier,
            cold_tier=cold_tier,
            metadata_store=store,
        )
        # Mock embedder
        engine.embedder = MagicMock()
        engine.embedder.embed = AsyncMock(return_value=[0.1, 0.2])
        return engine

    @pytest.mark.asyncio
    async def test_low_importance_gets_expires_at(self, engine):
        """Memory with importance below threshold gets expires_at."""
        from hot_and_cold_memory.tiers.base import RetrievedMemory
        from hot_and_cold_memory.tiers.compression import CompressedChunk

        mem_id = uuid.uuid4()
        record = RetrievedMemory(
            memory_id=mem_id,
            content="original content here",
            score=0.5,
            tier=Tier.HOT,
            is_decompressed=False,
        )
        compressed = CompressedChunk(
            chunk_id=mem_id,
            summary_text="compressed summary",
            key_entities=[],
            key_facts=[],
            compression_ratio=0.5,
        )

        # importance below FORGET_MIN_IMPORTANCE (0.2)
        engine.metadata_store.get_memory = AsyncMock(return_value=MemoryItem(
            memory_id=mem_id,
            tier=Tier.HOT,
            content="original",
            original_length=20,
            importance=0.1,
        ))
        engine.hot_tier.delete = AsyncMock()

        result = await engine._persist_hot_to_cold(record, compressed)
        assert result.success is True

        # Inspect the MemoryItem passed to create_memory
        call_args = engine.metadata_store.create_memory.await_args
        created_item: MemoryItem = call_args[0][0]
        assert created_item.compressed is True
        assert created_item.expires_at is not None
        assert created_item.importance == 0.1

    @pytest.mark.asyncio
    async def test_high_importance_no_expires_at(self, engine):
        """Memory with importance above threshold gets no expires_at."""
        from hot_and_cold_memory.tiers.base import RetrievedMemory
        from hot_and_cold_memory.tiers.compression import CompressedChunk

        mem_id = uuid.uuid4()
        record = RetrievedMemory(
            memory_id=mem_id,
            content="original content here",
            score=0.5,
            tier=Tier.HOT,
            is_decompressed=False,
        )
        compressed = CompressedChunk(
            chunk_id=mem_id,
            summary_text="compressed summary",
            key_entities=[],
            key_facts=[],
            compression_ratio=0.5,
        )

        # importance above FORGET_MIN_IMPORTANCE (0.2)
        engine.metadata_store.get_memory = AsyncMock(return_value=MemoryItem(
            memory_id=mem_id,
            tier=Tier.HOT,
            content="original",
            original_length=20,
            importance=0.5,
        ))
        engine.hot_tier.delete = AsyncMock()

        result = await engine._persist_hot_to_cold(record, compressed)
        assert result.success is True

        call_args = engine.metadata_store.create_memory.await_args
        created_item: MemoryItem = call_args[0][0]
        assert created_item.compressed is True
        assert created_item.expires_at is None
        assert created_item.importance == 0.5
