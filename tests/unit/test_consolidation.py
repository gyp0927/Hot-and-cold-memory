"""Tests for memory consolidation (deduplication + merging)."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from hot_and_cold_memory.consolidation.engine import ConsolidationEngine, ConsolidationResult
from hot_and_cold_memory.core.config import Tier
from hot_and_cold_memory.storage.metadata_store.base import MemoryItem


class TestConsolidationEngine:
    """Test consolidation logic."""

    @pytest.fixture
    def engine(self, monkeypatch):
        """Create engine with mocked dependencies."""
        monkeypatch.setenv("ENABLE_CONSOLIDATION", "true")
        monkeypatch.setenv("CONSOLIDATION_SIMILARITY_THRESHOLD", "0.92")
        monkeypatch.setenv("CONSOLIDATION_BATCH_SIZE", "50")
        monkeypatch.setenv("CONSOLIDATION_MAX_PAIRS_PER_RUN", "10")
        monkeypatch.setenv("CONSOLIDATION_MIN_CONTENT_LENGTH", "20")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        store = MagicMock()
        store.list_memories = AsyncMock(return_value=[])
        store.create_memory = AsyncMock()
        store.delete_memories = AsyncMock(return_value=2)

        embedder = MagicMock()
        embedder.embed_batch = AsyncMock(return_value=[])

        llm = MagicMock()
        llm.complete = AsyncMock(return_value="merged content from both memories")

        engine = ConsolidationEngine(
            metadata_store=store,
            embedder=embedder,
            llm_client=llm,
        )
        return engine

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, monkeypatch):
        """When consolidation is disabled, return empty result."""
        monkeypatch.setenv("ENABLE_CONSOLIDATION", "false")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        engine = ConsolidationEngine(metadata_store=MagicMock())
        result = await engine.consolidate()
        assert result == ConsolidationResult()

    @pytest.mark.asyncio
    async def test_insufficient_candidates(self, engine):
        """Less than 2 candidates means nothing to do."""
        engine.metadata_store.list_memories = AsyncMock(return_value=[
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.HOT,
                content="only one candidate memory exists",
                original_length=30,
            ),
        ])
        result = await engine.consolidate()
        assert result.candidates_checked == 1
        assert result.pairs_found == 0

    @pytest.mark.asyncio
    async def test_no_similar_pairs(self, engine):
        """Candidates with low similarity produce no pairs."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="Python is a programming language used for data science",
            original_length=50,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="The weather today is sunny and warm, perfect for a walk",
            original_length=50,
        )
        engine.metadata_store.list_memories = AsyncMock(return_value=[m1, m2])
        # Orthogonal embeddings -> similarity 0
        engine.embedder.embed_batch = AsyncMock(return_value=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        result = await engine.consolidate()
        assert result.candidates_checked == 2
        assert result.pairs_found == 0

    @pytest.mark.asyncio
    async def test_finds_and_merges_similar_pair(self, engine):
        """Highly similar candidates are merged."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="User prefers Python over JavaScript for backend work",
            original_length=50,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="User prefers Python over JavaScript for backend development",
            original_length=50,
        )
        engine.metadata_store.list_memories = AsyncMock(return_value=[m1, m2])
        # Nearly identical embeddings -> similarity ~1.0
        engine.embedder.embed_batch = AsyncMock(return_value=[
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
        ])

        result = await engine.consolidate()
        assert result.candidates_checked == 2
        assert result.pairs_found == 1
        assert len(result.merged) == 1
        assert len(result.deleted) == 2
        engine.metadata_store.create_memory.assert_awaited_once()
        engine.metadata_store.delete_memories.assert_awaited_once_with([m1.memory_id, m2.memory_id])

    @pytest.mark.asyncio
    async def test_skips_short_content(self, engine):
        """Memories below min content length are skipped."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="short",
            original_length=5,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="also short",
            original_length=10,
        )
        engine.metadata_store.list_memories = AsyncMock(return_value=[m1, m2])
        result = await engine.consolidate()
        assert result.candidates_checked == 0

    @pytest.mark.asyncio
    async def test_tier_filter(self, engine):
        """Tier filter restricts candidates."""
        hot = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="hot memory with sufficient length here",
            original_length=40,
        )
        cold = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.COLD,
            content="cold memory with sufficient length here",
            original_length=40,
        )
        engine.metadata_store.list_memories = AsyncMock(return_value=[hot, cold])
        engine.embedder.embed_batch = AsyncMock(return_value=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        result = await engine.consolidate(tier=Tier.HOT)
        assert result.candidates_checked == 1

    @pytest.mark.asyncio
    async def test_merge_uses_higher_importance(self, engine):
        """Merged memory inherits max importance from source pair."""
        m1 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="user prefers python for data science work",
            original_length=40,
            importance=0.3,
        )
        m2 = MemoryItem(
            memory_id=uuid.uuid4(),
            tier=Tier.HOT,
            content="user prefers python for data science tasks",
            original_length=40,
            importance=0.8,
        )
        engine.metadata_store.list_memories = AsyncMock(return_value=[m1, m2])
        engine.embedder.embed_batch = AsyncMock(return_value=[
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
        ])

        await engine.consolidate()
        call_args = engine.metadata_store.create_memory.await_args
        merged: MemoryItem = call_args[0][0]
        assert merged.importance == 0.8
        assert merged.memory_type == "consolidated"

    def test_find_similar_pairs(self, engine):
        """Direct test for pair-finding logic."""
        candidates = [
            MemoryItem(memory_id=uuid.uuid4(), tier=Tier.HOT, content="a", original_length=10),
            MemoryItem(memory_id=uuid.uuid4(), tier=Tier.HOT, content="b", original_length=10),
            MemoryItem(memory_id=uuid.uuid4(), tier=Tier.HOT, content="c", original_length=10),
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],  # very similar to 0
            [0.0, 1.0, 0.0],   # dissimilar
        ], dtype=np.float32)
        # normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        pairs = engine._find_similar_pairs(candidates, embeddings)
        assert len(pairs) == 1
        assert {pairs[0][0].memory_id, pairs[0][1].memory_id} == {
            candidates[0].memory_id,
            candidates[1].memory_id,
        }

    def test_find_similar_pairs_avoids_duplicates(self, engine):
        """Same unordered pair should not appear twice."""
        candidates = [
            MemoryItem(memory_id=uuid.uuid4(), tier=Tier.HOT, content="a", original_length=10),
            MemoryItem(memory_id=uuid.uuid4(), tier=Tier.HOT, content="b", original_length=10),
        ]
        embeddings = np.array([
            [1.0, 0.0],
            [0.99, 0.01],
        ], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        pairs = engine._find_similar_pairs(candidates, embeddings)
        assert len(pairs) == 1
