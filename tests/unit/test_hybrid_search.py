"""Tests for hybrid search (vector + keyword with RRF fusion)."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from hot_and_cold_memory.core.config import Tier
from hot_and_cold_memory.retrieval.hybrid import HybridRanker
from hot_and_cold_memory.retrieval.router import FrequencyRouter
from hot_and_cold_memory.storage.metadata_store.base import MemoryItem
from hot_and_cold_memory.tiers.base import RetrievedMemory


class TestHybridRanker:
    """Test RRF fusion logic."""

    def test_fuse_boosts_overlapping_results(self):
        """When a memory appears in both vector and keyword, its RRF score is higher."""
        ranker = HybridRanker(k=60)
        mem_id = uuid.uuid4()
        vector_results = [
            RetrievedMemory(
                memory_id=mem_id,
                content="python programming",
                score=0.9,
                tier=Tier.HOT,
                is_decompressed=False,
            ),
        ]
        keyword_results = [(mem_id, "python programming")]
        fused = ranker.fuse(vector_results, keyword_results, top_k=10)
        assert len(fused) == 1
        # RRF score should be sum of both contributions
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 61, rel=1e-6)

    def test_fuse_includes_keyword_only(self):
        """Keyword-only results should appear in fused output."""
        ranker = HybridRanker(k=60)
        vid = uuid.uuid4()
        kid = uuid.uuid4()
        vector_results = [
            RetrievedMemory(
                memory_id=vid,
                content="vector result",
                score=0.9,
                tier=Tier.HOT,
                is_decompressed=False,
            ),
        ]
        keyword_results = [(kid, "keyword result")]
        fused = ranker.fuse(vector_results, keyword_results, top_k=10)
        assert len(fused) == 2
        ids = {m.memory_id for m in fused}
        assert vid in ids
        assert kid in ids

    def test_fuse_respects_top_k(self):
        """Fused output should respect top_k limit."""
        ranker = HybridRanker(k=60)
        vector_results = [
            RetrievedMemory(
                memory_id=uuid.uuid4(),
                content=f"vec {i}",
                score=0.9,
                tier=Tier.HOT,
                is_decompressed=False,
            )
            for i in range(10)
        ]
        keyword_results = [(uuid.uuid4(), f"kw {i}") for i in range(10)]
        fused = ranker.fuse(vector_results, keyword_results, top_k=5)
        assert len(fused) == 5

    def test_fuse_empty_vector(self):
        """Keyword results alone should still be returned."""
        ranker = HybridRanker(k=60)
        kid = uuid.uuid4()
        fused = ranker.fuse([], [(kid, "keyword only")], top_k=10)
        assert len(fused) == 1
        assert fused[0].memory_id == kid

    def test_fuse_empty_keyword(self):
        """Vector results alone should still be returned."""
        ranker = HybridRanker(k=60)
        vid = uuid.uuid4()
        vector_results = [
            RetrievedMemory(
                memory_id=vid,
                content="vector only",
                score=0.9,
                tier=Tier.HOT,
                is_decompressed=False,
            ),
        ]
        fused = ranker.fuse(vector_results, [], top_k=10)
        assert len(fused) == 1
        assert fused[0].memory_id == vid


class TestFrequencyRouterHybrid:
    """Test FrequencyRouter hybrid search integration."""

    @pytest.fixture
    def router(self, monkeypatch):
        """Create router with mocked tiers and metadata store."""
        monkeypatch.setenv("ENABLE_HYBRID_SEARCH", "true")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        hot_tier = MagicMock()
        cold_tier = MagicMock()
        ft = MagicMock()
        ft.get_topic_frequency = AsyncMock(return_value=MagicMock(frequency=0.5, access_count=0))
        store = MagicMock()
        store.search_by_keyword = AsyncMock(return_value=[])

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2])

        router = FrequencyRouter(
            hot_tier=hot_tier,
            cold_tier=cold_tier,
            frequency_tracker=ft,
            embedder=embedder,
            metadata_store=store,
        )
        return router

    @pytest.mark.asyncio
    async def test_hybrid_calls_keyword_search(self, router):
        """When use_hybrid=True, keyword search is invoked."""
        router.hot_tier.retrieve = AsyncMock(return_value=[])
        router.cold_tier.retrieve = AsyncMock(return_value=[])

        await router.route("python tutorial", use_hybrid=True)

        router.metadata_store.search_by_keyword.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_hybrid_skips_keyword_search(self, router):
        """When use_hybrid=False, keyword search is skipped."""
        router.hot_tier.retrieve = AsyncMock(return_value=[])
        router.cold_tier.retrieve = AsyncMock(return_value=[])

        await router.route("python tutorial", use_hybrid=False)

        router.metadata_store.search_by_keyword.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_hybrid_disabled_globally(self, monkeypatch):
        """When ENABLE_HYBRID_SEARCH=false, keyword search is skipped even if use_hybrid=True."""
        monkeypatch.setenv("ENABLE_HYBRID_SEARCH", "false")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        hot_tier = MagicMock()
        cold_tier = MagicMock()
        ft = MagicMock()
        ft.get_topic_frequency = AsyncMock(return_value=MagicMock(frequency=0.5, access_count=0))
        store = MagicMock()
        store.search_by_keyword = AsyncMock(return_value=[])
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2])

        router = FrequencyRouter(
            hot_tier=hot_tier,
            cold_tier=cold_tier,
            frequency_tracker=ft,
            embedder=embedder,
            metadata_store=store,
        )
        hot_tier.retrieve = AsyncMock(return_value=[])
        cold_tier.retrieve = AsyncMock(return_value=[])

        await router.route("python tutorial", use_hybrid=True)
        store.search_by_keyword.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_hybrid_no_metadata_store(self, monkeypatch):
        """If metadata_store is None, hybrid search is skipped."""
        monkeypatch.setenv("ENABLE_HYBRID_SEARCH", "true")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        hot_tier = MagicMock()
        cold_tier = MagicMock()
        ft = MagicMock()
        ft.get_topic_frequency = AsyncMock(return_value=MagicMock(frequency=0.5, access_count=0))
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2])

        router = FrequencyRouter(
            hot_tier=hot_tier,
            cold_tier=cold_tier,
            frequency_tracker=ft,
            embedder=embedder,
            metadata_store=None,
        )
        hot_tier.retrieve = AsyncMock(return_value=[])
        cold_tier.retrieve = AsyncMock(return_value=[])

        result = await router.route("python tutorial", use_hybrid=True)
        assert result.routing_strategy.value == "both"


class TestMetadataStoreKeywordSearch:
    """Test PostgresMetadataStore.search_by_keyword."""

    @pytest.mark.asyncio
    async def test_keyword_match_single_term(self, metadata_store):
        """Single keyword should match content."""
        await metadata_store.create_memory(
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.HOT,
                content="Python is great for data science",
                original_length=30,
            )
        )
        await metadata_store.create_memory(
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.HOT,
                content="JavaScript is for web development",
                original_length=30,
            )
        )
        results = await metadata_store.search_by_keyword("Python", limit=10)
        assert len(results) == 1
        assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_keyword_match_multiple_terms(self, metadata_store):
        """Multiple keywords should all match (AND logic)."""
        await metadata_store.create_memory(
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.HOT,
                content="Python data science tutorial",
                original_length=30,
            )
        )
        await metadata_store.create_memory(
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.HOT,
                content="Python tutorial",
                original_length=20,
            )
        )
        results = await metadata_store.search_by_keyword("Python data", limit=10)
        assert len(results) == 1
        assert "Python" in results[0].content
        assert "data" in results[0].content

    @pytest.mark.asyncio
    async def test_keyword_tier_filter(self, metadata_store):
        """Tier filter should restrict results."""
        await metadata_store.create_memory(
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.HOT,
                content="Python hot memory",
                original_length=20,
            )
        )
        await metadata_store.create_memory(
            MemoryItem(
                memory_id=uuid.uuid4(),
                tier=Tier.COLD,
                content="Python cold memory",
                original_length=20,
            )
        )
        hot_results = await metadata_store.search_by_keyword("Python", tier=Tier.HOT, limit=10)
        assert len(hot_results) == 1
        assert hot_results[0].tier == Tier.HOT
