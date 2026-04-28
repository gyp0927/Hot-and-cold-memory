"""Unit tests for metadata store."""

import uuid
from datetime import datetime

import pytest

from adaptive_rag.core.config import Tier
from adaptive_rag.storage.metadata_store.base import (
    ChunkMetadata,
    DocumentMetadata,
    QueryCluster,
)


class TestPostgresMetadataStore:
    """Test metadata store operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_chunk(self, metadata_store):
        """Test creating and retrieving a chunk."""
        chunk_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_id,
            tier=Tier.HOT,
            original_length=100,
            access_count=0,
            frequency_score=1.0,
        )
        await metadata_store.create_chunk(meta)

        retrieved = await metadata_store.get_chunk(chunk_id)
        assert retrieved is not None
        assert retrieved.chunk_id == chunk_id
        assert retrieved.tier == Tier.HOT
        assert retrieved.frequency_score == 1.0

    @pytest.mark.asyncio
    async def test_get_chunks_batch(self, metadata_store):
        """Test batch chunk retrieval."""
        doc_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4() for _ in range(5)]
        for cid in chunk_ids:
            await metadata_store.create_chunk(
                ChunkMetadata(
                    chunk_id=cid,
                    document_id=doc_id,
                    tier=Tier.HOT,
                    original_length=50,
                )
            )

        results = await metadata_store.get_chunks_batch(chunk_ids)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_update_chunks_batch(self, metadata_store):
        """Test batch chunk update."""
        doc_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4() for _ in range(3)]
        for cid in chunk_ids:
            await metadata_store.create_chunk(
                ChunkMetadata(
                    chunk_id=cid,
                    document_id=doc_id,
                    tier=Tier.HOT,
                    original_length=50,
                    frequency_score=0.0,
                )
            )

        updates = {cid: {"frequency_score": 0.8} for cid in chunk_ids}
        await metadata_store.update_chunks_batch(updates)

        for cid in chunk_ids:
            retrieved = await metadata_store.get_chunk(cid)
            assert retrieved.frequency_score == 0.8

    @pytest.mark.asyncio
    async def test_increment_access_batch(self, metadata_store):
        """Test batch access increment."""
        doc_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4() for _ in range(3)]
        for cid in chunk_ids:
            await metadata_store.create_chunk(
                ChunkMetadata(
                    chunk_id=cid,
                    document_id=doc_id,
                    tier=Tier.HOT,
                    original_length=50,
                    access_count=0,
                )
            )

        await metadata_store.increment_access(
            chunk_ids=chunk_ids,
            cluster_id=None,
            timestamp=datetime.utcnow(),
        )

        for cid in chunk_ids:
            retrieved = await metadata_store.get_chunk(cid)
            assert retrieved.access_count == 1

    @pytest.mark.asyncio
    async def test_get_document_by_hash(self, metadata_store):
        """Test document lookup by content hash."""
        doc_id = uuid.uuid4()
        await metadata_store.create_document(
            DocumentMetadata(
                document_id=doc_id,
                source_type="text",
                source_uri="test.txt",
                content_hash="abc123",
                total_chunks=5,
            )
        )

        found = await metadata_store.get_document_by_hash("abc123")
        assert found is not None
        assert found.document_id == doc_id

        not_found = await metadata_store.get_document_by_hash("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_count_chunks_by_tier(self, metadata_store):
        """Test counting chunks by tier."""
        doc_id = uuid.uuid4()
        for _ in range(3):
            await metadata_store.create_chunk(
                ChunkMetadata(
                    chunk_id=uuid.uuid4(),
                    document_id=doc_id,
                    tier=Tier.HOT,
                    original_length=50,
                )
            )
        for _ in range(2):
            await metadata_store.create_chunk(
                ChunkMetadata(
                    chunk_id=uuid.uuid4(),
                    document_id=doc_id,
                    tier=Tier.COLD,
                    original_length=50,
                )
            )

        hot_count = await metadata_store.count_chunks_by_tier(Tier.HOT)
        cold_count = await metadata_store.count_chunks_by_tier(Tier.COLD)
        assert hot_count == 3
        assert cold_count == 2

    @pytest.mark.asyncio
    async def test_get_clusters_batch(self, metadata_store):
        """Test batch cluster retrieval."""
        cluster_ids = [uuid.uuid4() for _ in range(3)]
        for cid in cluster_ids:
            await metadata_store.create_cluster(
                QueryCluster(
                    cluster_id=cid,
                    centroid=[0.1, 0.2, 0.3],
                    representative_query="test query",
                )
            )

        results = await metadata_store.get_clusters_batch(cluster_ids)
        assert len(results) == 3
