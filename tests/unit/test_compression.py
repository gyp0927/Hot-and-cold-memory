"""Unit tests for compression and decompression."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from adaptive_rag.ingestion.chunker import Chunk
from adaptive_rag.tiers.compression import CompressedChunk, CompressionEngine
from adaptive_rag.tiers.decompression import DecompressionEngine


class TestCompression:
    """Test compression functionality."""

    def test_compressed_chunk_dataclass(self):
        """Test CompressedChunk dataclass."""
        chunk_id = uuid.uuid4()
        comp = CompressedChunk(
            chunk_id=chunk_id,
            summary_text="This is a summary.",
            key_entities=["Entity1", "Entity2"],
            key_facts=["Fact1"],
            compression_ratio=0.25,
        )
        assert comp.chunk_id == chunk_id
        assert comp.summary_text == "This is a summary."
        assert len(comp.key_entities) == 2
        assert comp.compression_ratio == 0.25

    @pytest.mark.asyncio
    async def test_compression_engine_parse_json_response(self):
        """Test JSON response parsing."""
        engine = CompressionEngine()

        # Valid JSON
        result = engine._parse_json_response(
            '{"summary": "test", "key_entities": ["e1"], "key_facts": ["f1"]}'
        )
        assert result["summary"] == "test"

        # Malformed JSON fallback
        result = engine._parse_json_response("Just plain text")
        assert result["summary"] == "Just plain text"
        assert result["key_entities"] == []

    @pytest.mark.asyncio
    async def test_compression_batch(self, monkeypatch):
        """Test batch compression with mocked LLM."""
        engine = CompressionEngine()

        # Mock the compress method
        async def mock_compress(chunk):
            return CompressedChunk(
                chunk_id=chunk.chunk_id,
                summary_text="summary",
                key_entities=[],
                key_facts=[],
                compression_ratio=0.5,
            )

        monkeypatch.setattr(engine, "compress", mock_compress)

        chunks = [
            Chunk(chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), text="text 1"),
            Chunk(chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), text="text 2"),
        ]
        results = await engine.compress_batch(chunks)
        assert len(results) == 2
        assert all(r.summary_text == "summary" for r in results)


class TestChunkDataclass:
    """Test Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            text="Sample text content",
            index=0,
            tags=["test"],
        )
        assert chunk.text == "Sample text content"
        assert chunk.document_id == doc_id
        assert chunk.index == 0


class TestDecompressionEngine:
    """Test decompression engine."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from adaptive_rag.tiers.decompression import _cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == 1.0

        c = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, c) == 0.0

        # Mismatched lengths
        assert _cosine_similarity(a, [1.0, 0.0]) == 0.0

    def test_flag_for_review(self):
        """Test flagging chunks for review."""
        engine = DecompressionEngine()
        engine.flag_for_review("chunk-123")
        assert "chunk-123" in engine.flagged_chunk_ids
