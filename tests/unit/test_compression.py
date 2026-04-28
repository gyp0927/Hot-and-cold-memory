"""Unit tests for compression and decompression."""

import uuid

import pytest

from hot_and_cold_memory.tiers.base import MemoryEntry
from hot_and_cold_memory.tiers.compression import CompressedChunk, CompressionEngine
from hot_and_cold_memory.tiers.decompression import DecompressionEngine


class TestCompression:
    """Test compression functionality."""

    def test_compressed_chunk_dataclass(self):
        """Test CompressedChunk dataclass."""
        memory_id = uuid.uuid4()
        comp = CompressedChunk(
            chunk_id=memory_id,
            summary_text="This is a summary.",
            key_entities=["Entity1", "Entity2"],
            key_facts=["Fact1"],
            compression_ratio=0.25,
        )
        assert comp.chunk_id == memory_id
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
        async def mock_compress(memory):
            return CompressedChunk(
                chunk_id=memory.memory_id,
                summary_text="summary",
                key_entities=[],
                key_facts=[],
                compression_ratio=0.5,
            )

        monkeypatch.setattr(engine, "compress", mock_compress)

        memories = [
            MemoryEntry(memory_id=uuid.uuid4(), content="text 1"),
            MemoryEntry(memory_id=uuid.uuid4(), content="text 2"),
        ]
        results = await engine.compress_batch(memories)
        assert len(results) == 2
        assert all(r.summary_text == "summary" for r in results)


class TestMemoryEntryDataclass:
    """Test MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        memory_id = uuid.uuid4()
        entry = MemoryEntry(
            memory_id=memory_id,
            content="Sample memory content",
            tags=["test"],
        )
        assert entry.content == "Sample memory content"
        assert entry.tags == ["test"]


class TestDecompressionEngine:
    """Test decompression engine."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from hot_and_cold_memory.tiers.decompression import _cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == 1.0

        c = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, c) == 0.0

        # Mismatched lengths
        assert _cosine_similarity(a, [1.0, 0.0]) == 0.0

    def test_flag_for_review(self):
        """Test flagging memories for review."""
        engine = DecompressionEngine()
        engine.flag_for_review("memory-123")
        assert "memory-123" in engine.flagged_chunk_ids
