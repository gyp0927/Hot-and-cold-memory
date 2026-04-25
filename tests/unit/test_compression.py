"""Unit tests for compression and decompression."""

import uuid

import pytest

from adaptive_rag.ingestion.chunker import Chunk
from adaptive_rag.tiers.compression import CompressedChunk


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
