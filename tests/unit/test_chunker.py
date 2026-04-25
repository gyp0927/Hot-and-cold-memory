"""Unit tests for text chunking."""

import uuid

import pytest

from adaptive_rag.ingestion.chunker import RecursiveChunker, FixedSizeChunker


class TestRecursiveChunker:
    """Test recursive chunking strategy."""

    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk."""
        chunker = RecursiveChunker(chunk_size=1000)
        doc_id = uuid.uuid4()
        text = "This is a short text."

        chunks = chunker.chunk(text=text, document_id=doc_id)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].document_id == doc_id

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        doc_id = uuid.uuid4()
        text = "A" * 200

        chunks = chunker.chunk(text=text, document_id=doc_id)

        assert len(chunks) > 1
        # Each chunk should be reasonable size
        for chunk in chunks:
            assert len(chunk.text) <= 60  # Allow some overflow for boundaries

    def test_paragraph_boundaries(self):
        """Chunker should respect paragraph boundaries when possible."""
        chunker = RecursiveChunker(chunk_size=100)
        doc_id = uuid.uuid4()
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = chunker.chunk(text=text, document_id=doc_id)

        # Should create chunks that align with paragraphs
        assert len(chunks) >= 1


class TestFixedSizeChunker:
    """Test fixed-size chunking strategy."""

    def test_exact_size_chunks(self):
        """Chunks should be approximately the target size."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        doc_id = uuid.uuid4()
        text = "A" * 150

        chunks = chunker.chunk(text=text, document_id=doc_id)

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
