"""Text chunking strategies."""

import re
from dataclasses import dataclass
from typing import Iterator
import uuid

from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk from a document."""

    text: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    tags: list[str] | None = None


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    # Split on multiple newlines or whitespace boundaries
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _split_paragraph(paragraph: str, max_size: int, overlap: int) -> list[str]:
    """Split a paragraph into chunks of max_size with overlap."""
    if len(paragraph) <= max_size:
        return [paragraph]

    chunks = []
    start = 0

    while start < len(paragraph):
        end = min(start + max_size, len(paragraph))

        # Try to find a natural boundary (period, newline, space)
        if end < len(paragraph):
            # Look for sentence boundary first
            for boundary_char in ["\n", "。", ".", "?", "!", " "]:
                boundary = paragraph.rfind(boundary_char, start, end)
                if boundary != -1 and boundary > start + max_size // 2:
                    end = boundary + 1
                    break

        chunks.append(paragraph[start:end].strip())
        start = end - overlap if end < len(paragraph) else end

    return chunks


class RecursiveChunker:
    """Recursive text chunker that respects natural boundaries.

    Strategy:
    1. Split into paragraphs
    2. For large paragraphs, split into sentences/clauses
    3. Merge small adjacent chunks to reach target size
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        document_id: uuid.UUID,
        tags: list[str] | None = None,
    ) -> list[Chunk]:
        """Chunk text into pieces.

        Args:
            text: Full document text.
            document_id: Parent document ID.
            tags: Optional tags for chunks.

        Returns:
            List of chunks.
        """
        paragraphs = _split_into_paragraphs(text)
        all_chunks: list[Chunk] = []
        global_char = 0

        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph:
                continue

            para_chunks = _split_paragraph(
                paragraph,
                self.chunk_size,
                self.chunk_overlap,
            )

            for sub_idx, sub_text in enumerate(para_chunks):
                chunk = Chunk(
                    text=sub_text,
                    chunk_id=uuid.uuid4(),
                    document_id=document_id,
                    index=len(all_chunks),
                    start_char=global_char,
                    end_char=global_char + len(sub_text),
                    tags=tags or [],
                )
                all_chunks.append(chunk)
                global_char += len(sub_text)

        logger.info(
            "chunking_complete",
            document_id=str(document_id),
            paragraphs=len(paragraphs),
            chunks=len(all_chunks),
        )
        return all_chunks


class FixedSizeChunker:
    """Simple fixed-size chunker with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        document_id: uuid.UUID,
        tags: list[str] | None = None,
    ) -> list[Chunk]:
        """Chunk text into fixed-size pieces."""
        chunks: list[Chunk] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=uuid.uuid4(),
                        document_id=document_id,
                        index=len(chunks),
                        start_char=start,
                        end_char=end,
                        tags=tags or [],
                    )
                )

            start = end - self.chunk_overlap if end < len(text) else end

        return chunks
