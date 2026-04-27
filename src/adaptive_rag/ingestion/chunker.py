"""Text chunking strategies."""

import asyncio
import re
from dataclasses import dataclass
from typing import Iterator
import uuid

from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)

_CHUNK_DELIMITER = "\n\n---CHUNK_DELIMITER---\n\n"


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

    async def chunk(
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

    async def chunk(
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


class LLMChunker:
    """Semantic chunker powered by an LLM.

    Sends the full document to an LLM and asks it to split the text
    into coherent semantic units (paragraphs grouped by meaning).
    Much better quality than rule-based chunking for structured docs
    with headings, lists, and mixed content.
    """

    _PROMPT_TEMPLATE = """请将以下文本按语义主题进行分段。

要求：
1. 每个段落围绕一个完整的主题或意思，内容要连贯完整
2. 不要把标题单独分成一段，标题必须和对应的内容合并在一起
3. 相近的小主题尽量合并成一个段落，避免过度碎片化
4. 每个段落长度在 200-800 字之间（中文）
5. 保持原文的完整性，不要遗漏或修改任何内容
6. 直接输出分段后的文本，段落之间用以下分隔符隔开：

---CHUNK_DELIMITER---

文本：
{text}"""

    def __init__(
        self,
        llm_client=None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._llm_client = llm_client

    async def chunk(
        self,
        text: str,
        document_id: uuid.UUID,
        tags: list[str] | None = None,
    ) -> list[Chunk]:
        """Chunk text using LLM semantic analysis."""
        if not text.strip():
            return []

        # Lazy-init LLM client
        if self._llm_client is None:
            from adaptive_rag.core.llm_client import LLMClient
            self._llm_client = LLMClient()

        prompt = self._PROMPT_TEMPLATE.format(text=text)

        try:
            raw = await self._llm_client.complete(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(
                "llm_chunking_failed",
                document_id=str(document_id),
                error=str(e),
                fallback="recursive",
            )
            # Fallback to recursive chunker on LLM failure
            fallback = RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return await fallback.chunk(text, document_id, tags)

        # Parse delimited output
        segments = [s.strip() for s in raw.split(_CHUNK_DELIMITER) if s.strip()]

        # If LLM didn't use delimiter, try paragraph boundaries
        if len(segments) <= 1:
            segments = _split_into_paragraphs(raw)

        # Filter out any segments that are just the delimiter or empty
        segments = [s for s in segments if s and s != "---CHUNK_DELIMITER---"]

        all_chunks: list[Chunk] = []
        global_char = 0

        for idx, seg_text in enumerate(segments):
            chunk = Chunk(
                text=seg_text,
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                index=idx,
                start_char=global_char,
                end_char=global_char + len(seg_text),
                tags=tags or [],
            )
            all_chunks.append(chunk)
            global_char += len(seg_text)

        logger.info(
            "llm_chunking_complete",
            document_id=str(document_id),
            segments=len(segments),
            chunks=len(all_chunks),
        )
        return all_chunks
