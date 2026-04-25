"""LLM-based compression engine for cold tier."""

import asyncio
import json
import re
from dataclasses import dataclass
import uuid

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.exceptions import CompressionError
from adaptive_rag.core.llm_client import LLMClient
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.chunker import Chunk

logger = get_logger(__name__)


@dataclass
class CompressedChunk:
    """A compressed chunk with preserved key information."""

    chunk_id: uuid.UUID
    summary_text: str
    key_entities: list[str]
    key_facts: list[str]
    compression_ratio: float


class CompressionEngine:
    """LLM-based compression that preserves semantic meaning."""

    COMPRESSION_PROMPT = """You are a semantic compression engine. Compress the following text into a dense summary that preserves all key information, entities, and relationships.

Requirements:
1. Preserve all named entities (people, organizations, locations, products)
2. Preserve all numerical data and statistics
3. Preserve key relationships and causal links
4. Maintain the original meaning and intent
5. Target approximately {target_ratio}% of original length

Original text:
{text}

Output as JSON:
{{
    "summary": "The compressed summary text...",
    "key_entities": ["Entity1", "Entity2"],
    "key_facts": ["Fact1", "Fact2"]
}}
"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = LLMClient()
        self.model = self.settings.COMPRESSION_MODEL

    async def compress(self, chunk: Chunk) -> CompressedChunk:
        """Compress a single chunk.

        Args:
            chunk: Chunk to compress.

        Returns:
            Compressed chunk.
        """
        target_ratio = int(self.settings.COLD_TIER_COMPRESSION_RATIO * 100)

        prompt = self.COMPRESSION_PROMPT.format(
            text=chunk.text,
            target_ratio=target_ratio,
        )

        try:
            # Use JSON response format for OpenAI, plain text for Anthropic
            if self.client._is_anthropic_format():
                response_text = await self.client.complete(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=self.settings.COMPRESSION_MAX_TOKENS,
                    temperature=0.0,
                )
                result = self._parse_json_response(response_text)
            else:
                response_text = await self.client.complete(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=self.settings.COMPRESSION_MAX_TOKENS,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response_text)

            summary = result.get("summary", "")

            compression_ratio = len(summary) / max(len(chunk.text), 1)

            logger.debug(
                "chunk_compressed",
                chunk_id=str(chunk.chunk_id),
                original_len=len(chunk.text),
                compressed_len=len(summary),
                ratio=compression_ratio,
            )

            return CompressedChunk(
                chunk_id=chunk.chunk_id,
                summary_text=summary,
                key_entities=result.get("key_entities", []),
                key_facts=result.get("key_facts", []),
                compression_ratio=compression_ratio,
            )

        except Exception as e:
            logger.error("compression_failed", chunk_id=str(chunk.chunk_id), error=str(e))
            raise CompressionError(f"Failed to compress chunk {chunk.chunk_id}: {e}") from e

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response text (for non-OpenAI models)."""
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: wrap entire response as summary
        return {"summary": text, "key_entities": [], "key_facts": []}

    async def compress_batch(self, chunks: list[Chunk]) -> list[CompressedChunk]:
        """Compress multiple chunks in parallel with rate limiting.

        Args:
            chunks: Chunks to compress.

        Returns:
            List of compressed chunks.
        """
        semaphore = asyncio.Semaphore(self.settings.COMPRESSION_BATCH_SIZE)

        async def compress_one(chunk: Chunk) -> CompressedChunk:
            async with semaphore:
                return await self.compress(chunk)

        return await asyncio.gather(*[
            compress_one(c) for c in chunks
        ])
