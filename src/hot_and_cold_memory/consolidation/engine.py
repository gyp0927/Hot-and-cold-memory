"""Consolidation engine: deduplicate and merge semantically similar memories."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.core.llm_client import LLMClient
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.ingestion.embedder import Embedder
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore, MemoryItem

logger = get_logger(__name__)


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    merged: list[uuid.UUID] = field(default_factory=list)
    deleted: list[uuid.UUID] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    candidates_checked: int = 0
    pairs_found: int = 0


class ConsolidationEngine:
    """Detects duplicate memories by embedding cosine similarity and merges them via LLM."""

    def __init__(
        self,
        metadata_store: BaseMetadataStore,
        embedder: Embedder | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = get_settings()
        self.metadata_store = metadata_store
        self.embedder = embedder or Embedder()
        self.llm_client = llm_client or LLMClient()

    async def consolidate(self, tier: Tier | None = None) -> ConsolidationResult:
        """Run one consolidation cycle.

        Args:
            tier: If provided, only consolidate memories in this tier.

        Returns:
            Consolidation result.
        """
        if not self.settings.ENABLE_CONSOLIDATION:
            return ConsolidationResult()

        result = ConsolidationResult()

        # 1. Fetch candidate memories
        candidates = await self._fetch_candidates(tier)
        result.candidates_checked = len(candidates)
        if len(candidates) < 2:
            logger.info("consolidation_insufficient_candidates", count=len(candidates))
            return result

        # 2. Embed contents
        embeddings = await self._embed_candidates(candidates)

        # 3. Find similar pairs
        pairs = self._find_similar_pairs(candidates, embeddings)
        result.pairs_found = len(pairs)
        if not pairs:
            logger.info("consolidation_no_similar_pairs")
            return result

        # 4. Merge each pair
        for mem_a, mem_b in pairs[: self.settings.CONSOLIDATION_MAX_PAIRS_PER_RUN]:
            try:
                merged_id = await self._merge_pair(mem_a, mem_b)
                if merged_id:
                    result.merged.append(merged_id)
                    result.deleted.extend([mem_a.memory_id, mem_b.memory_id])
            except Exception as e:
                result.errors.append(f"Merge failed for {mem_a.memory_id}-{mem_b.memory_id}: {e}")
                logger.warning(
                    "consolidation_merge_failed",
                    a=str(mem_a.memory_id),
                    b=str(mem_b.memory_id),
                    error=str(e),
                )

        logger.info(
            "consolidation_complete",
            candidates=result.candidates_checked,
            pairs=result.pairs_found,
            merged=len(result.merged),
            deleted=len(result.deleted),
            errors=len(result.errors),
        )
        return result

    async def _fetch_candidates(self, tier: Tier | None) -> list[MemoryItem]:
        """Fetch memories eligible for consolidation."""
        all_memories: list[MemoryItem] = []
        offset = 0
        batch = self.settings.CONSOLIDATION_BATCH_SIZE
        while True:
            page = await self.metadata_store.list_memories(
                limit=batch,
                offset=offset,
            )
            if not page:
                break
            all_memories.extend(page)
            offset += batch
            if len(page) < batch:
                break

        filtered = [
            m
            for m in all_memories
            if len(m.content) >= self.settings.CONSOLIDATION_MIN_CONTENT_LENGTH
            and (tier is None or m.tier == tier)
        ]
        return filtered

    async def _embed_candidates(self, candidates: list[MemoryItem]) -> np.ndarray:
        """Embed candidate contents."""
        texts = [m.content for m in candidates]
        embeddings = await self.embedder.embed_batch(texts)
        arr = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _find_similar_pairs(
        self,
        candidates: list[MemoryItem],
        embeddings: np.ndarray,
    ) -> list[tuple[MemoryItem, MemoryItem]]:
        """Find candidate pairs with cosine similarity above threshold."""
        threshold = self.settings.CONSOLIDATION_SIMILARITY_THRESHOLD
        n = len(candidates)
        pairs: list[tuple[MemoryItem, MemoryItem]] = []
        seen: set[tuple[uuid.UUID, uuid.UUID]] = set()

        # Pairwise cosine similarity via matrix multiplication (normalized embeddings)
        sim_matrix = embeddings @ embeddings.T

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    pair_key = tuple(sorted([candidates[i].memory_id, candidates[j].memory_id]))  # type: ignore[arg-type]
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append((candidates[i], candidates[j]))
        return pairs

    async def _merge_pair(self, mem_a: MemoryItem, mem_b: MemoryItem) -> uuid.UUID | None:
        """Merge two memories into one using LLM.

        Returns:
            ID of the newly created merged memory, or None if skipped.
        """
        merged_content = await self._llm_merge(mem_a.content, mem_b.content)
        if not merged_content or len(merged_content) < 10:
            logger.warning(
                "consolidation_empty_merge",
                a=str(mem_a.memory_id),
                b=str(mem_b.memory_id),
            )
            return None

        now = datetime.now(timezone.utc)
        merged_id = uuid.uuid4()
        merged = MemoryItem(
            memory_id=merged_id,
            tier=mem_a.tier,
            content=merged_content,
            original_length=len(merged_content),
            memory_type="consolidated",
            source=mem_a.source or mem_b.source,
            importance=max(mem_a.importance, mem_b.importance),
            access_count=max(mem_a.access_count, mem_b.access_count),
            frequency_score=max(mem_a.frequency_score, mem_b.frequency_score),
            created_at=min(mem_a.created_at, mem_b.created_at),
            updated_at=now,
            last_accessed_at=max(
                mem_a.last_accessed_at or mem_a.created_at,
                mem_b.last_accessed_at or mem_b.created_at,
            ),
            tags=list(set(mem_a.tags + mem_b.tags)),
            attributes={**mem_b.attributes, **mem_a.attributes},
        )

        # Store merged memory
        await self.metadata_store.create_memory(merged)

        # Delete originals
        await self.metadata_store.delete_memories([mem_a.memory_id, mem_b.memory_id])

        logger.info(
            "memory_consolidated",
            merged=str(merged_id),
            a=str(mem_a.memory_id),
            b=str(mem_b.memory_id),
            similarity_hint="high",
        )
        return merged_id

    async def _llm_merge(self, content_a: str, content_b: str) -> str:
        """Call LLM to merge two memory contents."""
        prompt = (
            "Merge the following two memory observations into a single coherent observation. "
            "Preserve all important facts. If there are contradictions, prefer the more specific one.\n\n"
            f"Memory A: {content_a}\n\n"
            f"Memory B: {content_b}\n\n"
            "Merged memory:"
        )
        merged = await self.llm_client.complete(
            prompt,
            model=self.settings.COMPRESSION_MODEL,
            max_tokens=self.settings.COMPRESSION_MAX_TOKENS,
            temperature=0.0,
        )
        return merged.strip()
