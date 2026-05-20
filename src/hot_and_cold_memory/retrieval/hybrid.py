"""Hybrid search: fuse vector similarity and keyword search with RRF."""

import uuid
from collections import defaultdict
from dataclasses import dataclass

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.tiers.base import RetrievedMemory

logger = get_logger(__name__)


@dataclass(frozen=True)
class _RankedItem:
    memory_id: uuid.UUID
    score: float
    source: str  # "vector" or "keyword"


class HybridRanker:
    """Reciprocal Rank Fusion (RRF) ranker for combining vector + keyword results."""

    def __init__(self, k: int | None = None) -> None:
        self.k = k if k is not None else get_settings().HYBRID_RRF_K

    def fuse(
        self,
        vector_results: list[RetrievedMemory],
        keyword_results: list[tuple[uuid.UUID, str]],
        top_k: int = 10,
    ) -> list[RetrievedMemory]:
        """Fuse vector and keyword results using RRF.

        Args:
            vector_results: Ranked list from vector similarity search.
            keyword_results: Ranked list from keyword search as (memory_id, content) tuples.
            top_k: Number of top fused results to return.

        Returns:
            Re-ranked list of RetrievedMemory with updated scores.
        """
        scores: defaultdict[uuid.UUID, float] = defaultdict(float)
        content_map: dict[uuid.UUID, str] = {}
        tier_map: dict[uuid.UUID, str] = {}
        access_count_map: dict[uuid.UUID, int] = {}
        freq_score_map: dict[uuid.UUID, float] = {}
        memory_type_map: dict[uuid.UUID, str] = {}

        # Vector scores by rank
        for rank, mem in enumerate(vector_results):
            scores[mem.memory_id] += 1.0 / (self.k + rank + 1)
            content_map[mem.memory_id] = mem.content
            tier_map[mem.memory_id] = mem.tier.value
            access_count_map[mem.memory_id] = mem.access_count
            freq_score_map[mem.memory_id] = mem.frequency_score
            memory_type_map[mem.memory_id] = mem.memory_type

        # Keyword scores by rank
        for rank, (mid, content) in enumerate(keyword_results):
            scores[mid] += 1.0 / (self.k + rank + 1)
            if mid not in content_map:
                content_map[mid] = content

        # Sort by fused score descending
        sorted_ids = sorted(scores.keys(), key=lambda mid: scores[mid], reverse=True)

        fused: list[RetrievedMemory] = []
        for mid in sorted_ids[:top_k]:
            # Determine tier if known, default to cold for keyword-only hits
            tier_val = tier_map.get(mid, "cold")
            from hot_and_cold_memory.core.config import Tier

            fused.append(
                RetrievedMemory(
                    memory_id=mid,
                    content=content_map.get(mid, ""),
                    score=scores[mid],
                    tier=Tier(tier_val),
                    is_decompressed=False,
                    access_count=access_count_map.get(mid, 0),
                    frequency_score=freq_score_map.get(mid, 0.0),
                    memory_type=memory_type_map.get(mid, "observation"),
                )
            )

        logger.info(
            "hybrid_fuse_complete",
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            fused_count=len(fused),
            k=self.k,
        )
        return fused
