"""Result merging and re-ranking across tiers."""

from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.tiers.base import RetrievedMemory

logger = get_logger(__name__)


class ResultRanker:
    """Merges and re-ranks results from hot and cold tiers."""

    def merge_and_rank(
        self,
        hot_results: list[RetrievedMemory],
        cold_results: list[RetrievedMemory],
        top_k: int,
    ) -> list[RetrievedMemory]:
        """Merge results from both tiers and re-rank.

        Hot tier results are slightly boosted since they contain
        full text and are generally more reliable.

        Args:
            hot_results: Results from hot tier.
            cold_results: Results from cold tier.
            top_k: Maximum number of results to return.

        Returns:
            Merged and ranked results.
        """
        # Apply tier-specific score adjustments
        adjusted_hot = [
            RetrievedMemory(
                memory_id=r.memory_id,
                content=r.content,
                score=r.score * 1.05,  # Slight boost for hot tier
                tier=r.tier,
                is_decompressed=r.is_decompressed,
                access_count=r.access_count,
                frequency_score=r.frequency_score,
                memory_type=r.memory_type,
                embedding=r.embedding,
                metadata=r.metadata,
            )
            for r in hot_results
        ]

        # Cold tier summaries may have lower semantic similarity
        # so we keep their scores as-is
        adjusted_cold = [
            RetrievedMemory(
                memory_id=r.memory_id,
                content=r.content,
                score=r.score * 0.95,  # Slight penalty for summaries
                tier=r.tier,
                is_decompressed=r.is_decompressed,
                access_count=r.access_count,
                frequency_score=r.frequency_score,
                memory_type=r.memory_type,
                embedding=r.embedding,
                metadata=r.metadata,
            )
            for r in cold_results
        ]

        # Merge and sort by adjusted score
        all_results = adjusted_hot + adjusted_cold
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Remove duplicates (same memory might appear in both tiers)
        # Build an index for O(1) lookup instead of O(n) scan
        original_by_id = {r.memory_id: r for r in hot_results + cold_results}
        seen: set = set()
        deduped = []
        for r in all_results:
            if r.memory_id not in seen:
                seen.add(r.memory_id)
                # Restore original score from the actual result
                deduped.append(original_by_id.get(r.memory_id, r))

        return deduped[:top_k]
