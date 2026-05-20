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
        import dataclasses

        # Apply tier-specific score adjustments
        adjusted_hot = [
            dataclasses.replace(r, score=r.score * 1.05)  # Slight boost for hot tier
            for r in hot_results
        ]

        # Cold tier summaries may have lower semantic similarity
        adjusted_cold = [
            dataclasses.replace(r, score=r.score * 0.95)  # Slight penalty for summaries
            for r in cold_results
        ]

        # Deduplicate before sorting: pick hot over cold if same ID appears in both
        merged_by_id: dict = {}
        for r in adjusted_hot + adjusted_cold:
            if r.memory_id not in merged_by_id:
                merged_by_id[r.memory_id] = r
            else:
                # Keep the higher adjusted score
                if r.score > merged_by_id[r.memory_id].score:
                    merged_by_id[r.memory_id] = r

        # Sort by adjusted score and truncate
        deduped = sorted(merged_by_id.values(), key=lambda r: r.score, reverse=True)

        return deduped[:top_k]
