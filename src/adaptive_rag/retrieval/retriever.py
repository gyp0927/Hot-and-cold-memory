"""Unified retrieval interface."""

from typing import Any

from adaptive_rag.core.config import Tier
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.frequency.tracker import FrequencyTracker
from adaptive_rag.tiers.hot_tier import HotTier
from adaptive_rag.tiers.cold_tier import ColdTier

from .router import FrequencyRouter, RetrievalResult

logger = get_logger(__name__)


class UnifiedRetriever:
    """Unified retrieval interface that handles all retrieval operations.

    This is the main entry point for query operations, abstracting
    away the complexity of tier routing and frequency tracking.
    """

    def __init__(
        self,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        frequency_tracker: FrequencyTracker,
        embedder: Embedder | None = None,
    ) -> None:
        self.router = FrequencyRouter(
            hot_tier=hot_tier,
            cold_tier=cold_tier,
            frequency_tracker=frequency_tracker,
            embedder=embedder,
        )

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        tier: Tier | None = None,
        decompress: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Execute a query and retrieve relevant chunks.

        Args:
            query_text: User query.
            top_k: Number of results.
            tier: Force specific tier.
            decompress: Decompress cold chunks.
            filters: Metadata filters.

        Returns:
            Retrieval result.
        """
        return await self.router.route(
            query_text=query_text,
            top_k=top_k,
            tier_preference=tier,
            force_decompress=decompress,
            filters=filters,
        )
