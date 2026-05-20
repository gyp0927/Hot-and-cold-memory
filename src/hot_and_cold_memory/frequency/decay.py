"""Exponential time decay for frequency scores.

Recent accesses weighted higher; old accesses fade over time.
"""

import math
from datetime import datetime, timezone

from hot_and_cold_memory.core.config import get_settings


class DecayEngine:
    """Implements exponential time decay for frequency scores."""

    # Normalization denominator for log1p(access_count) in min_score
    _ACCESS_LOG_DENOMINATOR: float = 6.0
    # Score ceiling for normalization in compute_score
    _SCORE_CEILING: float = 10.0
    # Weights for compute_score components
    _ACCESS_WEIGHT: float = 0.4
    _RECENCY_WEIGHT: float = 0.3
    _CLUSTER_WEIGHT: float = 0.3
    # Recency multiplier to keep it in comparable range with access_component
    _RECENCY_MULTIPLIER: float = 10.0

    def __init__(self) -> None:
        settings = get_settings()
        self.half_life_seconds = settings.DECAY_HALF_LIFE_HOURS * 3600
        self.decay_constant = math.log(2) / self.half_life_seconds

    @staticmethod
    def _to_aware(dt: datetime) -> datetime:
        """Ensure a datetime is timezone-aware (naive assumed UTC)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def apply_decay(
        self,
        base_score: float,
        last_accessed: datetime | None,
        access_count: int,
    ) -> float:
        """Apply time decay to a frequency score.

        Args:
            base_score: The stored frequency score.
            last_accessed: When the chunk was last accessed.
            access_count: Total number of accesses.

        Returns:
            Decayed score.
        """
        if last_accessed is None:
            return base_score

        elapsed = (datetime.now(timezone.utc) - self._to_aware(last_accessed)).total_seconds()
        decay_factor = math.exp(-self.decay_constant * elapsed)

        # Score decays but never below a minimum based on total accesses
        min_score = math.log1p(access_count) / self._ACCESS_LOG_DENOMINATOR

        return max(base_score * decay_factor, min_score)

    def compute_score(
        self,
        access_count: int,
        last_accessed: datetime | None,
        cluster_score: float,
    ) -> float:
        """Compute composite frequency score.

        Combines:
        - Individual chunk access count
        - Time since last access (decay)
        - Topic cluster popularity

        Args:
            access_count: Number of times chunk was accessed.
            last_accessed: Last access timestamp.
            created_at: Chunk creation timestamp.
            cluster_score: Topic cluster frequency score.

        Returns:
            Normalized frequency score in [0, 1].
        """
        # Base score from access count
        access_component = math.log1p(access_count)

        # Recency component (higher = more recent)
        if last_accessed:
            age_seconds = (datetime.now(timezone.utc) - self._to_aware(last_accessed)).total_seconds()
            recency = math.exp(-self.decay_constant * age_seconds)
        else:
            recency = 0.0

        # Cluster popularity component
        cluster_component = math.log1p(cluster_score)

        # Weighted combination
        score = (
            self._ACCESS_WEIGHT * access_component +
            self._RECENCY_WEIGHT * recency * self._RECENCY_MULTIPLIER +
            self._CLUSTER_WEIGHT * cluster_component
        )

        # Normalize to [0, 1]
        return min(score / self._SCORE_CEILING, 1.0)
