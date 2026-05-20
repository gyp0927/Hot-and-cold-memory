"""Migration threshold policies."""

from dataclasses import dataclass

from hot_and_cold_memory.core.config import get_settings


@dataclass
class MigrationThresholds:
    """Thresholds for tier migration decisions."""

    hot_to_cold: float
    cold_to_hot: float
    hot_access_count: int
    batch_size: int
    max_concurrent: int


class MigrationPolicy:
    """Configurable migration policy."""

    def __init__(self) -> None:
        settings = get_settings()
        self.thresholds = MigrationThresholds(
            hot_to_cold=settings.HOT_TO_COLD_THRESHOLD,
            cold_to_hot=settings.COLD_TO_HOT_THRESHOLD,
            hot_access_count=settings.HOT_ACCESS_COUNT_THRESHOLD,
            batch_size=settings.MIGRATION_BATCH_SIZE,
            max_concurrent=settings.MIGRATION_MAX_CONCURRENT,
        )

    def should_demote(self, frequency_score: float, importance: float = 0.5) -> bool:
        """Check if a hot chunk should be demoted to cold.

        High-importance memories are protected by a lower effective threshold
        (harder to demote). The protection curve is:
          importance >= 0.8  → threshold reduced by 0.15
          importance >= 0.6  → threshold reduced by 0.08
          importance <  0.6  → no change

        Args:
            frequency_score: Current frequency score.
            importance: Memory importance (0-1).

        Returns:
            True if chunk should be demoted.
        """
        threshold = self.thresholds.hot_to_cold
        if importance >= 0.8:
            threshold -= 0.15
        elif importance >= 0.6:
            threshold -= 0.08
        # Clamp so we never drop below 0.05 (still possible to demote)
        threshold = max(0.05, threshold)
        return frequency_score <= threshold

    def should_promote(self, frequency_score: float, access_count: int) -> bool:
        """Check if a cold chunk should be promoted to hot.

        Args:
            frequency_score: Current frequency score.
            access_count: Total historical access count.

        Returns:
            True if chunk should be promoted.
        """
        if frequency_score >= self.thresholds.cold_to_hot:
            return True
        if access_count >= self.thresholds.hot_access_count:
            return True
        return False
