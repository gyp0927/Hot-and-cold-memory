"""Migration threshold policies."""

from dataclasses import dataclass

from adaptive_rag.core.config import get_settings


@dataclass
class MigrationThresholds:
    """Thresholds for tier migration decisions."""

    hot_to_cold: float
    cold_to_hot: float
    batch_size: int
    max_concurrent: int


class MigrationPolicy:
    """Configurable migration policy."""

    def __init__(self) -> None:
        settings = get_settings()
        self.thresholds = MigrationThresholds(
            hot_to_cold=settings.HOT_TO_COLD_THRESHOLD,
            cold_to_hot=settings.COLD_TO_HOT_THRESHOLD,
            batch_size=settings.MIGRATION_BATCH_SIZE,
            max_concurrent=settings.MIGRATION_MAX_CONCURRENT,
        )

    def should_demote(self, frequency_score: float) -> bool:
        """Check if a hot chunk should be demoted to cold.

        Args:
            frequency_score: Current frequency score.

        Returns:
            True if chunk should be demoted.
        """
        return frequency_score <= self.thresholds.hot_to_cold

    def should_promote(self, frequency_score: float) -> bool:
        """Check if a cold chunk should be promoted to hot.

        Args:
            frequency_score: Current frequency score.

        Returns:
            True if chunk should be promoted.
        """
        return frequency_score >= self.thresholds.cold_to_hot
