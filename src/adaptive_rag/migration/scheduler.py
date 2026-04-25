"""Migration scheduler using APScheduler."""

from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)


class MigrationScheduler:
    """Schedules periodic migration cycles."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        self._job = None

    def start(self, migration_callback) -> None:
        """Start the migration scheduler.

        Args:
            migration_callback: Async function to call for migration.
        """
        self._job = self.scheduler.add_job(
            migration_callback,
            trigger=IntervalTrigger(
                minutes=self.settings.MIGRATION_INTERVAL_MINUTES,
            ),
            id="migration_cycle",
            name="Tier Migration Cycle",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info(
            "migration_scheduler_started",
            interval_minutes=self.settings.MIGRATION_INTERVAL_MINUTES,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("migration_scheduler_stopped")

    async def trigger_now(self) -> None:
        """Trigger migration immediately."""
        if self._job:
            self._job.modify(next_run_time=datetime.utcnow())
