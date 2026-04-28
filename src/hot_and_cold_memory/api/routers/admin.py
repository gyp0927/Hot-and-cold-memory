"""Admin and configuration endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hot_and_cold_memory.core.config import Tier
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.migration.engine import MigrationEngine
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

# Global references
_migration_engine: MigrationEngine | None = None
_metadata_store: BaseMetadataStore | None = None


def set_migration_engine(engine: MigrationEngine) -> None:
    """Set the global migration engine."""
    global _migration_engine
    _migration_engine = engine


def set_metadata_store(store: BaseMetadataStore) -> None:
    """Set the global metadata store reference."""
    global _metadata_store
    _metadata_store = store


class MigrationTriggerResponse(BaseModel):
    """Migration trigger response."""

    success: bool
    hot_to_cold: int
    cold_to_hot: int
    errors: list[str]
    duration_seconds: float


class StatsResponse(BaseModel):
    """System statistics response."""

    total_memories: int
    hot_memories: int
    cold_memories: int
    total_clusters: int


@router.post("/migrate", response_model=MigrationTriggerResponse)
async def trigger_migration() -> MigrationTriggerResponse:
    """Trigger a manual migration cycle."""
    if not _migration_engine:
        raise HTTPException(status_code=503, detail="Migration engine not initialized")

    try:
        report = await _migration_engine.run_migration_cycle()

        duration = 0.0
        if report.completed_at and report.started_at:
            duration = (report.completed_at - report.started_at).total_seconds()

        return MigrationTriggerResponse(
            success=len(report.errors) == 0,
            hot_to_cold=len(report.hot_to_cold),
            cold_to_hot=len(report.cold_to_hot),
            errors=report.errors,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error("migration_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get system statistics."""
    if not _metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not initialized")

    try:
        hot_count = await _metadata_store.count_memories_by_tier(tier=Tier.HOT)
        cold_count = await _metadata_store.count_memories_by_tier(tier=Tier.COLD)
        clusters = await _metadata_store.get_all_clusters()

        return StatsResponse(
            total_memories=hot_count + cold_count,
            hot_memories=hot_count,
            cold_memories=cold_count,
            total_clusters=len(clusters),
        )
    except Exception as e:
        logger.error("stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
