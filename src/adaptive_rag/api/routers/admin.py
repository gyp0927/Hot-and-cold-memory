"""Admin and configuration endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from adaptive_rag.core.logging import get_logger
from adaptive_rag.migration.engine import MigrationEngine, MigrationReport

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

# Global migration engine
_migration_engine: MigrationEngine | None = None


def set_migration_engine(engine: MigrationEngine) -> None:
    """Set the global migration engine."""
    global _migration_engine
    _migration_engine = engine


class MigrationTriggerResponse(BaseModel):
    """Migration trigger response."""

    success: bool
    hot_to_cold: int
    cold_to_hot: int
    errors: list[str]
    duration_seconds: float


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


@router.get("/stats")
async def get_stats() -> dict:
    """Get system statistics."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
