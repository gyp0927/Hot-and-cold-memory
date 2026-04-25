"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "adaptive-rag"}


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness probe."""
    return {"status": "ready"}
