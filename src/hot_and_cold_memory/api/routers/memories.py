"""Memory management endpoints."""

import uuid

from fastapi import APIRouter, HTTPException

from hot_and_cold_memory.api.schemas.memory import (
    MemoryCreateRequest,
    MemoryCreateResponse,
    MemoryDetailResponse,
    MemoryListResponse,
)
from hot_and_cold_memory.core.logging import get_logger
from hot_and_cold_memory.ingestion.pipeline import MemoryPipeline
from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore
from hot_and_cold_memory.storage.metadata_store.postgres_store import PostgresMetadataStore

logger = get_logger(__name__)
router = APIRouter(prefix="/memories", tags=["Memories"])

# Global instances
_pipeline: MemoryPipeline | None = None
_metadata_store: PostgresMetadataStore | None = None


def set_pipeline(pipeline: MemoryPipeline) -> None:
    """Set the global memory pipeline."""
    global _pipeline
    _pipeline = pipeline


def set_metadata_store(store: BaseMetadataStore) -> None:
    """Set the metadata store reference."""
    global _metadata_store
    _metadata_store = store


@router.post("", response_model=MemoryCreateResponse)
async def create_memory(request: MemoryCreateRequest) -> MemoryCreateResponse:
    """Write a new memory into the system."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = await _pipeline.write_memory(
            content=request.content,
            memory_type=request.memory_type,
            source=request.source,
            importance=request.importance,
            tags=request.tags,
            attributes=request.attributes,
        )
        return MemoryCreateResponse(
            memory_id=result.memory_id,
            status=result.status,
            tier=result.tier,
            message=result.error or "Memory written successfully",
        )
    except Exception as e:
        logger.error("create_memory_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("", response_model=MemoryListResponse)
async def list_memories(
    memory_type: str | None = None,
    source: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> MemoryListResponse:
    """List memories with optional filtering."""
    if not _metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not initialized")

    memories = await _metadata_store.list_memories(
        memory_type=memory_type,
        source=source,
        limit=limit,
        offset=offset,
    )
    return MemoryListResponse(
        memories=[
            MemoryDetailResponse(
                memory_id=m.memory_id,
                content=m.content,
                memory_type=m.memory_type,
                source=m.source,
                importance=m.importance,
                tier=m.tier.value,
                access_count=m.access_count,
                frequency_score=m.frequency_score,
                created_at=m.created_at.isoformat() if m.created_at else "",
                tags=m.tags,
            )
            for m in memories
        ],
        total=len(memories),
    )


@router.get("/{memory_id}", response_model=MemoryDetailResponse)
async def get_memory(memory_id: str) -> MemoryDetailResponse:
    """Get a single memory by ID."""
    if not _metadata_store:
        raise HTTPException(status_code=503, detail="Metadata store not initialized")

    try:
        mid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory ID format")

    m = await _metadata_store.get_memory(mid)
    if not m:
        raise HTTPException(status_code=404, detail="Memory not found")

    return MemoryDetailResponse(
        memory_id=m.memory_id,
        content=m.content,
        memory_type=m.memory_type,
        source=m.source,
        importance=m.importance,
        tier=m.tier.value,
        access_count=m.access_count,
        frequency_score=m.frequency_score,
        created_at=m.created_at.isoformat() if m.created_at else "",
        tags=m.tags,
    )


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """Delete a memory."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        mid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory ID format")

    success = await _pipeline.delete_memory(mid)
    return {"success": success}
