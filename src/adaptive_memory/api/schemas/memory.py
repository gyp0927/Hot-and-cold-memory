"""API request/response schemas for memories."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class MemoryCreateRequest(BaseModel):
    """Memory creation request."""

    content: str = Field(..., min_length=1, max_length=100000)
    memory_type: Literal["observation", "fact", "reflection", "summary"] = "observation"
    source: str | None = Field(default=None, description="Source identifier, e.g., conversation ID")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    attributes: dict = Field(default_factory=dict)


class MemoryCreateResponse(BaseModel):
    """Memory creation response."""

    memory_id: uuid.UUID
    status: str
    tier: str
    message: str


class MemoryDetailResponse(BaseModel):
    """Memory detail response."""

    memory_id: uuid.UUID
    content: str
    memory_type: str
    source: str | None = None
    importance: float
    tier: str
    access_count: int
    frequency_score: float
    created_at: str
    tags: list[str]


class MemoryListResponse(BaseModel):
    """Memory list response."""

    memories: list[MemoryDetailResponse]
    total: int
