"""API request/response schemas for queries."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Query request."""

    query: str = Field(..., min_length=1, max_length=10000)
    top_k: int = Field(default=10, ge=1, le=100)
    tier: Literal["hot", "cold", "both"] | None = Field(
        default=None,
        description="Tier preference. None = auto-route based on frequency.",
    )
    decompress: bool = Field(
        default=False,
        description="Decompress cold chunks in response.",
    )
    filters: dict | None = Field(
        default=None,
        description="Metadata filters.",
    )


class RetrievedChunkSchema(BaseModel):
    """A retrieved chunk in the response."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    tier: Literal["hot", "cold"]
    is_decompressed: bool
    access_count: int
    frequency_score: float


class QueryResponse(BaseModel):
    """Query response."""

    chunks: list[RetrievedChunkSchema]
    routing_strategy: Literal["hot_only", "cold_only", "hot_first", "both"]
    hot_results_count: int
    cold_results_count: int
    total_latency_ms: float
    topic_frequency: float
