"""API request/response schemas for documents."""

from pydantic import BaseModel, Field
from typing import Literal
import uuid


class DocumentUploadRequest(BaseModel):
    """Document upload request."""

    chunking_strategy: Literal["fixed", "recursive"] = "recursive"
    chunk_size: int = Field(default=512, ge=100, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Document upload response."""

    document_id: uuid.UUID
    status: str
    chunks_created: int
    message: str


class DocumentResponse(BaseModel):
    """Document metadata response."""

    document_id: uuid.UUID
    source_type: str
    source_uri: str
    title: str | None = None
    total_chunks: int
    created_at: str
