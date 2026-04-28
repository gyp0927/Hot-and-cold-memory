"""SQLAlchemy models for metadata store (PostgreSQL/SQLite compatible)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy base class."""

    pass


class MemoryModel(Base):
    """Memory item metadata table."""

    __tablename__ = "memories"

    memory_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    tier: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    original_length: Mapped[int] = mapped_column(Integer, nullable=False)

    memory_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="observation", index=True
    )
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frequency_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_migrated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    topic_cluster_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("topic_clusters.cluster_id"), nullable=True, index=True
    )

    compression_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class TopicClusterModel(Base):
    """Topic cluster table for memory organization."""

    __tablename__ = "topic_clusters"

    cluster_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    representative_query: Mapped[str] = mapped_column(Text, nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frequency_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    centroid: Mapped[list[float]] = mapped_column(JSON, nullable=False)


class AccessLogModel(Base):
    """Access log table."""

    __tablename__ = "access_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("memories.memory_id"), nullable=False
    )
    query_cluster_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("topic_clusters.cluster_id"), nullable=True
    )
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tier_accessed: Mapped[str | None] = mapped_column(String(10), nullable=True)


class MigrationLogModel(Base):
    """Migration log table."""

    __tablename__ = "migration_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("memories.memory_id"), nullable=False
    )
    direction: Mapped[str] = mapped_column(String(20), nullable=False)
    original_size: Mapped[int] = mapped_column(Integer, nullable=False)
    new_size: Mapped[int] = mapped_column(Integer, nullable=False)
    compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
