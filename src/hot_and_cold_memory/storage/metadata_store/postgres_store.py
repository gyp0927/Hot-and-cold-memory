"""Metadata store implementation using SQLAlchemy async (PostgreSQL/SQLite)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import and_, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hot_and_cold_memory.core.config import Tier, get_settings
from hot_and_cold_memory.core.logging import get_logger

from .base import (
    AccessLog,
    BaseMetadataStore,
    MemoryItem,
    MigrationLog,
    TopicCluster,
)
from .models import (
    AccessLogModel,
    Base,
    MemoryModel,
    MigrationLogModel,
    TopicClusterModel,
)

logger = get_logger(__name__)


def _to_uuid_str(value: uuid.UUID | str) -> str:
    """Convert UUID to string."""
    return str(value) if isinstance(value, uuid.UUID) else value


def _memory_to_item(model: MemoryModel) -> MemoryItem:
    """Convert MemoryModel to MemoryItem dataclass."""
    return MemoryItem(
        memory_id=uuid.UUID(model.memory_id) if isinstance(model.memory_id, str) else model.memory_id,
        tier=Tier(model.tier),
        content=model.content or "",
        original_length=model.original_length,
        memory_type=model.memory_type,
        source=model.source,
        importance=model.importance,
        access_count=model.access_count,
        frequency_score=model.frequency_score,
        created_at=model.created_at,
        updated_at=model.updated_at,
        last_accessed_at=model.last_accessed_at,
        last_migrated_at=model.last_migrated_at,
        topic_cluster_id=uuid.UUID(model.topic_cluster_id) if model.topic_cluster_id else None,
        tags=list(model.tags) if model.tags else [],
        attributes=dict(model.attributes) if model.attributes else {},
        vector_id=model.vector_id,
    )


def _cluster_to_dataclass(model: TopicClusterModel) -> TopicCluster:
    """Convert TopicClusterModel to TopicCluster dataclass."""
    return TopicCluster(
        cluster_id=uuid.UUID(model.cluster_id) if isinstance(model.cluster_id, str) else model.cluster_id,
        centroid=list(model.centroid) if model.centroid else [],
        representative_query=model.representative_query,
        access_count=model.access_count,
        frequency_score=model.frequency_score,
        member_count=model.member_count,
        created_at=model.created_at,
        last_accessed_at=model.last_accessed_at,
    )


class PostgresMetadataStore(BaseMetadataStore):
    """Metadata store implementation supporting PostgreSQL and SQLite."""

    def __init__(self) -> None:
        self.settings = get_settings()
        db_url = str(self.settings.METADATA_DB_URL)

        # SQLite doesn't support connection pooling
        if db_url.startswith("sqlite"):
            self.engine = create_async_engine(
                db_url,
                echo=self.settings.DEBUG,
            )
        else:
            self.engine = create_async_engine(
                db_url,
                echo=self.settings.DEBUG,
                pool_size=10,
                max_overflow=20,
            )

        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("metadata_store_initialized")

    # --- Memory operations ---

    async def create_memory(self, metadata: MemoryItem) -> None:
        """Create a new memory record."""
        async with self.async_session() as session:
            model = MemoryModel(
                memory_id=_to_uuid_str(metadata.memory_id),
                tier=metadata.tier.value,
                content=metadata.content,
                original_length=metadata.original_length,
                memory_type=metadata.memory_type,
                source=metadata.source,
                importance=metadata.importance,
                access_count=metadata.access_count,
                frequency_score=metadata.frequency_score,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
                last_accessed_at=metadata.last_accessed_at,
                last_migrated_at=metadata.last_migrated_at,
                topic_cluster_id=_to_uuid_str(metadata.topic_cluster_id) if metadata.topic_cluster_id else None,
                tags=metadata.tags,
                attributes=metadata.attributes,
                vector_id=metadata.vector_id,
            )
            session.add(model)
            await session.commit()

    async def get_memory(self, memory_id: uuid.UUID) -> MemoryItem | None:
        """Get memory by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(MemoryModel).where(MemoryModel.memory_id == _to_uuid_str(memory_id))
            )
            model = result.scalar_one_or_none()
            return _memory_to_item(model) if model else None

    async def get_memories_batch(self, memory_ids: list[uuid.UUID]) -> list[MemoryItem]:
        """Get multiple memories by ID in a single query."""
        if not memory_ids:
            return []
        id_strs = [_to_uuid_str(mid) for mid in memory_ids]
        async with self.async_session() as session:
            result = await session.execute(
                select(MemoryModel).where(MemoryModel.memory_id.in_(id_strs))
            )
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def create_memories_batch(self, metadatas: list[MemoryItem]) -> None:
        """Create multiple memory records in a single transaction."""
        if not metadatas:
            return
        async with self.async_session() as session:
            models = [
                MemoryModel(
                    memory_id=_to_uuid_str(m.memory_id),
                    tier=m.tier.value,
                    content=m.content,
                    original_length=m.original_length,
                    memory_type=m.memory_type,
                    source=m.source,
                    importance=m.importance,
                    access_count=m.access_count,
                    frequency_score=m.frequency_score,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    last_accessed_at=m.last_accessed_at,
                    last_migrated_at=m.last_migrated_at,
                    topic_cluster_id=_to_uuid_str(m.topic_cluster_id) if m.topic_cluster_id else None,
                    tags=m.tags,
                    attributes=m.attributes,
                    vector_id=m.vector_id,
                )
                for m in metadatas
            ]
            session.add_all(models)
            await session.commit()

    async def update_memory(
        self,
        memory_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> MemoryItem | None:
        """Update memory fields."""
        async with self.async_session() as session:
            if "tier" in updates and isinstance(updates["tier"], Tier):
                updates["tier"] = updates["tier"].value

            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.memory_id == _to_uuid_str(memory_id))
                .values(**updates, updated_at=datetime.utcnow())
            )
            await session.commit()

            result = await session.execute(
                select(MemoryModel).where(MemoryModel.memory_id == _to_uuid_str(memory_id))
            )
            model = result.scalar_one_or_none()
            return _memory_to_item(model) if model else None

    async def update_memories_batch(
        self,
        updates: dict[uuid.UUID, dict[str, Any]],
    ) -> None:
        """Update multiple memories in a single transaction."""
        if not updates:
            return
        async with self.async_session() as session:
            for memory_id, memory_updates in updates.items():
                upd = dict(memory_updates)
                if "tier" in upd and isinstance(upd["tier"], Tier):
                    upd["tier"] = upd["tier"].value
                upd["updated_at"] = datetime.utcnow()
                await session.execute(
                    update(MemoryModel)
                    .where(MemoryModel.memory_id == _to_uuid_str(memory_id))
                    .values(**upd)
                )
            await session.commit()

    async def delete_memories(self, memory_ids: list[uuid.UUID]) -> int:
        """Delete memories."""
        id_strs = [_to_uuid_str(mid) for mid in memory_ids]
        async with self.async_session() as session:
            result = await session.execute(
                delete(MemoryModel).where(MemoryModel.memory_id.in_(id_strs))
            )
            await session.commit()
            return result.rowcount or 0

    async def list_memories(
        self,
        memory_type: str | None = None,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryItem]:
        """List memories with optional filtering."""
        async with self.async_session() as session:
            stmt = select(MemoryModel)
            if memory_type:
                stmt = stmt.where(MemoryModel.memory_type == memory_type)
            if source:
                stmt = stmt.where(MemoryModel.source == source)
            result = await session.execute(
                stmt.order_by(MemoryModel.created_at.desc()).limit(limit).offset(offset)
            )
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def count_memories_by_tier(self, tier: Tier) -> int:
        """Count memories in a given tier."""
        async with self.async_session() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(func.count(MemoryModel.memory_id)).where(MemoryModel.tier == tier.value)
            )
            return result.scalar() or 0

    async def query_memories_by_tier_and_score(
        self,
        tier: Tier,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 100,
        order_desc: bool = False,
    ) -> list[MemoryItem]:
        """Query memories by tier and frequency score range."""
        async with self.async_session() as session:
            conditions = [MemoryModel.tier == tier.value]

            if min_score is not None:
                conditions.append(MemoryModel.frequency_score >= min_score)
            if max_score is not None:
                conditions.append(MemoryModel.frequency_score <= max_score)

            stmt = (
                select(MemoryModel)
                .where(and_(*conditions))
                .limit(limit)
            )
            if order_desc:
                stmt = stmt.order_by(MemoryModel.frequency_score.desc())
            else:
                stmt = stmt.order_by(MemoryModel.frequency_score)

            result = await session.execute(stmt)
            models = result.scalars().all()
            return [_memory_to_item(m) for m in models]

    async def increment_access(
        self,
        memory_ids: list[uuid.UUID],
        cluster_id: uuid.UUID | None,
        timestamp: datetime,
    ) -> None:
        """Increment access count for memories in a single UPDATE."""
        if not memory_ids:
            return
        id_strs = [_to_uuid_str(mid) for mid in memory_ids]
        cluster_str = _to_uuid_str(cluster_id) if cluster_id else None
        async with self.async_session() as session:
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.memory_id.in_(id_strs))
                .values(
                    access_count=MemoryModel.access_count + 1,
                    last_accessed_at=timestamp,
                    topic_cluster_id=cluster_str,
                )
            )
            await session.commit()

    # --- Topic cluster operations ---

    async def create_cluster(self, cluster: TopicCluster) -> None:
        """Create a new topic cluster."""
        async with self.async_session() as session:
            model = TopicClusterModel(
                cluster_id=_to_uuid_str(cluster.cluster_id),
                representative_query=cluster.representative_query,
                access_count=cluster.access_count,
                frequency_score=cluster.frequency_score,
                member_count=cluster.member_count,
                created_at=cluster.created_at,
                last_accessed_at=cluster.last_accessed_at,
                centroid=cluster.centroid,
            )
            session.add(model)
            await session.commit()

    async def get_cluster(self, cluster_id: uuid.UUID) -> TopicCluster | None:
        """Get cluster by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(TopicClusterModel).where(TopicClusterModel.cluster_id == _to_uuid_str(cluster_id))
            )
            model = result.scalar_one_or_none()
            return _cluster_to_dataclass(model) if model else None

    async def update_cluster(
        self,
        cluster_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> TopicCluster | None:
        """Update cluster fields."""
        async with self.async_session() as session:
            await session.execute(
                update(TopicClusterModel)
                .where(TopicClusterModel.cluster_id == _to_uuid_str(cluster_id))
                .values(**updates)
            )
            await session.commit()

            result = await session.execute(
                select(TopicClusterModel).where(TopicClusterModel.cluster_id == _to_uuid_str(cluster_id))
            )
            model = result.scalar_one_or_none()
            return _cluster_to_dataclass(model) if model else None

    async def get_all_clusters(self) -> list[TopicCluster]:
        """Get all topic clusters."""
        async with self.async_session() as session:
            result = await session.execute(select(TopicClusterModel))
            models = result.scalars().all()
            return [_cluster_to_dataclass(m) for m in models]

    async def get_clusters_batch(self, cluster_ids: list[uuid.UUID]) -> list[TopicCluster]:
        """Get multiple clusters by ID in a single query."""
        if not cluster_ids:
            return []
        id_strs = [_to_uuid_str(cid) for cid in cluster_ids]
        async with self.async_session() as session:
            result = await session.execute(
                select(TopicClusterModel).where(TopicClusterModel.cluster_id.in_(id_strs))
            )
            models = result.scalars().all()
            return [_cluster_to_dataclass(m) for m in models]

    async def delete_clusters(self, cluster_ids: list[uuid.UUID]) -> int:
        """Delete topic clusters."""
        id_strs = [_to_uuid_str(cid) for cid in cluster_ids]
        async with self.async_session() as session:
            result = await session.execute(
                delete(TopicClusterModel).where(TopicClusterModel.cluster_id.in_(id_strs))
            )
            await session.commit()
            return result.rowcount or 0

    # --- Access / migration log operations ---

    async def create_access_log(self, log: AccessLog) -> None:
        """Create an access log entry."""
        async with self.async_session() as session:
            model = AccessLogModel(
                memory_id=_to_uuid_str(log.memory_id),
                query_cluster_id=_to_uuid_str(log.query_cluster_id) if log.query_cluster_id else None,
                query_text=log.query_text,
                retrieved_at=log.retrieved_at,
                response_time_ms=log.response_time_ms,
                tier_accessed=log.tier_accessed,
            )
            session.add(model)
            await session.commit()

    async def create_migration_log(self, log: MigrationLog) -> None:
        """Create a migration log entry."""
        async with self.async_session() as session:
            model = MigrationLogModel(
                memory_id=_to_uuid_str(log.memory_id),
                direction=log.direction,
                original_size=log.original_size,
                new_size=log.new_size,
                compression_ratio=log.compression_ratio,
                started_at=log.started_at,
                completed_at=log.completed_at,
                status=log.status,
                error_message=log.error_message,
            )
            session.add(model)
            await session.commit()

    async def update_migration_log(
        self,
        log_id: int,
        updates: dict[str, Any],
    ) -> None:
        """Update migration log."""
        async with self.async_session() as session:
            await session.execute(
                update(MigrationLogModel)
                .where(MigrationLogModel.log_id == log_id)
                .values(**updates)
            )
            await session.commit()
