"""Tier migration engine."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from adaptive_rag.core.config import Tier, get_settings
from adaptive_rag.core.exceptions import MigrationError, ChunkNotFoundError
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.chunker import Chunk
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.storage.metadata_store.base import (
    BaseMetadataStore,
    MigrationLog,
)
from adaptive_rag.tiers.base import RetrievedChunk
from adaptive_rag.tiers.hot_tier import HotTier
from adaptive_rag.tiers.cold_tier import ColdTier
from adaptive_rag.tiers.compression import CompressionEngine
from adaptive_rag.tiers.decompression import DecompressionEngine

from .policies import MigrationPolicy

logger = get_logger(__name__)


@dataclass
class MigrationResult:
    """Result of a single migration."""

    chunk_id: uuid.UUID
    direction: str
    original_size: int
    new_size: int
    compression_ratio: float
    success: bool = True
    error: str | None = None


@dataclass
class MigrationReport:
    """Report of a migration cycle."""

    started_at: datetime
    completed_at: datetime | None = None
    hot_to_cold: list[MigrationResult] = field(default_factory=list)
    cold_to_hot: list[MigrationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_processed: int = 0


class MigrationEngine:
    """Orchestrates chunk migration between hot and cold tiers."""

    def __init__(
        self,
        hot_tier: HotTier,
        cold_tier: ColdTier,
        metadata_store: BaseMetadataStore,
        policy: MigrationPolicy | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = get_settings()
        self.hot_tier = hot_tier
        self.cold_tier = cold_tier
        self.metadata_store = metadata_store
        self.policy = policy or MigrationPolicy()
        self.embedder = embedder or Embedder()
        self._lock = asyncio.Lock()

    async def run_migration_cycle(self) -> MigrationReport:
        """Execute one migration cycle.

        Identifies candidates and executes hot->cold and cold->hot migrations.

        Returns:
            Migration report.
        """
        report = MigrationReport(
            started_at=datetime.utcnow(),
        )

        async with self._lock:
            # Phase 1: Identify candidates
            hot_candidates = await self._identify_hot_to_cold_candidates()
            cold_candidates = await self._identify_cold_to_hot_candidates()

            logger.info(
                "migration_candidates",
                hot_to_cold=len(hot_candidates),
                cold_to_hot=len(cold_candidates),
            )

            # Phase 2: Execute hot -> cold migrations
            semaphore = asyncio.Semaphore(self.policy.thresholds.max_concurrent)

            async def migrate_hot_to_cold(chunk_id: uuid.UUID) -> MigrationResult:
                async with semaphore:
                    return await self._migrate_hot_to_cold(chunk_id)

            hot_results = await asyncio.gather(*[
                migrate_hot_to_cold(cid) for cid in hot_candidates
            ], return_exceptions=True)

            for result in hot_results:
                if isinstance(result, Exception):
                    report.errors.append(str(result))
                else:
                    report.hot_to_cold.append(result)

            # Phase 3: Execute cold -> hot migrations
            async def migrate_cold_to_hot(chunk_id: uuid.UUID) -> MigrationResult:
                async with semaphore:
                    return await self._migrate_cold_to_hot(chunk_id)

            cold_results = await asyncio.gather(*[
                migrate_cold_to_hot(cid) for cid in cold_candidates
            ], return_exceptions=True)

            for result in cold_results:
                if isinstance(result, Exception):
                    report.errors.append(str(result))
                else:
                    report.cold_to_hot.append(result)

        report.completed_at = datetime.utcnow()
        report.total_processed = len(report.hot_to_cold) + len(report.cold_to_hot)

        logger.info(
            "migration_cycle_complete",
            hot_to_cold=len(report.hot_to_cold),
            cold_to_hot=len(report.cold_to_hot),
            errors=len(report.errors),
            duration_seconds=(report.completed_at - report.started_at).total_seconds(),
        )

        return report

    async def _migrate_hot_to_cold(self, chunk_id: uuid.UUID) -> MigrationResult:
        """Migrate a chunk from hot to cold tier.

        Args:
            chunk_id: Chunk to migrate.

        Returns:
            Migration result.
        """
        log = MigrationLog(
            chunk_id=chunk_id,
            direction="hot_to_cold",
            original_size=0,
            new_size=0,
            started_at=datetime.utcnow(),
        )

        try:
            # 1. Retrieve from hot tier
            chunk = await self.hot_tier.get_by_id(chunk_id)
            if not chunk:
                raise ChunkNotFoundError(f"Chunk {chunk_id} not found in hot tier")

            # 2. Delete old metadata first (avoid unique constraint)
            await self.metadata_store.delete_chunks([chunk_id])

            # 3. Store in cold tier (compresses automatically)
            await self.cold_tier.store_chunks(
                chunks=[Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.content,
                    tags=chunk.metadata.get("tags", []) if chunk.metadata else [],
                )],
            )

            # 4. Delete from hot tier
            await self.hot_tier.delete([chunk_id])

            # 4. Get compressed info
            meta = await self.metadata_store.get_chunk(chunk_id)
            original_size = len(chunk.content)
            new_size = meta.compressed_length if meta else original_size
            ratio = new_size / original_size if original_size > 0 else 1.0

            # 5. Log migration
            log.original_size = original_size
            log.new_size = new_size
            log.compression_ratio = ratio
            log.completed_at = datetime.utcnow()
            log.status = "success"
            await self.metadata_store.create_migration_log(log)

            logger.info(
                "migrated_hot_to_cold",
                chunk_id=str(chunk_id),
                original=original_size,
                compressed=new_size,
                ratio=ratio,
            )

            return MigrationResult(
                chunk_id=chunk_id,
                direction="hot_to_cold",
                original_size=original_size,
                new_size=new_size,
                compression_ratio=ratio,
            )

        except Exception as e:
            log.status = "failed"
            log.error_message = str(e)
            log.completed_at = datetime.utcnow()
            await self.metadata_store.create_migration_log(log)
            raise MigrationError(f"Hot to cold migration failed for {chunk_id}: {e}") from e

    async def _migrate_cold_to_hot(self, chunk_id: uuid.UUID) -> MigrationResult:
        """Migrate a chunk from cold to hot tier.

        Args:
            chunk_id: Chunk to migrate.

        Returns:
            Migration result.
        """
        log = MigrationLog(
            chunk_id=chunk_id,
            direction="cold_to_hot",
            original_size=0,
            new_size=0,
            started_at=datetime.utcnow(),
        )

        try:
            # 1. Retrieve from cold tier
            chunk = await self.cold_tier.get_by_id(chunk_id)
            if not chunk:
                raise ChunkNotFoundError(f"Chunk {chunk_id} not found in cold tier")

            # Get original content (summary)
            summary = chunk.content

            # 2. Decompress
            decompressed = await self.cold_tier.decompression_engine.decompress(summary)

            # 3. Generate embedding
            embedding = await self.embedder.embed(decompressed)

            # 4. Delete old metadata first (avoid unique constraint)
            await self.metadata_store.delete_chunks([chunk_id])

            # 5. Store in hot tier
            await self.hot_tier.store_chunks(
                chunks=[Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=decompressed,
                    tags=chunk.metadata.get("tags", []) if chunk.metadata else [],
                )],
                embeddings=[embedding],
            )

            # 6. Delete from cold tier
            await self.cold_tier.delete([chunk_id])

            # 6. Log migration
            log.original_size = len(summary)
            log.new_size = len(decompressed)
            log.compression_ratio = len(summary) / len(decompressed) if len(decompressed) > 0 else 1.0
            log.completed_at = datetime.utcnow()
            log.status = "success"
            await self.metadata_store.create_migration_log(log)

            logger.info(
                "migrated_cold_to_hot",
                chunk_id=str(chunk_id),
                summary_len=len(summary),
                expanded_len=len(decompressed),
            )

            return MigrationResult(
                chunk_id=chunk_id,
                direction="cold_to_hot",
                original_size=len(summary),
                new_size=len(decompressed),
                compression_ratio=log.compression_ratio or 0,
            )

        except Exception as e:
            log.status = "failed"
            log.error_message = str(e)
            log.completed_at = datetime.utcnow()
            await self.metadata_store.create_migration_log(log)
            raise MigrationError(f"Cold to hot migration failed for {chunk_id}: {e}") from e

    async def _identify_hot_to_cold_candidates(self) -> list[uuid.UUID]:
        """Identify hot chunks with low frequency for demotion.

        Returns:
            List of chunk IDs to demote.
        """
        chunks = await self.metadata_store.query_chunks_by_tier_and_score(
            tier=Tier.HOT,
            max_score=self.policy.thresholds.hot_to_cold,
            limit=self.policy.thresholds.batch_size,
        )
        return [c.chunk_id for c in chunks]

    async def _identify_cold_to_hot_candidates(self) -> list[uuid.UUID]:
        """Identify cold chunks with high frequency for promotion.

        Returns:
            List of chunk IDs to promote.
        """
        chunks = await self.metadata_store.query_chunks_by_tier_and_score(
            tier=Tier.COLD,
            min_score=self.policy.thresholds.cold_to_hot,
            limit=self.policy.thresholds.batch_size,
        )
        return [c.chunk_id for c in chunks]
