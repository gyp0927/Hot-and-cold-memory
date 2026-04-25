"""Document ingestion pipeline."""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from adaptive_rag.core.config import Tier, get_settings
from adaptive_rag.core.exceptions import IngestionError
from adaptive_rag.core.logging import get_logger
from adaptive_rag.storage.document_store.base import BaseDocumentStore
from adaptive_rag.storage.metadata_store.base import (
    BaseMetadataStore,
    DocumentMetadata,
    ChunkMetadata,
)
from adaptive_rag.storage.vector_store.base import BaseVectorStore
from adaptive_rag.tiers.hot_tier import HotTier

from .chunker import Chunk, RecursiveChunker
from .embedder import Embedder
from .extractors.text import extract_text

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    document_id: uuid.UUID
    status: str = "pending"
    chunks_created: int = 0
    total_chunks: int = 0
    error: str | None = None
    processing_time_ms: float = 0.0


class IngestionPipeline:
    """Orchestrates document ingestion into the system.

    New documents always start in the Hot Tier with maximum frequency score.
    """

    def __init__(
        self,
        metadata_store: BaseMetadataStore,
        hot_tier: HotTier,
        embedder: Embedder,
        chunker: RecursiveChunker | None = None,
    ) -> None:
        self.settings = get_settings()
        self.metadata_store = metadata_store
        self.hot_tier = hot_tier
        self.embedder = embedder
        self.chunker = chunker or RecursiveChunker(
            chunk_size=512,
            chunk_overlap=50,
        )

    async def ingest_text(
        self,
        text: str,
        source_uri: str = "inline",
        title: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a text document.

        Args:
            text: Document text content.
            source_uri: Source identifier.
            title: Optional document title.
            tags: Optional tags.
            metadata: Optional metadata dict.

        Returns:
            Ingestion result.
        """
        start_time = datetime.utcnow()
        document_id = uuid.uuid4()

        try:
            # 1. Compute content hash for deduplication
            content_hash = hashlib.sha256(text.encode()).hexdigest()

            # 2. Store document metadata
            doc_meta = DocumentMetadata(
                document_id=document_id,
                source_type="text",
                source_uri=source_uri,
                title=title,
                content_hash=content_hash,
                total_chunks=0,
                metadata=metadata or {},
            )
            await self.metadata_store.create_document(doc_meta)

            # 3. Chunk the document
            chunks = self.chunker.chunk(
                text=text,
                document_id=document_id,
                tags=tags or [],
            )

            # 4. Generate embeddings
            texts = [c.text for c in chunks]
            embeddings = await self.embedder.embed_batch(texts)

            # 5. Store in Hot Tier (new docs start hot)
            chunk_metadata_list = await self.hot_tier.store_chunks(
                chunks=chunks,
                embeddings=embeddings,
            )

            # 6. Update document with chunk count
            await self.metadata_store.update_document(
                document_id=document_id,
                updates={"total_chunks": len(chunks)},
            )

            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.info(
                "ingestion_complete",
                document_id=str(document_id),
                chunks=len(chunks),
                elapsed_ms=elapsed,
            )

            return IngestionResult(
                document_id=document_id,
                status="success",
                chunks_created=len(chunks),
                total_chunks=len(chunks),
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error("ingestion_failed", document_id=str(document_id), error=str(e))
            return IngestionResult(
                document_id=document_id,
                status="failed",
                error=str(e),
            )

    async def ingest_file(
        self,
        file_path: str,
        title: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a file from disk.

        Args:
            file_path: Path to file.
            title: Optional title.
            tags: Optional tags.
            metadata: Optional metadata.

        Returns:
            Ingestion result.
        """
        text = extract_text(file_path)
        path = Path(file_path)

        return await self.ingest_text(
            text=text,
            source_uri=str(path.absolute()),
            title=title or path.name,
            tags=tags,
            metadata=metadata,
        )
