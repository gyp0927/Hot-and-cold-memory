"""Qdrant vector store implementation."""

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.exceptions import VectorStoreError
from hot_and_cold_memory.core.logging import get_logger

from .base import BaseVectorStore, VectorSearchResult

logger = get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-based vector store."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        """Connect to Qdrant and ensure collections exist."""
        self.client = AsyncQdrantClient(
            host=self.settings.VECTOR_DB_HOST,
            port=self.settings.VECTOR_DB_PORT,
        )
        await self._ensure_collection(self.settings.VECTOR_DB_COLLECTION)
        await self._ensure_collection("query_clusters")
        logger.info("qdrant_initialized")

    async def _ensure_collection(self, name: str) -> None:
        """Create collection if it doesn't exist."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        collections = await self.client.get_collections()
        existing = {c.name for c in collections.collections}

        if name not in existing:
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("collection_created", collection=name)

    async def upsert(
        self,
        collection: str,
        ids: list[uuid.UUID],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store or update vectors."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        if payloads is None:
            payloads = [{} for _ in ids]

        points = [
            PointStruct(
                id=str(id_),
                vector=vec,
                payload=payload,
            )
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]

        await self.client.upsert(
            collection_name=collection,
            points=points,
        )

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        qdrant_filter = self._build_filter(filters) if filters else None

        results = await self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [
            VectorSearchResult(
                chunk_id=uuid.UUID(r.id),
                score=r.score,
                vector=None,
                payload=r.payload or {},
            )
            for r in results
        ]

    async def delete(
        self,
        collection: str,
        ids: list[uuid.UUID],
    ) -> int:
        """Delete vectors by ID."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        await self.client.delete(
            collection_name=collection,
            points_selector=PointIdsList(
                points=[str(id_) for id_ in ids],
            ),
        )
        return len(ids)

    async def get_by_id(
        self,
        collection: str,
        chunk_id: uuid.UUID,
    ) -> VectorSearchResult | None:
        """Get a vector by ID."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        results = await self.client.retrieve(
            collection_name=collection,
            ids=[str(chunk_id)],
            with_payload=True,
            with_vectors=True,
        )

        if not results:
            return None

        r = results[0]
        return VectorSearchResult(
            chunk_id=uuid.UUID(r.id),
            score=1.0,
            vector=r.vector,
            payload=r.payload or {},
        )

    async def count(self, collection: str) -> int:
        """Count vectors in collection."""
        if not self.client:
            raise VectorStoreError("Client not initialized")

        result = await self.client.count(collection_name=collection)
        return result.count

    async def search_batch(
        self,
        collection: str,
        query_vectors: list[list[float]],
        limit: int = 1,
    ) -> list[list[VectorSearchResult]]:
        """Batch search for multiple query vectors.

        Uses Qdrant's native search_batch API for efficient multi-query search.
        """
        if not self.client:
            raise VectorStoreError("Client not initialized")

        if not query_vectors:
            return []

        from qdrant_client.models import SearchRequest

        requests = [
            SearchRequest(
                vector=qv, limit=limit, with_payload=True, with_vector=False
            )
            for qv in query_vectors
        ]

        results = await self.client.search_batch(
            collection_name=collection,
            requests=requests,
        )

        return [
            [
                VectorSearchResult(
                    chunk_id=uuid.UUID(r.id),
                    score=r.score,
                    vector=None,
                    payload=r.payload or {},
                )
                for r in batch
            ]
            for batch in results
        ]

    def _build_filter(self, filters: dict[str, Any]) -> Filter | None:
        """Build Qdrant filter from dict."""
        conditions: list[FieldCondition] = []

        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        return Filter(must=conditions) if conditions else None
