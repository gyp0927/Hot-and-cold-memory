"""Semantic query clustering for topic-based frequency tracking."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.logging import get_logger
from adaptive_rag.ingestion.embedder import Embedder
from adaptive_rag.storage.metadata_store.base import BaseMetadataStore, QueryCluster
from adaptive_rag.storage.vector_store.base import BaseVectorStore

logger = get_logger(__name__)


@dataclass
class ClusterMatch:
    """Result of cluster matching."""

    cluster: QueryCluster
    similarity: float


class QueryClusterStore:
    """Manages query clusters using vector store for similarity search."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        metadata_store: BaseMetadataStore,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.collection = "query_clusters"

    async def find_nearest_cluster(
        self,
        query_embedding: list[float],
        threshold: float | None = None,
    ) -> QueryCluster | None:
        """Find the nearest cluster within similarity threshold.

        Args:
            query_embedding: Query embedding vector.
            threshold: Minimum cosine similarity (default from config).

        Returns:
            Nearest cluster or None if none within threshold.
        """
        threshold = threshold or self.settings.QUERY_CLUSTERING_THRESHOLD

        results = await self.vector_store.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=1,
        )

        if not results:
            return None

        # Qdrant returns cosine similarity directly (higher = more similar)
        if results[0].score < threshold:
            return None

        cluster_id = uuid.UUID(results[0].payload.get("cluster_id")) if results[0].payload else None
        if not cluster_id:
            return None

        return await self.metadata_store.get_cluster(cluster_id)

    async def create_cluster(self, cluster: QueryCluster) -> None:
        """Create a new query cluster.

        Args:
            cluster: Cluster to create.
        """
        # Store centroid in vector store
        await self.vector_store.upsert(
            collection=self.collection,
            ids=[cluster.cluster_id],
            vectors=[cluster.centroid],
            payloads=[{
                "cluster_id": str(cluster.cluster_id),
                "representative_query": cluster.representative_query,
                "access_count": cluster.access_count,
                "frequency_score": cluster.frequency_score,
                "member_count": cluster.member_count,
            }],
        )

        # Store metadata
        await self.metadata_store.create_cluster(cluster)

        logger.info(
            "cluster_created",
            cluster_id=str(cluster.cluster_id),
            query=cluster.representative_query[:100],
        )

    async def update_cluster(
        self,
        cluster_id: uuid.UUID,
        updates: dict[str, Any],
    ) -> None:
        """Update cluster fields.

        Args:
            cluster_id: Cluster ID.
            updates: Fields to update.
        """
        await self.metadata_store.update_cluster(cluster_id, updates)

    async def increment_access(
        self,
        cluster_id: uuid.UUID,
        timestamp: datetime,
    ) -> None:
        """Increment cluster access count.

        Args:
            cluster_id: Cluster ID.
            timestamp: Access timestamp.
        """
        cluster = await self.metadata_store.get_cluster(cluster_id)
        if not cluster:
            return

        await self.metadata_store.update_cluster(
            cluster_id=cluster_id,
            updates={
                "access_count": cluster.access_count + 1,
                "last_accessed_at": timestamp,
                "member_count": cluster.member_count + 1,
            },
        )

    async def merge_clusters(
        self,
        cluster_id_1: uuid.UUID,
        cluster_id_2: uuid.UUID,
    ) -> QueryCluster:
        """Merge two clusters into one.

        Args:
            cluster_id_1: First cluster ID.
            cluster_id_2: Second cluster ID.

        Returns:
            Merged cluster.
        """
        c1 = await self.metadata_store.get_cluster(cluster_id_1)
        c2 = await self.metadata_store.get_cluster(cluster_id_2)

        if not c1 or not c2:
            raise ValueError("One or both clusters not found")

        # Weighted average of centroids
        total = c1.member_count + c2.member_count
        new_centroid = [
            (c1.centroid[i] * c1.member_count + c2.centroid[i] * c2.member_count) / total
            for i in range(len(c1.centroid))
        ]

        merged = QueryCluster(
            cluster_id=uuid.uuid4(),
            centroid=new_centroid,
            representative_query=c1.representative_query,
            access_count=c1.access_count + c2.access_count,
            frequency_score=max(c1.frequency_score, c2.frequency_score),
            member_count=total,
            created_at=min(c1.created_at, c2.created_at),
            last_accessed_at=max(
                c1.last_accessed_at or c1.created_at,
                c2.last_accessed_at or c2.created_at,
            ),
        )

        # Delete old clusters from vector store
        await self.vector_store.delete(
            collection=self.collection,
            ids=[cluster_id_1, cluster_id_2],
        )

        # Create merged cluster
        await self.create_cluster(merged)

        logger.info(
            "clusters_merged",
            cluster_1=str(cluster_id_1),
            cluster_2=str(cluster_id_2),
            merged=str(merged.cluster_id),
        )

        return merged
