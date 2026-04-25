"""Custom exception hierarchy."""


class AdaptiveRAGError(Exception):
    """Base exception for all adaptive RAG errors."""

    pass


class StorageError(AdaptiveRAGError):
    """Storage layer failure."""

    pass


class VectorStoreError(StorageError):
    """Vector database operation failed."""

    pass


class MetadataStoreError(StorageError):
    """Metadata database operation failed."""

    pass


class DocumentStoreError(StorageError):
    """Document store operation failed."""

    pass


class CacheError(StorageError):
    """Cache operation failed."""

    pass


class CompressionError(AdaptiveRAGError):
    """LLM compression failed."""

    pass


class DecompressionError(AdaptiveRAGError):
    """LLM decompression failed."""

    pass


class MigrationError(AdaptiveRAGError):
    """Tier migration failed."""

    pass


class TierError(AdaptiveRAGError):
    """Tier operation failed."""

    pass


class RoutingError(AdaptiveRAGError):
    """Query routing failed."""

    pass


class IngestionError(AdaptiveRAGError):
    """Document ingestion failed."""

    pass


class ChunkNotFoundError(AdaptiveRAGError):
    """Requested chunk not found."""

    pass


class ClusterNotFoundError(AdaptiveRAGError):
    """Query cluster not found."""

    pass
