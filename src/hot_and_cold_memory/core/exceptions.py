"""Custom exception hierarchy."""


class AdaptiveMemoryError(Exception):
    """Base exception for all adaptive memory errors."""

    pass


class StorageError(AdaptiveMemoryError):
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


class CompressionError(AdaptiveMemoryError):
    """LLM compression failed."""

    pass


class DecompressionError(AdaptiveMemoryError):
    """LLM decompression failed."""

    pass


class MigrationError(AdaptiveMemoryError):
    """Tier migration failed."""

    pass


class TierError(AdaptiveMemoryError):
    """Tier operation failed."""

    pass


class RoutingError(AdaptiveMemoryError):
    """Query routing failed."""

    pass


class IngestionError(AdaptiveMemoryError):
    """Document ingestion failed."""

    pass


class ChunkNotFoundError(AdaptiveMemoryError):
    """Requested chunk not found."""

    pass


class ClusterNotFoundError(AdaptiveMemoryError):
    """Query cluster not found."""

    pass
