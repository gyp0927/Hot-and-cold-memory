"""Initialize database tables."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_rag.storage.metadata_store.postgres_store import PostgresMetadataStore
from adaptive_rag.storage.vector_store.local_qdrant_store import LocalQdrantStore
from adaptive_rag.core.logging import setup_logging


async def init_db() -> None:
    """Create all database tables and collections."""
    setup_logging("INFO")

    print("Initializing metadata store...")
    metadata_store = PostgresMetadataStore()
    await metadata_store.initialize()
    print("Metadata store initialized.")

    print("Initializing vector store (local mode)...")
    vector_store = LocalQdrantStore()
    await vector_store.initialize()
    print("Vector store initialized.")

    print("Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(init_db())
