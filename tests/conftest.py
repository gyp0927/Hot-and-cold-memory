"""Shared pytest fixtures."""

import pytest_asyncio


@pytest_asyncio.fixture
async def metadata_store():
    """Create a metadata store for testing."""
    from adaptive_rag.storage.metadata_store.postgres_store import PostgresMetadataStore
    store = PostgresMetadataStore()
    await store.initialize()
    yield store
