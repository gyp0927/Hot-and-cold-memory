"""Shared pytest fixtures."""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def metadata_store():
    """Create a SQLite-backed metadata store for testing."""
    import os

    from hot_and_cold_memory.core.config import get_settings
    from hot_and_cold_memory.storage.metadata_store.postgres_store import PostgresMetadataStore

    settings = get_settings()
    # Use a temporary SQLite database for tests
    original_url = settings.METADATA_DB_URL
    tmp_db = tempfile.mktemp(suffix=".db")
    settings.METADATA_DB_URL = f"sqlite+aiosqlite:///{tmp_db}"

    store = PostgresMetadataStore()
    await store.initialize()
    yield store

    await store.engine.dispose()
    settings.METADATA_DB_URL = original_url
    if os.path.exists(tmp_db):
        os.unlink(tmp_db)


@pytest_asyncio.fixture
async def vector_store():
    """Create a local Qdrant store for testing."""
    import shutil

    from hot_and_cold_memory.storage.vector_store.local_qdrant_store import LocalQdrantStore

    tmp_dir = tempfile.mkdtemp()
    store = LocalQdrantStore()
    store._path = str(Path(tmp_dir) / "qdrant_test")
    await store.initialize()
    yield store

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def document_store():
    """Create a local document store for testing."""
    import shutil

    from hot_and_cold_memory.core.config import get_settings
    from hot_and_cold_memory.storage.document_store.local_store import LocalDocumentStore

    settings = get_settings()
    tmp_dir = tempfile.mkdtemp()
    original_path = settings.DOCUMENT_STORE_PATH
    settings.DOCUMENT_STORE_PATH = tmp_dir

    store = LocalDocumentStore()
    yield store

    settings.DOCUMENT_STORE_PATH = original_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mock LLM client to avoid real API calls."""
    class MockClient:
        async def complete(self, *args, **kwargs):
            return '{"summary": "mock summary", "key_entities": [], "key_facts": []}'

    monkeypatch.setattr(
        "hot_and_cold_memory.core.llm_client.LLMClient",
        MockClient,
    )
    return MockClient()
