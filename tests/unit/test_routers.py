"""Unit tests for API routers."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from adaptive_rag.api.main import create_app
from adaptive_rag.api.routers import query, documents, admin, health


class TestQueryRouter:
    """Test query endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked retriever."""
        app = create_app()
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.routing_strategy.value = "hot_only"
        mock_result.hot_results_count = 0
        mock_result.cold_results_count = 0
        mock_result.total_latency_ms = 10.0
        mock_result.topic_frequency = 0.5
        mock_retriever.query = AsyncMock(return_value=mock_result)
        query.set_retriever(mock_retriever)
        return TestClient(app)

    def test_query_endpoint(self, client):
        """Test POST /query."""
        response = client.post(
            "/api/v1/query",
            json={"query": "test query", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["routing_strategy"] == "hot_only"
        assert "chunks" in data

    def test_query_tier_both(self, client):
        """Test query with tier='both' doesn't crash."""
        response = client.post(
            "/api/v1/query",
            json={"query": "test query", "tier": "both"},
        )
        assert response.status_code == 200

    def test_query_tier_hot(self, client):
        """Test query with tier='hot'."""
        response = client.post(
            "/api/v1/query",
            json={"query": "test query", "tier": "hot"},
        )
        assert response.status_code == 200


class TestDocumentsRouter:
    """Test document endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked pipeline."""
        app = create_app()
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.document_id = uuid.uuid4()
        mock_result.status = "success"
        mock_result.chunks_created = 3
        mock_result.message = None
        mock_result.error = None
        mock_pipeline.ingest_text = AsyncMock(return_value=mock_result)
        documents.set_pipeline(mock_pipeline)
        documents.set_stores(MagicMock(), MagicMock())
        return TestClient(app)

    def test_upload_text(self, client):
        """Test POST /documents/text."""
        response = client.post(
            "/api/v1/documents/text",
            json={"text": "Hello world, this is a test document.", "title": "Test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_created"] == 3

    def test_upload_text_empty(self, client):
        """Test text upload with empty text fails validation."""
        response = client.post(
            "/api/v1/documents/text",
            json={"text": ""},
        )
        assert response.status_code == 422


class TestAdminRouter:
    """Test admin endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked services."""
        app = create_app()
        mock_engine = MagicMock()
        mock_report = MagicMock()
        mock_report.errors = []
        mock_report.hot_to_cold = []
        mock_report.cold_to_hot = []
        mock_report.started_at = None
        mock_report.completed_at = None
        mock_engine.run_migration_cycle = AsyncMock(return_value=mock_report)
        admin.set_migration_engine(mock_engine)

        mock_store = MagicMock()
        mock_store.count_chunks_by_tier = AsyncMock(return_value=5)
        mock_store.list_documents = AsyncMock(return_value=[])
        mock_store.get_all_clusters = AsyncMock(return_value=[])
        admin.set_metadata_store(mock_store)
        return TestClient(app)

    def test_stats_endpoint(self, client):
        """Test GET /admin/stats."""
        response = client.get("/api/v1/admin/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "hot_chunks" in data
        assert "cold_chunks" in data

    def test_migrate_endpoint(self, client):
        """Test POST /admin/migrate."""
        response = client.post("/api/v1/admin/migrate")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestHealthRouter:
    """Test health endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked stores."""
        app = create_app()
        mock_meta = MagicMock()
        mock_meta.count_chunks_by_tier = AsyncMock(return_value=5)
        mock_vec = MagicMock()
        mock_vec.count = AsyncMock(return_value=10)
        health.set_stores(mock_meta, mock_vec)
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test GET /health."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_ready_endpoint(self, client):
        """Test GET /ready."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "checks" in data
