"""Unit tests for API routers."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from adaptive_memory.api.main import create_app
from adaptive_memory.api.routers import admin, health, memories, retrieve


class TestRetrieveRouter:
    """Test retrieve endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked retriever."""
        app = create_app()
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.memories = []
        mock_result.routing_strategy.value = "hot_only"
        mock_result.hot_results_count = 0
        mock_result.cold_results_count = 0
        mock_result.total_latency_ms = 10.0
        mock_result.topic_frequency = 0.5
        mock_retriever.retrieve = AsyncMock(return_value=mock_result)
        retrieve.set_retriever(mock_retriever)
        return TestClient(app)

    def test_retrieve_endpoint(self, client):
        """Test POST /retrieve."""
        response = client.post(
            "/api/v1/retrieve",
            json={"query": "test query", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["routing_strategy"] == "hot_only"
        assert "memories" in data

    def test_retrieve_tier_both(self, client):
        """Test retrieve with tier='both' doesn't crash."""
        response = client.post(
            "/api/v1/retrieve",
            json={"query": "test query", "tier": "both"},
        )
        assert response.status_code == 200

    def test_retrieve_tier_hot(self, client):
        """Test retrieve with tier='hot'."""
        response = client.post(
            "/api/v1/retrieve",
            json={"query": "test query", "tier": "hot"},
        )
        assert response.status_code == 200


class TestMemoriesRouter:
    """Test memory endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked pipeline."""
        app = create_app()
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.memory_id = uuid.uuid4()
        mock_result.status = "success"
        mock_result.tier = "hot"
        mock_result.error = None
        mock_pipeline.write_memory = AsyncMock(return_value=mock_result)
        mock_pipeline.delete_memory = AsyncMock(return_value=True)
        memories.set_pipeline(mock_pipeline)

        mock_store = MagicMock()
        mock_store.list_memories = AsyncMock(return_value=[])
        mock_store.get_memory = AsyncMock(return_value=None)
        memories.set_metadata_store(mock_store)
        return TestClient(app)

    def test_create_memory(self, client):
        """Test POST /memories."""
        response = client.post(
            "/api/v1/memories",
            json={"content": "User likes Python", "memory_type": "fact"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_create_memory_empty(self, client):
        """Test memory creation with empty content fails validation."""
        response = client.post(
            "/api/v1/memories",
            json={"content": ""},
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
        mock_store.count_memories_by_tier = AsyncMock(return_value=5)
        mock_store.list_memories = AsyncMock(return_value=[])
        mock_store.get_all_clusters = AsyncMock(return_value=[])
        admin.set_metadata_store(mock_store)
        return TestClient(app)

    def test_stats_endpoint(self, client):
        """Test GET /admin/stats."""
        response = client.get("/api/v1/admin/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_memories" in data
        assert "hot_memories" in data
        assert "cold_memories" in data

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
        mock_meta.count_memories_by_tier = AsyncMock(return_value=5)
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
