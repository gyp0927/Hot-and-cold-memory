"""Integration tests for retrieval flow."""

import pytest


class TestRetrievalFlow:
    """Test end-to-end retrieval with routing."""

    @pytest.mark.asyncio
    async def test_routing_strategy_selection(self):
        """Test that routing strategy is selected based on frequency."""
        from adaptive_memory.core.config import RoutingStrategy

        # This is a basic smoke test - in a real test we'd mock dependencies
        assert RoutingStrategy.HOT_ONLY.value == "hot_only"
        assert RoutingStrategy.COLD_ONLY.value == "cold_only"
        assert RoutingStrategy.BOTH.value == "both"
