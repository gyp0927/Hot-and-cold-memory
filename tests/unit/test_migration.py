"""Unit tests for migration system."""

from datetime import datetime

import pytest

from adaptive_rag.core.config import get_settings
from adaptive_rag.migration.policies import MigrationPolicy
from adaptive_rag.migration.engine import MigrationEngine


class TestMigrationPolicy:
    """Test migration policy decisions."""

    def test_should_demote_low_frequency(self):
        """Chunks with very low frequency should be demoted."""
        policy = MigrationPolicy()
        assert policy.should_demote(0.1) is True
        assert policy.should_demote(0.3) is True

    def test_should_not_demote_high_frequency(self):
        """Chunks with high frequency should not be demoted."""
        policy = MigrationPolicy()
        assert policy.should_demote(0.8) is False
        assert policy.should_demote(0.5) is False

    def test_should_promote_high_frequency(self):
        """Chunks with high frequency should be promoted."""
        policy = MigrationPolicy()
        assert policy.should_promote(0.7) is True
        assert policy.should_promote(0.9) is True

    def test_should_not_promote_low_frequency(self):
        """Chunks with low frequency should not be promoted."""
        policy = MigrationPolicy()
        assert policy.should_promote(0.3) is False
        assert policy.should_promote(0.5) is False


class TestMigrationEngine:
    """Test migration engine logic."""

    def test_is_off_peak_midnight(self, monkeypatch):
        """Test off-peak detection at midnight."""
        from unittest.mock import MagicMock

        engine = MigrationEngine(
            hot_tier=MagicMock(),
            cold_tier=MagicMock(),
            metadata_store=MagicMock(),
        )

        # Mock datetime.now() to return 3 AM
        class MockDatetime:
            @classmethod
            def now(cls):
                class Time:
                    hour = 3
                return Time()

        monkeypatch.setattr("adaptive_rag.migration.engine.datetime", MockDatetime)
        assert engine._is_off_peak() is True

    def test_is_off_peak_noon(self, monkeypatch):
        """Test off-peak detection at noon."""
        from unittest.mock import MagicMock

        engine = MigrationEngine(
            hot_tier=MagicMock(),
            cold_tier=MagicMock(),
            metadata_store=MagicMock(),
        )

        class MockDatetime:
            @classmethod
            def now(cls):
                class Time:
                    hour = 12
                return Time()

        monkeypatch.setattr("adaptive_rag.migration.engine.datetime", MockDatetime)
        assert engine._is_off_peak() is False
