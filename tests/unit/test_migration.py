"""Unit tests for migration system."""



from hot_and_cold_memory.migration.engine import MigrationEngine
from hot_and_cold_memory.migration.policies import MigrationPolicy


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
        assert policy.should_promote(0.7, 0) is True
        assert policy.should_promote(0.9, 0) is True

    def test_should_not_promote_low_frequency(self):
        """Chunks with low frequency should not be promoted."""
        policy = MigrationPolicy()
        assert policy.should_promote(0.3, 0) is False
        assert policy.should_promote(0.5, 0) is False

    def test_should_promote_high_access_count(self):
        """Chunks with high cumulative access count should be promoted."""
        policy = MigrationPolicy()
        assert policy.should_promote(0.3, 50) is True
        assert policy.should_promote(0.5, 100) is True

    def test_should_not_promote_low_access_count(self):
        """Chunks with low access count should not be promoted."""
        policy = MigrationPolicy()
        assert policy.should_promote(0.3, 10) is False
        assert policy.should_promote(0.5, 30) is False


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

        monkeypatch.setattr("hot_and_cold_memory.migration.engine.datetime", MockDatetime)
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

        monkeypatch.setattr("hot_and_cold_memory.migration.engine.datetime", MockDatetime)
        assert engine._is_off_peak() is False
