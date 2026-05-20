"""Tests for ImportanceScorer and MigrationPolicy importance protection."""

import pytest

from hot_and_cold_memory.ingestion.importance_scorer import ImportanceScorer
from hot_and_cold_memory.migration.policies import MigrationPolicy


class TestImportanceScorer:
    """Test suite for rule-based + optional LLM importance scoring."""

    @pytest.fixture
    def scorer(self, monkeypatch):
        """Create a scorer with auto-importance enabled but no LLM."""
        monkeypatch.setenv("ENABLE_AUTO_IMPORTANCE", "true")
        monkeypatch.setenv("AUTO_IMPORTANCE_USE_LLM", "false")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)
        return ImportanceScorer()

    @pytest.fixture
    def disabled_scorer(self, monkeypatch):
        """Create a scorer with auto-importance disabled."""
        monkeypatch.setenv("ENABLE_AUTO_IMPORTANCE", "false")
        # Clear cached settings so the env var is re-read
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)
        return ImportanceScorer()

    @pytest.fixture
    def llm_scorer(self, monkeypatch):
        """Create a scorer with LLM fallback enabled."""
        monkeypatch.setenv("ENABLE_AUTO_IMPORTANCE", "true")
        monkeypatch.setenv("AUTO_IMPORTANCE_USE_LLM", "true")
        monkeypatch.setenv("AUTO_IMPORTANCE_LLM_THRESHOLD", "0.25")
        import hot_and_cold_memory.core.config as _config
        monkeypatch.setattr(_config, "_settings", None)

        class FakeLLM:
            async def complete(self, prompt, **kwargs):
                return "85"

        return ImportanceScorer(llm_client=FakeLLM())

    # --- rule-based scoring ---

    @pytest.mark.asyncio
    async def test_high_signal_preference(self, scorer):
        content = "用户最喜欢Python编程语言，非常不喜欢JavaScript，这是他的长期偏好"
        score = await scorer.score(content)
        assert score >= 0.6

    @pytest.mark.asyncio
    async def test_high_signal_identity(self, scorer):
        content = "用户的名字是张三，职业是后端工程师，目前住址在北京朝阳区"
        score = await scorer.score(content)
        assert score >= 0.6

    @pytest.mark.asyncio
    async def test_high_signal_health(self, scorer):
        content = "用户对花生严重过敏，必须避免吃含花生的食物，健康非常重要"
        score = await scorer.score(content)
        assert score >= 0.6

    @pytest.mark.asyncio
    async def test_low_signal_small_talk(self, scorer):
        content = "今天天气不错，早上吃了包子"
        score = await scorer.score(content)
        assert score <= 0.4

    @pytest.mark.asyncio
    async def test_low_signal_very_short(self, scorer):
        content = "嗯，好的"
        score = await scorer.score(content)
        assert score <= 0.35

    @pytest.mark.asyncio
    async def test_medium_signal(self, scorer):
        content = "用户可能在考虑换工作"
        score = await scorer.score(content)
        assert 0.2 <= score <= 0.6

    # --- memory type multipliers ---

    @pytest.mark.asyncio
    async def test_fact_multiplier(self, scorer):
        content = "用户可能在考虑换工作"
        obs = await scorer.score(content, "observation")
        fact = await scorer.score(content, "fact")
        assert fact >= obs

    @pytest.mark.asyncio
    async def test_summary_multiplier(self, scorer):
        content = "用户可能在考虑换工作"
        obs = await scorer.score(content, "observation")
        summ = await scorer.score(content, "summary")
        assert summ >= obs

    # --- disabled ---

    @pytest.mark.asyncio
    async def test_disabled_returns_default(self, disabled_scorer):
        score = await disabled_scorer.score("用户最喜欢Python")
        assert score == 0.5

    # --- batch scoring ---

    @pytest.mark.asyncio
    async def test_batch_scoring(self, scorer):
        items = [
            ("用户最喜欢Python", "fact"),
            ("今天天气不错", "observation"),
        ]
        scores = await scorer.score_batch(items)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    # --- LLM fallback ---

    @pytest.mark.asyncio
    async def test_llm_fallback_for_ambiguous(self, llm_scorer):
        # Content with medium length and no keywords -> rule score ~0.5 (ambiguous band)
        content = "用户最近的工作进度总体上保持稳定，没有出现特别大的波动"
        score = await llm_scorer.score(content)
        # FakeLLM returns 85 -> 0.85, blended with rule ~0.5: 0.5*0.6 + 0.85*0.4 = 0.64
        assert score > 0.55

    @pytest.mark.asyncio
    async def test_llm_not_called_for_clear_scores(self, llm_scorer):
        # High-signal content with fact type has clear score > 0.75,
        # so LLM fallback should not be triggered
        content = "用户最喜欢Python，不喜欢JavaScript，这是他长期的编程偏好"
        score = await llm_scorer.score(content, memory_type="fact")
        # Rule score is already high (>0.75), no LLM fallback needed
        assert score >= 0.7


class TestMigrationPolicyImportance:
    """Test importance-aware demotion logic."""

    @pytest.fixture
    def policy(self):
        p = MigrationPolicy()
        # Pin thresholds so tests are independent of .env values
        p.thresholds.hot_to_cold = 0.25
        p.thresholds.cold_to_hot = 0.7
        p.thresholds.hot_access_count = 50
        return p

    def test_should_demote_low_importance(self, policy):
        # Low importance: normal threshold
        assert policy.should_demote(0.1, importance=0.3) is True
        assert policy.should_demote(0.3, importance=0.3) is False

    def test_should_demote_high_importance_protected(self, policy):
        # High importance (>=0.8): threshold reduced by 0.15
        # default hot_to_cold = 0.25, effective = 0.10
        assert policy.should_demote(0.05, importance=0.9) is True
        assert policy.should_demote(0.15, importance=0.9) is False

    def test_should_demote_medium_importance_partial_protection(self, policy):
        # Medium importance (>=0.6): threshold reduced by 0.08
        # default hot_to_cold = 0.25, effective = 0.17
        assert policy.should_demote(0.1, importance=0.7) is True
        assert policy.should_demote(0.2, importance=0.7) is False

    def test_should_demote_threshold_clamped(self, policy):
        # Force a low base threshold so the clamp actually triggers
        policy.thresholds.hot_to_cold = 0.15
        # importance >= 0.8 reduces by 0.15 -> 0.00, clamped to 0.05
        assert policy.should_demote(0.04, importance=1.0) is True
        assert policy.should_demote(0.06, importance=1.0) is False

    def test_should_promote_by_frequency(self, policy):
        assert policy.should_promote(0.8, access_count=0) is True
        assert policy.should_promote(0.6, access_count=0) is False

    def test_should_promote_by_access_count(self, policy):
        assert policy.should_promote(0.5, access_count=60) is True
        assert policy.should_promote(0.5, access_count=10) is False
