"""Auto-importance scoring for memories using rule-based + optional LLM fallback.

Rules cover high-signal patterns (preferences, identity, facts) and low-signal
patterns (weather, small talk).  When the result is ambiguous and LLM fallback
is enabled, a lightweight prompt refines the score.
"""

import re
from typing import Literal

from hot_and_cold_memory.core.config import get_settings
from hot_and_cold_memory.core.llm_client import LLMClient
from hot_and_cold_memory.core.logging import get_logger

logger = get_logger(__name__)


# Keywords that indicate high personal relevance (preference, identity, constraints)
_HIGH_SIGNAL_KEYWORDS = {
    "喜欢", "偏好", "最爱", "讨厌", "厌恶", "不喜欢",
    "总是", "从不", "绝不会", "一定", "必须",
    "身份", "职业", "工作", "名字", "姓名",
    "住址", "地址", "电话", "邮箱", "生日",
    "过敏", "疾病", "健康", "药物",
    "密码", "账号", "账户", "银行",
    "目标", "梦想", "规划", "计划", "志向",
    "习惯", "日常", "规律",
    "家人", "父母", "配偶", "孩子", "朋友",
}

# Keywords that indicate medium relevance (soft facts, tentative info)
_MEDIUM_SIGNAL_KEYWORDS = {
    "尝试", "可能", "也许", "考虑", "想法", "觉得",
    "认为", "看起来", "似乎", "大概",
    "最近", "前段时间", "以前", "曾经",
    "项目", "任务", "会议", "讨论",
}

# Keywords that indicate low relevance (ephemeral / small talk)
_LOW_SIGNAL_KEYWORDS = {
    "天气", "今天", "明天", "昨天", "早上", "晚上",
    "随便", "没事", "好的", "嗯", "哦",
    "哈哈", "呵呵", "有趣", "无聊",
    "吃了", "喝了", "睡了", "醒了",
}

# Memory-type multipliers
_TYPE_MULTIPLIERS: dict[str, float] = {
    "fact": 1.2,
    "summary": 1.15,
    "reflection": 1.1,
    "observation": 1.0,
}


class ImportanceScorer:
    """Score memory importance (0.0-1.0) using fast rules + optional LLM."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.settings = get_settings()
        self._llm = llm_client
        self._enabled = self.settings.ENABLE_AUTO_IMPORTANCE
        self._use_llm = self.settings.AUTO_IMPORTANCE_USE_LLM
        self._llm_threshold = self.settings.AUTO_IMPORTANCE_LLM_THRESHOLD

    async def score(
        self,
        content: str,
        memory_type: str = "observation",
    ) -> float:
        """Return an importance score between 0.0 and 1.0.

        Args:
            content: Memory text content.
            memory_type: Type of memory (affects multiplier).

        Returns:
            Importance score clamped to [0.0, 1.0].
        """
        if not self._enabled:
            return 0.5

        rule_score = self._rule_based_score(content)
        multiplier = _TYPE_MULTIPLIERS.get(memory_type, 1.0)
        adjusted = min(rule_score * multiplier, 1.0)

        # If the score sits in the ambiguous band and LLM fallback is on, refine it
        if self._use_llm and self._llm and self._is_ambiguous(adjusted):
            try:
                llm_score = await self._llm_score(content)
                # Weighted blend: rule gets 60 %, LLM gets 40 %
                blended = adjusted * 0.6 + llm_score * 0.4
                adjusted = round(blended, 2)
            except Exception as exc:
                logger.warning("llm_importance_score_failed", error=str(exc))

        return round(adjusted, 2)

    async def score_batch(
        self,
        items: list[tuple[str, str]],
    ) -> list[float]:
        """Score a batch of memories efficiently.

        Args:
            items: List of (content, memory_type) tuples.

        Returns:
            List of scores in the same order.
        """
        if not self._enabled:
            return [0.5] * len(items)

        base_scores = [self._rule_based_score(c) for c, _ in items]

        if not (self._use_llm and self._llm):
            return [
                round(min(base * _TYPE_MULTIPLIERS.get(mt, 1.0), 1.0), 2)
                for base, (_, mt) in zip(base_scores, items)
            ]

        # Async LLM refinement for ambiguous items
        results: list[float] = []
        for base, (content, mt) in zip(base_scores, items):
            adjusted = min(base * _TYPE_MULTIPLIERS.get(mt, 1.0), 1.0)
            if self._is_ambiguous(adjusted):
                try:
                    llm_score = await self._llm_score(content)
                    adjusted = round(adjusted * 0.6 + llm_score * 0.4, 2)
                except Exception as exc:
                    logger.warning("llm_importance_score_failed", error=str(exc))
            results.append(adjusted)
        return results

    # --- internal ---

    def _rule_based_score(self, content: str) -> float:
        """Fast keyword-driven scoring."""
        text = content.lower()

        high_hits = sum(1 for kw in _HIGH_SIGNAL_KEYWORDS if kw in text)
        medium_hits = sum(1 for kw in _MEDIUM_SIGNAL_KEYWORDS if kw in text)
        low_hits = sum(1 for kw in _LOW_SIGNAL_KEYWORDS if kw in text)

        # Length heuristic: very short utterances are usually low-signal
        length_score = 0.5
        char_count = len(text)
        if char_count > 200:
            length_score = 0.7
        elif char_count < 30:
            length_score = 0.2

        # Combine signals
        score = length_score + high_hits * 0.15 + medium_hits * 0.05 - low_hits * 0.1
        return max(0.0, min(1.0, score))

    def _is_ambiguous(self, score: float) -> bool:
        """Check if a score is in the ambiguous band where LLM help is useful."""
        return abs(score - 0.5) < self._llm_threshold

    async def _llm_score(self, content: str) -> float:
        """Ask a lightweight LLM to rate importance 0-100, then normalize."""
        if not self._llm:
            return 0.5

        prompt = (
            "Rate how important the following personal memory is for an AI to remember "
            "long-term, on a scale of 0 to 100.\n\n"
            "0 = trivial (e.g., \"The weather is nice\")\n"
            "50 = moderately useful (e.g., \"User is working on a project\")\n"
            "100 = critical (e.g., \"User is allergic to peanuts\")\n\n"
            f"Memory: \"{content[:500]}\"\n\n"
            "Reply with ONLY a number between 0 and 100, no explanation."
        )
        response = await self._llm.complete(prompt, max_tokens=10, temperature=0.0)
        # Extract first number from response
        match = re.search(r"\b(\d{1,3})\b", response.strip())
        if match:
            raw = int(match.group(1))
            return max(0.0, min(1.0, raw / 100.0))
        return 0.5
