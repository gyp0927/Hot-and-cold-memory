"""On-demand decompression engine for cold tier."""

from adaptive_rag.core.config import get_settings
from adaptive_rag.core.exceptions import DecompressionError
from adaptive_rag.core.llm_client import LLMClient
from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)


class DecompressionEngine:
    """On-demand decompression that expands cold summaries.

    Uses the compressed summary + LLM to generate a contextually
    appropriate expansion while maintaining factual accuracy.
    """

    DECOMPRESSION_PROMPT = """You are a knowledge expansion engine. Expand the following compressed summary into a detailed, informative response.

Compressed summary:
{summary}

Instructions:
1. Expand the summary into a comprehensive response
2. Include all key entities and facts from the summary
3. Add relevant context and explanations where appropriate
4. Maintain factual accuracy - do not hallucinate information not in the summary
5. Write in a clear, professional tone

Expanded response:
"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = LLMClient()
        self.model = self.settings.DECOMPRESSION_MODEL

    async def decompress(self, summary: str) -> str:
        """Decompress a summary back to full detail.

        Args:
            summary: Compressed summary text.

        Returns:
            Expanded/detailed text.
        """
        prompt = self.DECOMPRESSION_PROMPT.format(summary=summary)

        try:
            response = await self.client.complete(
                prompt=prompt,
                model=self.model,
                max_tokens=self.settings.LLM_MAX_TOKENS,
                temperature=0.3,
            )

            logger.debug(
                "chunk_decompressed",
                summary_len=len(summary),
                expanded_len=len(response),
            )

            return response

        except Exception as e:
            logger.error("decompression_failed", error=str(e))
            raise DecompressionError(f"Failed to decompress: {e}") from e
