"""Embedding generation supporting multiple providers."""

import asyncio
from typing import Any

import numpy as np

from adaptive_rag.core.config import get_settings, EmbeddingProvider
from adaptive_rag.core.exceptions import IngestionError
from adaptive_rag.core.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """Embedding generator supporting OpenAI and local models.

    Providers:
        - openai: OpenAI API (paid, requires API key)
        - sentence-transformers: Local models (free, runs on CPU/GPU)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.EMBEDDING_PROVIDER
        self._batch_size = 100
        self._semaphore = asyncio.Semaphore(10)

        # Lazy-loaded clients
        self._openai_client: Any = None
        self._local_model: Any = None

    def _get_openai_client(self) -> Any:
        """Lazy initialize OpenAI client."""
        if self._openai_client is None:
            import openai
            self._openai_client = openai.AsyncOpenAI(api_key=self.settings.LLM_API_KEY)
        return self._openai_client

    def _get_local_model(self) -> Any:
        """Lazy initialize sentence-transformers model."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise IngestionError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

            logger.info(
                "loading_local_embedding_model",
                model=self.settings.LOCAL_EMBEDDING_MODEL,
                device=self.settings.LOCAL_EMBEDDING_DEVICE,
            )
            self._local_model = SentenceTransformer(
                self.settings.LOCAL_EMBEDDING_MODEL,
                device=self.settings.LOCAL_EMBEDDING_DEVICE,
            )
            logger.info("local_embedding_model_loaded")
        return self._local_model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if not text.strip():
            # Return zero vector for empty text
            dim = self.settings.EMBEDDING_DIMENSION
            return [0.0] * dim

        if self.provider == EmbeddingProvider.OPENAI:
            return await self._embed_openai(text)
        else:
            return await self._embed_local(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order.
        """
        if not texts:
            return []

        # Filter out empty texts
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            dim = self.settings.EMBEDDING_DIMENSION
            return [[0.0] * dim for _ in texts]

        if self.provider == EmbeddingProvider.OPENAI:
            embeddings = await self._embed_batch_openai([t for _, t in non_empty])
        else:
            embeddings = await self._embed_batch_local([t for _, t in non_empty])

        # Reconstruct with empty texts as zero vectors
        result: list[list[float]] = []
        emb_idx = 0
        dim = self.settings.EMBEDDING_DIMENSION
        for i in range(len(texts)):
            if any(idx == i for idx, _ in non_empty):
                result.append(embeddings[emb_idx])
                emb_idx += 1
            else:
                result.append([0.0] * dim)

        return result

    async def _embed_openai(self, text: str) -> list[float]:
        """Embed using OpenAI API."""
        client = self._get_openai_client()
        async with self._semaphore:
            try:
                response = await client.embeddings.create(
                    model=self.settings.EMBEDDING_MODEL,
                    input=text,
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error("openai_embed_error", text=text[:100], error=str(e))
                raise IngestionError(f"OpenAI embedding failed: {e}") from e

    async def _embed_batch_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed batch using OpenAI API."""
        client = self._get_openai_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            async with self._semaphore:
                for attempt in range(3):
                    try:
                        response = await client.embeddings.create(
                            model=self.settings.EMBEDDING_MODEL,
                            input=batch,
                        )
                        embeddings = sorted(response.data, key=lambda x: x.index)
                        all_embeddings.extend([e.embedding for e in embeddings])
                        break
                    except Exception as e:
                        if attempt == 2:
                            logger.error("openai_batch_error", count=len(batch), error=str(e))
                            raise IngestionError(f"OpenAI batch embedding failed: {e}") from e
                        await asyncio.sleep(2 ** attempt)

        return all_embeddings

    async def _embed_local(self, text: str) -> list[float]:
        """Embed using local sentence-transformers model."""
        model = self._get_local_model()
        # Run in thread pool to avoid blocking event loop
        embedding = await asyncio.to_thread(model.encode, text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    async def _embed_batch_local(self, texts: list[str]) -> list[list[float]]:
        """Embed batch using local sentence-transformers model."""
        model = self._get_local_model()
        # Run in thread pool to avoid blocking event loop
        embeddings = await asyncio.to_thread(model.encode, texts)
        # Convert numpy arrays to lists
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [list(e) for e in embeddings]
