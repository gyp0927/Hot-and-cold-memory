"""Embedding generation supporting multiple providers."""

import asyncio
import hashlib
from collections import OrderedDict
from typing import Any

from hot_and_cold_memory.core.config import EmbeddingProvider, get_settings
from hot_and_cold_memory.core.exceptions import IngestionError
from hot_and_cold_memory.core.logging import get_logger

logger = get_logger(__name__)


class _LRUCache:
    """Simple async-safe LRU cache for embedding vectors."""

    def __init__(self, maxsize: int = 2000) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def get(self, text: str) -> list[float] | None:
        async with self._lock:
            key = self._key(text)
            if key in self._cache:
                self._cache.move_to_end(key)
                return list(self._cache[key])
            return None

    async def set(self, text: str, vector: list[float]) -> None:
        async with self._lock:
            key = self._key(text)
            self._cache[key] = list(vector)
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    async def get_batch(self, texts: list[str]) -> tuple[list[int], list[str]]:
        """Return (cached_indices, missing_texts) for a batch.

        cached_indices maps the original index to the cached vector.
        """
        async with self._lock:
            cached: list[int] = []
            missing: list[str] = []
            missing_indices: list[int] = []
            for i, text in enumerate(texts):
                key = self._key(text)
                if key in self._cache:
                    self._cache.move_to_end(key)
                    cached.append(i)
                else:
                    missing.append(text)
                    missing_indices.append(i)
            return cached, missing, missing_indices

    async def set_batch(self, texts: list[str], vectors: list[list[float]]) -> None:
        async with self._lock:
            for text, vector in zip(texts, vectors):
                key = self._key(text)
                self._cache[key] = list(vector)
                self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)


class Embedder:
    """Embedding generator supporting OpenAI and local models.

    Providers:
        - openai: OpenAI API (paid, requires API key)
        - sentence-transformers: Local models (free, runs on CPU/GPU)

    Caches the most recent 2,000 embedding vectors to avoid recomputing
    identical texts (common during repeated queries or re-ingestion).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.EMBEDDING_PROVIDER
        self._batch_size = 256
        self._semaphore = asyncio.Semaphore(10)

        # Lazy-loaded clients
        self._openai_client: Any = None
        self._local_model: Any = None

        # Embedding cache
        self._cache = _LRUCache(maxsize=2000)

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
        """Embed a single text (cached).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if not text.strip():
            dim = self.settings.EMBEDDING_DIMENSION
            return [0.0] * dim

        cached = await self._cache.get(text)
        if cached is not None:
            return cached

        if self.provider == EmbeddingProvider.OPENAI:
            vector = await self._embed_openai(text)
        else:
            vector = await self._embed_local(text)

        await self._cache.set(text, vector)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently (with cache lookup).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in same order.
        """
        if not texts:
            return []

        # Handle empty texts first
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not non_empty_indices:
            dim = self.settings.EMBEDDING_DIMENSION
            return [[0.0] * dim for _ in texts]

        non_empty_texts = [texts[i] for i in non_empty_indices]

        # Cache lookup for non-empty texts
        cached_indices, missing_texts, missing_orig_indices = await self._cache.get_batch(non_empty_texts)

        # Build result for cached items
        cached_vectors: dict[int, list[float]] = {}
        for orig_idx in cached_indices:
            cached_vectors[non_empty_indices[orig_idx]] = await self._cache.get(non_empty_texts[orig_idx])

        # Compute missing embeddings
        if missing_texts:
            if self.provider == EmbeddingProvider.OPENAI:
                computed = await self._embed_batch_openai(missing_texts)
            else:
                computed = await self._embed_batch_local(missing_texts)

            # Store in cache
            await self._cache.set_batch(missing_texts, computed)

            # Map back to original indices
            for miss_idx, orig_idx_in_non_empty in enumerate(missing_orig_indices):
                original_index = non_empty_indices[orig_idx_in_non_empty]
                cached_vectors[original_index] = computed[miss_idx]

        # Reconstruct full result with empty texts as zero vectors
        dim = self.settings.EMBEDDING_DIMENSION
        result: list[list[float]] = []
        for i in range(len(texts)):
            if i in cached_vectors:
                result.append(cached_vectors[i])
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
