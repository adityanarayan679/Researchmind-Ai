"""Vector embeddings using SentenceTransformers with caching."""

import hashlib
import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import Settings

settings = Settings()

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingClient:
    """Generates vector embeddings for text using SentenceTransformers.

    The model is loaded lazily (on first call) to avoid long startup
    times. An in-memory cache prevents re-encoding identical texts.

    Usage:
        client = EmbeddingClient()
        vec = client.encode("What is RAG?")
        vecs = client.encode_batch(["chunk one", "chunk two"])
        print(client.embedding_dimension)  # 384
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model: Optional[SentenceTransformer] = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors (384 for all-MiniLM-L6-v2)."""
        model = self._get_model()
        return model.get_embedding_dimension() or 384

    # ── Public API ──────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a vector.

        Args:
            text: The text to embed.

        Returns:
            A 1-D numpy array of floats representing the text.

        Raises:
            EmbeddingError: If encoding fails.
        """
        cache_key = self._hash(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            model = self._get_model()
            vector = model.encode(text, normalize_embeddings=True)
            self._cache[cache_key] = vector
            return vector
        except Exception as exc:
            logger.exception("Failed to encode text")
            raise EmbeddingError(f"Encoding failed: {exc}") from exc

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a matrix of vectors.

        Uses batched inference and respects the cache for texts that
        have already been encoded.

        Args:
            texts: List of text strings to embed.

        Returns:
            A 2-D numpy array of shape (len(texts), embedding_dim).
        """
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for idx, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                continue
            uncached_texts.append(text)
            uncached_indices.append(idx)

        if uncached_texts:
            try:
                model = self._get_model()
                new_vectors = model.encode(
                    uncached_texts, normalize_embeddings=True, show_progress_bar=False
                )
                for text, vec in zip(uncached_texts, new_vectors):
                    self._cache[self._hash(text)] = vec
            except Exception as exc:
                logger.exception("Failed to encode batch")
                raise EmbeddingError(f"Batch encoding failed: {exc}") from exc

        all_vectors = np.array([self._cache[self._hash(t)] for t in texts])
        return all_vectors

    # ── Private helpers ─────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            try:
                logger.info("Loading embedding model: %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load model '{self._model_name}': {exc}"
                ) from exc
        return self._model

    @staticmethod
    def _hash(text: str) -> str:
        """Return a short hash of the text for cache key lookup."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
