"""FAISS vector store for semantic search over document chunks."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from backend.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class VectorStore:
    """A FAISS-powered vector index that maps vectors to document chunks.

    The index stores normalised embedding vectors and supports fast
    approximate nearest-neighbour search via inner-product (equivalent
    to cosine similarity for L2-normalised vectors).

    Usage:
        store = VectorStore()
        store.add(chunks, vectors)
        results = store.search(query_vector, top_k=4)
        store.save("data/vector_store/index.faiss")
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.chunks: list[DocumentChunk] = []
        self._dimension: int = 0

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.chunks)

    def add(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None:
        """Add document chunks and their embedding vectors to the index.

        Args:
            chunks: List of DocumentChunk objects, parallel to vectors.
            vectors: 2-D numpy array of shape (len(chunks), embedding_dim).

        Raises:
            VectorStoreError: If chunk/vector count mismatches.
        """
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                f"Chunk count ({len(chunks)}) and vector count "
                f"({vectors.shape[0]}) must match."
            )

        if self.index is None:
            self._dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self._dimension)

        if vectors.shape[1] != self._dimension:
            raise VectorStoreError(
                f"Vector dimension ({vectors.shape[1]}) does not match "
                f"index dimension ({self._dimension})."
            )

        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info("Added %d vectors to index (total: %d)", len(chunks), self.size)

    def search(
        self, query_vector: np.ndarray, top_k: int = 4
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve the top-k most similar chunks for a query vector.

        Args:
            query_vector: 1-D or 2-D numpy array of the query embedding.
            top_k: Number of results to return.

        Returns:
            List of (DocumentChunk, similarity_score) tuples, sorted
            by score descending.

        Raises:
            VectorStoreError: If the index is empty.
        """
        if self.index is None or self.index.ntotal == 0:
            raise VectorStoreError("Cannot search an empty index.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, top_k)

        results: list[tuple[DocumentChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append((chunk, float(dist)))

        return results

    def save(self, directory: str) -> None:
        """Persist the index and chunk data to disk.

        Creates two files:
            <directory>/index.faiss  — FAISS binary index
            <directory>/chunks.pkl   — serialised chunk metadata

        Args:
            directory: Path to a directory (will be created if needed).
        """
        store_path = Path(directory)
        store_path.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise VectorStoreError("Nothing to save — index is empty.")

        faiss.write_index(self.index, str(store_path / "index.faiss"))

        with open(store_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("Vector store saved to %s", directory)

    def load(self, directory: str) -> None:
        """Restore a previously saved index and chunk data.

        Args:
            directory: Path containing index.faiss and chunks.pkl.
        """
        store_path = Path(directory)
        index_file = store_path / "index.faiss"
        chunks_file = store_path / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            raise VectorStoreError(
                f"Vector store not found at {directory}. "
                f"Expected {index_file.name} and {chunks_file.name}."
            )

        self.index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)

        self._dimension = self.index.d
        logger.info(
            "Vector store loaded from %s (%d vectors, %d dims)",
            directory,
            self.size,
            self._dimension,
        )
