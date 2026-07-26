"""Retrieval module that connects question → embedding → vector search."""

import logging
from typing import Optional

import numpy as np

from backend.chunker import DocumentChunk
from backend.embeddings import EmbeddingClient, EmbeddingError
from backend.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when the retrieval pipeline fails."""


class Retriever:
    """Orchestrates embedding and vector search to find relevant chunks.

    Given a question, this module:
    1. Embeds it using EmbeddingClient
    2. Searches the VectorStore for the most similar chunks
    3. Returns the top-k results with similarity scores

    Usage:
        retriever = Retriever(embedder, vector_store)
        results = retriever.retrieve("What is RAG?")
        for chunk, score in results:
            print(f"{chunk.document_name} p.{chunk.page_number} ({score:.3f})")
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingClient] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.embedder = embedder or EmbeddingClient()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self, question: str, top_k: int = 4
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve the top-k most relevant chunks for a question.

        Args:
            question: The user's question as plain text.
            top_k: How many chunks to retrieve.

        Returns:
            List of (DocumentChunk, similarity_score) tuples sorted by
            score descending.

        Raises:
            RetrievalError: If embedding or search fails.
        """
        try:
            query_vector = self.embedder.encode(question)
        except EmbeddingError as exc:
            raise RetrievalError(f"Failed to embed question: {exc}") from exc

        try:
            results = self.vector_store.search(query_vector, top_k=top_k)
        except VectorStoreError as exc:
            raise RetrievalError(f"Failed to search vector store: {exc}") from exc

        return results
