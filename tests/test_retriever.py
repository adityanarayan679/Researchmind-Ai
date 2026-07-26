"""Tests for the retriever module."""

from backend.chunker import DocumentChunk
from backend.embeddings import EmbeddingClient
from backend.retriever import Retriever, RetrievalError
from backend.vector_store import VectorStore

_embedder = EmbeddingClient()


def _seed_vector_store() -> VectorStore:
    store = VectorStore()
    chunks = [
        DocumentChunk("c0", "The cat sat on the mat", "doc1.pdf", 1),
        DocumentChunk("c1", "Dogs love to play fetch", "doc2.pdf", 1),
        DocumentChunk("c2", "Quantum computing uses qubits", "doc3.pdf", 2),
    ]
    vectors = _embedder.encode_batch([c.text for c in chunks])
    store.add(chunks, vectors)
    return store


def test_retrieve_returns_relevant_chunks():
    store = _seed_vector_store()
    retriever = Retriever(_embedder, store)
    results = retriever.retrieve("Tell me about cats", top_k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "c0"


def test_retrieve_returns_empty_for_empty_store():
    store = VectorStore()
    retriever = Retriever(_embedder, store)
    try:
        retriever.retrieve("anything")
        assert False, "Expected RetrievalError"
    except RetrievalError:
        pass


def test_retrieve_respects_top_k():
    store = _seed_vector_store()
    retriever = Retriever(_embedder, store)
    results = retriever.retrieve("pets and animals", top_k=1)
    assert len(results) == 1


if __name__ == "__main__":
    test_retrieve_returns_relevant_chunks()
    print("PASS: test_retrieve_returns_relevant_chunks")
    test_retrieve_returns_empty_for_empty_store()
    print("PASS: test_retrieve_returns_empty_for_empty_store")
    test_retrieve_respects_top_k()
    print("PASS: test_retrieve_respects_top_k")
    print("\nAll tests passed.")
