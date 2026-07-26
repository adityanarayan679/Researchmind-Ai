"""Tests for the FAISS vector store module."""

import os
import tempfile

import numpy as np

from backend.chunker import DocumentChunk
from backend.vector_store import VectorStore, VectorStoreError


def _make_chunks(n: int) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"c{i}", text=f"chunk {i}", document_name="test.pdf", page_number=1
        )
        for i in range(n)
    ]


def _make_vectors(n: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_add_and_search():
    store = VectorStore()
    chunks = _make_chunks(10)
    vectors = _make_vectors(10)
    store.add(chunks, vectors)
    assert store.size == 10

    results = store.search(vectors[0], top_k=3)
    assert len(results) == 3
    assert results[0][0].chunk_id == "c0"
    assert results[0][1] > 0.99


def test_search_returns_all_if_top_k_exceeds_size():
    store = VectorStore()
    chunks = _make_chunks(3)
    vectors = _make_vectors(3)
    store.add(chunks, vectors)
    results = store.search(vectors[0], top_k=10)
    assert len(results) == 3


def test_empty_index_raises_error():
    store = VectorStore()
    try:
        store.search(np.array([0.0] * 384))
        assert False, "Expected VectorStoreError"
    except VectorStoreError:
        pass


def test_dimension_mismatch_raises_error():
    store = VectorStore()
    chunks = _make_chunks(5)
    vectors = _make_vectors(5, dim=384)
    store.add(chunks, vectors)
    bad_vectors = _make_vectors(3, dim=128)
    try:
        store.add(_make_chunks(3), bad_vectors)
        assert False, "Expected VectorStoreError"
    except VectorStoreError:
        pass


def test_save_and_load():
    store = VectorStore()
    chunks = _make_chunks(5)
    vectors = _make_vectors(5)
    store.add(chunks, vectors)

    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)
        loaded = VectorStore()
        loaded.load(tmpdir)

    assert loaded.size == 5
    results = loaded.search(vectors[0], top_k=1)
    assert results[0][0].chunk_id == "c0"


if __name__ == "__main__":
    test_add_and_search()
    print("PASS: test_add_and_search")
    test_search_returns_all_if_top_k_exceeds_size()
    print("PASS: test_search_returns_all_if_top_k_exceeds_size")
    test_empty_index_raises_error()
    print("PASS: test_empty_index_raises_error")
    test_dimension_mismatch_raises_error()
    print("PASS: test_dimension_mismatch_raises_error")
    test_save_and_load()
    print("PASS: test_save_and_load")
    print("\nAll tests passed.")
