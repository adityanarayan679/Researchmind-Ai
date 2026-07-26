"""Tests for the embedding module."""

import numpy as np

from backend.embeddings import EmbeddingClient, EmbeddingError


def test_encode_single_text_returns_vector():
    client = EmbeddingClient()
    vec = client.encode("What is machine learning?")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == client.embedding_dimension


def test_encode_batch_returns_matrix():
    client = EmbeddingClient()
    texts = ["first chunk", "second chunk", "third chunk"]
    vecs = client.encode_batch(texts)
    assert vecs.shape == (3, client.embedding_dimension)


def test_cache_prevents_recomputation():
    client = EmbeddingClient()
    text = "Cache me if you can"
    vec1 = client.encode(text)
    vec2 = client.encode(text)
    assert np.array_equal(vec1, vec2)


def test_similar_texts_have_similar_vectors():
    client = EmbeddingClient()
    v1 = client.encode("The cat sat on the mat")
    v2 = client.encode("A cat is sitting on a rug")
    v3 = client.encode("Quantum computing is fascinating")
    sim_similar = float(np.dot(v1, v2))
    sim_different = float(np.dot(v1, v3))
    assert sim_similar > sim_different, (
        f"Similar texts ({sim_similar}) should be closer than "
        f"different texts ({sim_different})"
    )


if __name__ == "__main__":
    test_encode_single_text_returns_vector()
    print("PASS: test_encode_single_text_returns_vector")
    test_encode_batch_returns_matrix()
    print("PASS: test_encode_batch_returns_matrix")
    test_cache_prevents_recomputation()
    print("PASS: test_cache_prevents_recomputation")
    test_similar_texts_have_similar_vectors()
    print("PASS: test_similar_texts_have_similar_vectors")
    print("\nAll tests passed.")
