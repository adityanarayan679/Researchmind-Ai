"""Tests for the text chunker module."""

import tempfile
from dataclasses import dataclass, field

from backend.chunker import TextChunker, DocumentChunk


@dataclass
class FakePage:
    """Minimal stand-in for a PDFPage during testing."""
    text: str
    page_number: int = 1
    document_name: str = "test.pdf"
    metadata: dict = field(default_factory=dict)


def test_single_short_page_returns_one_chunk():
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
    pages = [FakePage(text="Hello world")]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].document_name == "test.pdf"
    assert chunks[0].page_number == 1


def test_long_text_is_split():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "word ".join(["hello"] * 30)
    pages = [FakePage(text=text)]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 60


def test_chunks_retain_page_numbers():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    pages = [
        FakePage(text="Page one content " * 20, page_number=1),
        FakePage(text="Page two content " * 20, page_number=2),
    ]
    chunks = chunker.chunk_pages(pages)
    assert any(c.page_number == 1 for c in chunks)
    assert any(c.page_number == 2 for c in chunks)


def test_chunk_ids_are_unique():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    pages = [
        FakePage(text="Content " * 30, page_number=1),
        FakePage(text="Content " * 30, page_number=2),
    ]
    chunks = chunker.chunk_pages(pages)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_text_returns_no_chunks():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    pages = [FakePage(text="   ")]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) == 0


def test_overlap_less_than_size():
    try:
        TextChunker(chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_single_short_page_returns_one_chunk()
    print("PASS: test_single_short_page_returns_one_chunk")
    test_long_text_is_split()
    print("PASS: test_long_text_is_split")
    test_chunks_retain_page_numbers()
    print("PASS: test_chunks_retain_page_numbers")
    test_chunk_ids_are_unique()
    print("PASS: test_chunk_ids_are_unique")
    test_empty_text_returns_no_chunks()
    print("PASS: test_empty_text_returns_no_chunks")
    test_overlap_less_than_size()
    print("PASS: test_overlap_less_than_size")
    print("\nAll tests passed.")
