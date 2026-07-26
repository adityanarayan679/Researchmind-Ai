"""Tests for the citation module."""

from backend.chunker import DocumentChunk
from backend.citations import CitationBuilder


def test_build_deduplicates_same_page():
    builder = CitationBuilder()
    results = [
        (DocumentChunk("c0", "text", "paper.pdf", 3), 0.95),
        (DocumentChunk("c1", "more text", "paper.pdf", 3), 0.90),
        (DocumentChunk("c2", "other text", "paper.pdf", 5), 0.85),
    ]
    citations = builder.build(results)
    assert len(citations) == 2
    assert citations[0].page_number == 3
    assert citations[1].page_number == 5


def test_build_sorts_by_score_descending():
    builder = CitationBuilder()
    results = [
        (DocumentChunk("c0", "text", "a.pdf", 1), 0.70),
        (DocumentChunk("c1", "text", "b.pdf", 1), 0.95),
        (DocumentChunk("c2", "text", "c.pdf", 1), 0.85),
    ]
    citations = builder.build(results)
    scores = [c.similarity_score for c in citations]
    assert scores == sorted(scores, reverse=True)


def test_build_returns_empty_for_empty_results():
    builder = CitationBuilder()
    citations = builder.build([])
    assert citations == []


def test_to_markdown():
    builder = CitationBuilder()
    results = [
        (DocumentChunk("c0", "text", "paper.pdf", 3), 0.9532),
    ]
    citations = builder.build(results)
    md = builder.to_markdown(citations)
    assert "**Sources:**" in md
    assert "paper.pdf" in md
    assert "Page 3" in md
    assert "0.953" in md


def test_to_inline():
    builder = CitationBuilder()
    results = [
        (DocumentChunk("c0", "text", "paper.pdf", 3), 0.95),
    ]
    citations = builder.build(results)
    inline = builder.to_inline(citations)
    assert "paper.pdf, p.3" in inline


if __name__ == "__main__":
    test_build_deduplicates_same_page()
    print("PASS: test_build_deduplicates_same_page")
    test_build_sorts_by_score_descending()
    print("PASS: test_build_sorts_by_score_descending")
    test_build_returns_empty_for_empty_results()
    print("PASS: test_build_returns_empty_for_empty_results")
    test_to_markdown()
    print("PASS: test_to_markdown")
    test_to_inline()
    print("PASS: test_to_inline")
    print("\nAll tests passed.")
