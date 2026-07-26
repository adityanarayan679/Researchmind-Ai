"""Citation formatting for transparent source attribution."""

import logging
from dataclasses import dataclass
from typing import Optional

from backend.chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single source reference with document, page, and relevance score."""

    document_name: str
    page_number: int
    similarity_score: float


class CitationBuilder:
    """Builds and formats source citations from retrieval results.

    Handles deduplication (same document + page appearing in multiple chunks),
    ordering by relevance, and multiple output formats.

    Usage:
        builder = CitationBuilder()
        citations = builder.build(results)
        markdown = builder.to_markdown(citations)
    """

    def build(self, results: list[tuple[DocumentChunk, float]]) -> list[Citation]:
        """Convert retrieval results into a deduplicated, sorted citation list.

        Args:
            results: List of (DocumentChunk, similarity_score) from the retriever.

        Returns:
            List of Citation objects, sorted by similarity_score descending,
            with duplicates (same document_name + page_number) removed.
        """
        seen: set[tuple[str, int]] = set()
        citations: list[Citation] = []

        for chunk, score in results:
            key = (chunk.document_name, chunk.page_number)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        similarity_score=score,
                    )
                )

        citations.sort(key=lambda c: c.similarity_score, reverse=True)
        return citations

    # ── Output formats ──────────────────────────────────────────────────

    @staticmethod
    def to_markdown(citations: list[Citation]) -> str:
        """Render citations as a Markdown block.

        Example output:
            **Sources:**
            1. paper.pdf — Page 3 (relevance: 0.95)
            2. paper.pdf — Page 5 (relevance: 0.89)
        """
        if not citations:
            return "*No sources available.*"

        lines = ["**Sources:**"]
        for i, c in enumerate(citations, start=1):
            lines.append(
                f"{i}. {c.document_name} — Page {c.page_number} "
                f"(relevance: {c.similarity_score:.3f})"
            )
        return "\n".join(lines)

    @staticmethod
    def to_inline(citations: list[Citation]) -> str:
        """Render citations as a compact inline string.

        Example output:
            [paper.pdf, p.3; paper.pdf, p.5]
        """
        if not citations:
            return ""

        parts = [
            f"{c.document_name}, p.{c.page_number}" for c in citations
        ]
        return f"[{' | '.join(parts)}]"
