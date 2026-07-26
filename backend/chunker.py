"""Text chunking that splits documents into overlapping segments."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings

settings = Settings()

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single text chunk with its source provenance.

    Every chunk remembers exactly which document and page it came from,
    enabling source citation downstream.
    """

    chunk_id: str
    text: str
    document_name: str
    page_number: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Splits extracted PDF pages into overlapping chunks.

    Splitting respects word boundaries (no mid-word cuts) and configurable
    overlap to preserve context at chunk seams.

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk(pages)  # pages is List[PDFPage]
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Chunk overlap ({self.chunk_overlap}) must be less than "
                f"chunk size ({self.chunk_size})."
            )

    # ── Public API ──────────────────────────────────────────────────────

    def chunk_pages(self, pages: list) -> list[DocumentChunk]:
        """Chunk a list of PDFPage-like objects.

        Each page is split independently so chunks never cross page
        boundaries (page number is a critical citation anchor).

        Args:
            pages: Iterable of objects with .text, .page_number,
                   .document_name, and optional .metadata attributes.

        Returns:
            A flat list of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []
        for page in pages:
            page_chunks = self._split_page(page)
            chunks.extend(page_chunks)
        return chunks

    # ── Private helpers ─────────────────────────────────────────────────

    def _split_page(self, page) -> list[DocumentChunk]:
        """Split a single page into overlapping chunks."""
        text = page.text
        if not text.strip():
            return []

        page_chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0
        last_end = 0

        while start < len(text):
            end = self._find_split_end(text, start)

            if end <= last_end:
                break

            chunk_text = text[start:end].strip()
            if chunk_text:
                page_chunks.append(
                    DocumentChunk(
                        chunk_id=self._build_id(
                            page.document_name, page.page_number, chunk_index
                        ),
                        text=chunk_text,
                        document_name=page.document_name,
                        page_number=page.page_number,
                        metadata=getattr(page, "metadata", {}),
                    )
                )
                chunk_index += 1

            last_end = end
            start = max(end - self.chunk_overlap, 0)

        return page_chunks

    def _find_split_end(self, text: str, start: int) -> int:
        """Determine where the current chunk ends.

        Tries to break at a whitespace boundary near chunk_size.
        If the remaining text fits, returns the end of text.
        """
        if start + self.chunk_size >= len(text):
            return len(text)

        end = start + self.chunk_size

        while end > start and not text[end].isspace():
            end -= 1

        if end == start:
            end = start + self.chunk_size

        return end

    @staticmethod
    def _build_id(doc_name: str, page: int, index: int) -> str:
        """Create a globally unique chunk identifier."""
        safe_name = doc_name.replace(" ", "_").replace("/", "_")
        return f"{safe_name}::p{page}::c{index}"
