"""PDF text extraction that preserves page numbers and document metadata."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


@dataclass
class PDFPage:
    """A single page of extracted text with its source information.

    This structured representation lets downstream modules (chunker,
    retriever, citations) always know exactly where text came from.
    """

    text: str
    page_number: int
    document_name: str
    total_pages: int
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """Loads PDF files and extracts text on a per-page basis.

    Usage:
        loader = PDFLoader()
        pages = loader.load("paper.pdf")
        for page in pages:
            print(page.text)
    """

    def load(self, file_path: str) -> list[PDFPage]:
        """Extract text from every page of a single PDF.

        Args:
            file_path: Path to the PDF file on disk.

        Returns:
            A list of PDFPage objects, one per page.

        Raises:
            PDFLoadError: If the file is missing, corrupted, encrypted,
                          or has no selectable text.
            FileNotFoundError: If the path does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise PDFLoadError(
                f"Cannot open PDF (corrupted or invalid format): {exc}"
            ) from exc

        if doc.is_encrypted:
            doc.close()
            raise PDFLoadError(
                f"Cannot process encrypted PDF: {path.name}. "
                "Password-protected PDFs are not supported."
            )

        pages: list[PDFPage] = []
        total = doc.page_count

        if total == 0:
            doc.close()
            raise PDFLoadError(f"PDF is empty (0 pages): {path.name}")

        for page_num in range(total):
            page = doc.load_page(page_num)
            text = page.get_text().strip()

            if not text:
                logger.warning(
                    "Page %d of '%s' has no selectable text "
                    "(may be a scanned image).",
                    page_num + 1,
                    path.name,
                )

            pages.append(
                PDFPage(
                    text=text,
                    page_number=page_num + 1,
                    document_name=path.name,
                    total_pages=total,
                    metadata=self._extract_metadata(doc),
                )
            )

        doc.close()
        return pages

    def load_all(self, file_paths: list[str]) -> list[PDFPage]:
        """Load and extract text from multiple PDFs.

        Args:
            file_paths: List of paths to PDF files.

        Returns:
            A combined list of PDFPage objects from all documents.
        """
        all_pages: list[PDFPage] = []
        for fpath in file_paths:
            try:
                all_pages.extend(self.load(fpath))
            except (PDFLoadError, FileNotFoundError) as exc:
                logger.error("Skipping '%s': %s", fpath, exc)
        return all_pages

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_metadata(doc: fitz.Document) -> dict:
        """Extract useful metadata from an open PyMuPDF document."""
        raw = doc.metadata
        return {
            "title": raw.get("title", "") or "",
            "author": raw.get("author", "") or "",
            "subject": raw.get("subject", "") or "",
            "producer": raw.get("producer", "") or "",
        }
