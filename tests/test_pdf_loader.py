"""Tests for the PDF loader module."""

import os
import tempfile
from pathlib import Path

import fitz

from backend.pdf_loader import PDFLoader, PDFLoadError


def _create_test_pdf(text_pages: list[str]) -> str:
    """Create a temporary PDF with one page per string in text_pages."""
    doc = fitz.open()
    for text in text_pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(path)
    doc.close()
    return path


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def test_extracts_text_from_single_page():
    path = _create_test_pdf(["Hello world"])
    try:
        loader = PDFLoader()
        pages = loader.load(path)
        assert len(pages) == 1
        assert pages[0].text.strip() == "Hello world"
        assert pages[0].page_number == 1
        assert pages[0].total_pages == 1
    finally:
        _cleanup(path)


def test_extracts_text_from_multiple_pages():
    path = _create_test_pdf(["Page one content", "Page two content"])
    try:
        loader = PDFLoader()
        pages = loader.load(path)
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
        assert pages[0].document_name == Path(path).name
    finally:
        _cleanup(path)


def test_raises_error_for_missing_file():
    loader = PDFLoader()
    try:
        loader.load("nonexistent.pdf")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_raises_error_for_corrupted_pdf():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    with open(path, "w") as f:
        f.write("not a real pdf")
    loader = PDFLoader()
    try:
        loader.load(path)
        assert False, "Expected PDFLoadError"
    except PDFLoadError:
        pass
    finally:
        _cleanup(path)


def test_load_all_skips_bad_files():
    good = _create_test_pdf(["Good content"])
    try:
        loader = PDFLoader()
        pages = loader.load_all(["nonexistent.pdf", good])
        assert len(pages) == 1
        assert pages[0].text.strip() == "Good content"
    finally:
        _cleanup(good)


if __name__ == "__main__":
    test_extracts_text_from_single_page()
    print("PASS: test_extracts_text_from_single_page")
    test_extracts_text_from_multiple_pages()
    print("PASS: test_extracts_text_from_multiple_pages")
    test_raises_error_for_missing_file()
    print("PASS: test_raises_error_for_missing_file")
    test_raises_error_for_corrupted_pdf()
    print("PASS: test_raises_error_for_corrupted_pdf")
    test_load_all_skips_bad_files()
    print("PASS: test_load_all_skips_bad_files")
    print("\nAll tests passed.")
