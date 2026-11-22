"""Unit tests for PDF extraction and page metadata."""

from pathlib import Path

import pytest

from app.utils import file_processor
from app.utils.file_processor import DocumentProcessingError, process_pdf


class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakeReader:
    def __init__(self, _path: str, strict: bool = False) -> None:
        assert strict is False
        self.is_encrypted = False
        self.pages = [FakePage(" First page text "), FakePage(None), FakePage("Second page")]


def test_process_pdf_keeps_original_page_numbers_and_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "stored-uuid.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(file_processor, "PdfReader", FakeReader)

    pages = process_pdf(
        pdf_path,
        source_filename="annual-report.pdf",
        document_id="doc-123",
    )

    assert len(pages) == 2
    assert pages[0]["page_content"] == "First page text"
    assert pages[0]["metadata"] == {
        "source": "annual-report.pdf",
        "filename": "annual-report.pdf",
        "page": 1,
        "total_pages": 3,
        "document_id": "doc-123",
    }
    assert pages[1]["metadata"]["page"] == 3


def test_process_pdf_rejects_non_pdf_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(DocumentProcessingError, match="Only PDF"):
        process_pdf(file_path)


def test_process_pdf_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        process_pdf(tmp_path / "missing.pdf")


def test_process_pdf_rejects_encrypted_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EncryptedReader:
        def __init__(self, _path: str, strict: bool = False) -> None:
            self.is_encrypted = True
            self.pages = []

        def decrypt(self, _password: str) -> int:
            return 0

    file_path = tmp_path / "locked.pdf"
    file_path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(file_processor, "PdfReader", EncryptedReader)

    with pytest.raises(DocumentProcessingError, match="Password-protected"):
        process_pdf(file_path)


def test_blank_pdf_pages_are_not_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BlankReader:
        def __init__(self, _path: str, strict: bool = False) -> None:
            self.is_encrypted = False
            self.pages = [FakePage(" "), FakePage("")]

    file_path = tmp_path / "blank.pdf"
    file_path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(file_processor, "PdfReader", BlankReader)

    assert process_pdf(file_path) == []
