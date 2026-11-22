"""Unit tests for safe local PDF persistence."""

import uuid
from pathlib import Path

import pytest

from app.storage.document_store import DocumentStoreError, LocalDocumentStore


PDF_BYTES = b"%PDF-1.4\n% test fixture"


def test_save_uses_generated_uuid_and_can_read_back(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)
    path = store.save(PDF_BYTES, original_filename="../../unsafe-name.pdf")

    assert path.parent == tmp_path
    assert path.suffix == ".pdf"
    assert path.name != "unsafe-name.pdf"
    assert path.read_bytes() == PDF_BYTES
    assert store.get_path(path.stem) == path


def test_save_accepts_explicit_uuid(tmp_path: Path) -> None:
    document_id = str(uuid.uuid4())
    store = LocalDocumentStore(tmp_path)

    path = store.save(PDF_BYTES, original_filename="source.PDF", document_id=document_id)

    assert path.name == f"{document_id}.pdf"


def test_save_rejects_invalid_uploads(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)

    with pytest.raises(DocumentStoreError, match="Only PDF"):
        store.save(b"content", original_filename="malware.exe")
    with pytest.raises(DocumentStoreError, match="empty"):
        store.save(b"", original_filename="empty.pdf")
    with pytest.raises(DocumentStoreError, match="PDF header"):
        store.save(b"not a pdf", original_filename="fake.pdf")


def test_save_enforces_size_limit(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path, max_file_size_bytes=5)

    with pytest.raises(DocumentStoreError, match="maximum"):
        store.save(PDF_BYTES, original_filename="large.pdf")


def test_document_id_must_be_uuid(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)

    with pytest.raises(DocumentStoreError, match="valid UUID"):
        store.get_path("../../outside")


def test_delete_returns_false_for_missing_and_true_for_existing(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)
    document_id = str(uuid.uuid4())

    assert store.delete(document_id) is False
    path = store.save(PDF_BYTES, original_filename="source.pdf", document_id=document_id)
    assert path.exists()
    assert store.delete(document_id) is True
    assert not path.exists()
