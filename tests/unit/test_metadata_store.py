"""Unit tests for SQLite document and chunk metadata persistence."""

from pathlib import Path

from app.storage.metadata_store import SQLiteMetadataStore


def test_document_and_chunk_metadata_persist(tmp_path: Path) -> None:
    database = tmp_path / "metadata.db"
    store = SQLiteMetadataStore(database)
    store.upsert_document(
        document_id="doc-1",
        filename="guide.pdf",
        file_path="/local/doc-1.pdf",
        page_count=4,
    )
    store.replace_document_chunks(
        "doc-1",
        [
            {
                "text": "Relevant passage",
                "faiss_id": 101,
                "metadata": {"chunk_id": "chunk-1", "page": 3},
            }
        ],
    )

    reopened = SQLiteMetadataStore(database)
    documents = reopened.list_documents()
    chunk = reopened.get_chunk("chunk-1")

    assert documents[0]["document_id"] == "doc-1"
    assert documents[0]["chunk_count"] == 1
    assert chunk is not None
    assert chunk["faiss_id"] == 101
    assert chunk["metadata"]["page"] == 3


def test_chunk_replacement_is_scoped_to_document(tmp_path: Path) -> None:
    store = SQLiteMetadataStore(tmp_path / "metadata.db")
    store.upsert_document(
        document_id="doc-1", filename="one.pdf", file_path="/one.pdf"
    )
    store.upsert_document(
        document_id="doc-2", filename="two.pdf", file_path="/two.pdf"
    )
    store.replace_document_chunks(
        "doc-1", [{"text": "first", "metadata": {"chunk_id": "one"}}]
    )
    store.replace_document_chunks(
        "doc-1", [{"text": "replacement", "metadata": {"chunk_id": "one-new"}}]
    )

    assert store.get_chunk("one") is None
    assert store.get_chunk("one-new")["text"] == "replacement"
