"""Unit tests for chunk boundaries, overlap, and metadata."""

import pytest

from app.core.chunker import SmartChunker


def test_long_text_is_split_and_respects_maximum_size() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 5
    chunks = SmartChunker(chunk_size=20, chunk_overlap=5).chunk_documents(
        [{"page_content": text, "metadata": {"source": "notes.pdf", "page": 2}}]
    )

    assert len(chunks) > 1
    assert all(0 < len(chunk["text"]) <= 20 for chunk in chunks)
    assert chunks[0]["text"] + "".join(chunk["text"][5:] for chunk in chunks[1:]) == text


def test_overlap_is_present_between_adjacent_chunks() -> None:
    text = "0123456789" * 5
    chunks = SmartChunker(chunk_size=12, chunk_overlap=4).chunk_documents(
        [{"page_content": text, "metadata": {"source": "a.pdf"}}]
    )

    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:]):
        assert previous["text"][-4:] == current["text"][:4]


def test_long_paragraph_is_split_without_losing_content() -> None:
    text = "A" * 37
    chunks = SmartChunker(chunk_size=10, chunk_overlap=0).chunk_documents(
        [{"page_content": text, "metadata": {}}]
    )

    assert [chunk["text"] for chunk in chunks] == ["A" * 10, "A" * 10, "A" * 10, "A" * 7]


def test_metadata_and_page_information_are_preserved() -> None:
    chunks = SmartChunker(chunk_size=12, chunk_overlap=2).chunk_documents(
        [
            {
                "page_content": "This is page content that must be chunked.",
                "metadata": {"document_id": "doc-1", "source": "report.pdf", "page": 7},
            }
        ]
    )

    assert chunks
    assert all(chunk["metadata"]["page"] == 7 for chunk in chunks)
    assert all(chunk["metadata"]["document_id"] == "doc-1" for chunk in chunks)
    assert all(chunk["metadata"]["source"] == "report.pdf" for chunk in chunks)
    assert len({chunk["metadata"]["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["embedding"] is None for chunk in chunks)


def test_empty_documents_are_skipped() -> None:
    chunks = SmartChunker().chunk_documents(
        [{"page_content": "  \n ", "metadata": {"source": "empty.pdf"}}]
    )

    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size: int, chunk_overlap: int
) -> None:
    with pytest.raises(ValueError):
        SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
