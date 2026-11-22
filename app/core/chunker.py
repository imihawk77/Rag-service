"""Character-based document chunking with real sliding-window overlap."""

from typing import Any


class SmartChunker:
    """Split page documents into bounded chunks while preserving metadata.

    Chunk sizes and overlap are measured in characters, not model tokens.
    Splitting prefers paragraph, line, sentence, and word boundaries; long
    unbroken strings are split at the configured hard limit.
    """

    _BOUNDARIES = ("\n\n", "\n", ". ", " ")

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _find_end(self, text: str, start: int) -> int:
        hard_end = min(start + self.chunk_size, len(text))
        if hard_end == len(text):
            return hard_end

        minimum_end = start + max(1, int(self.chunk_size * 0.6))
        candidates: list[int] = []
        for boundary in self._BOUNDARIES:
            position = text.rfind(boundary, minimum_end, hard_end)
            if position >= 0:
                candidates.append(position + len(boundary))

        return max(candidates) if candidates else hard_end

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chunk extracted pages; output remains compatible with the original API."""
        chunks: list[dict[str, Any]] = []

        for document in documents:
            text = document.get("page_content", "")
            metadata = document.get("metadata", {})
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(metadata, dict):
                metadata = {}

            document_key = str(
                metadata.get("document_id") or metadata.get("source") or "unknown"
            )
            page_number = metadata.get("page", 1)
            start = 0
            page_chunk_index = 0

            while start < len(text):
                end = self._find_end(text, start)
                raw_chunk = text[start:end]
                leading_whitespace = len(raw_chunk) - len(raw_chunk.lstrip())
                chunk_text = raw_chunk.strip()

                if chunk_text:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_id": f"{document_key}_p{page_number}_c{page_chunk_index}",
                            "chunk_index": len(chunks),
                            "page_chunk_index": page_chunk_index,
                            "start_char": start + leading_whitespace,
                            "end_char": start + leading_whitespace + len(chunk_text),
                        }
                    )
                    chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": chunk_metadata,
                            "embedding": None,
                        }
                    )
                    page_chunk_index += 1

                if end >= len(text):
                    break
                next_start = max(start + 1, end - self.chunk_overlap)
                start = next_start

        return chunks
