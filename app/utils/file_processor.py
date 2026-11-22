"""PDF text extraction with stable document and page metadata."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentProcessingError(ValueError):
    """Raised when a document cannot be safely parsed."""


def process_pdf(
    file_path: str | Path,
    *,
    source_filename: str | None = None,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract non-empty PDF pages while retaining their original page numbers.

    source_filename should be the user's original filename when the PDF is
    processed from a temporary or UUID-based storage path.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    if path.suffix.lower() != ".pdf":
        raise DocumentProcessingError("Only PDF documents are supported")

    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise DocumentProcessingError("Encrypted PDF is not supported") from exc
            if not decrypted:
                raise DocumentProcessingError("Password-protected PDF is not supported")

        total_pages = len(reader.pages)
        filename = source_filename or path.name
        documents: list[dict[str, Any]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not isinstance(page_text, str) or not page_text.strip():
                continue

            metadata: dict[str, Any] = {
                "source": filename,
                "filename": filename,
                "page": page_number,
                "total_pages": total_pages,
            }
            if document_id is not None:
                metadata["document_id"] = document_id

            documents.append(
                {
                    "page_content": page_text.strip(),
                    "metadata": metadata,
                }
            )

        return documents
    except DocumentProcessingError:
        raise
    except (PdfReadError, OSError) as exc:
        raise DocumentProcessingError("Unable to read PDF document") from exc
    except Exception as exc:
        raise DocumentProcessingError("Failed to extract text from PDF document") from exc
