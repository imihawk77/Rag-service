"""Safe local storage for uploaded PDF documents."""

import os
import tempfile
import uuid
from pathlib import Path


class DocumentStoreError(ValueError):
    """Raised when a document cannot be stored or addressed safely."""


class LocalDocumentStore:
    """Store PDFs under generated UUID names, never under user-supplied paths."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        max_file_size_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be greater than zero")
        self.root_dir = Path(root_dir)
        self.max_file_size_bytes = max_file_size_bytes
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        content: bytes,
        *,
        original_filename: str,
        document_id: str | None = None,
    ) -> Path:
        """Atomically persist a PDF and return its generated local path."""
        if Path(original_filename).suffix.lower() != ".pdf":
            raise DocumentStoreError("Only PDF files are supported")
        if not content:
            raise DocumentStoreError("Uploaded document is empty")
        if len(content) > self.max_file_size_bytes:
            raise DocumentStoreError("Uploaded document exceeds the maximum allowed size")
        if b"%PDF-" not in content[:1024]:
            raise DocumentStoreError("Uploaded file does not have a valid PDF header")

        try:
            safe_id = str(uuid.UUID(document_id)) if document_id else str(uuid.uuid4())
        except (ValueError, AttributeError) as exc:
            raise DocumentStoreError("document_id must be a valid UUID") from exc

        destination = self.root_dir / f"{safe_id}.pdf"
        if destination.exists():
            raise DocumentStoreError("A document with this ID already exists")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{safe_id}.",
                suffix=".tmp",
                dir=self.root_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, destination)
            return destination
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def get_path(self, document_id: str) -> Path:
        """Resolve a document UUID to its path without accepting arbitrary paths."""
        try:
            safe_id = str(uuid.UUID(document_id))
        except (ValueError, AttributeError) as exc:
            raise DocumentStoreError("document_id must be a valid UUID") from exc

        path = self.root_dir / f"{safe_id}.pdf"
        if not path.is_file():
            raise FileNotFoundError("Document not found")
        return path

    def delete(self, document_id: str) -> bool:
        """Delete a stored document; return False when it does not exist."""
        try:
            path = self.get_path(document_id)
        except FileNotFoundError:
            return False
        path.unlink()
        return True
