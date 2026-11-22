"""SQLite metadata repository for documents and their indexed chunks."""

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteMetadataStore:
    """Small synchronous SQLite repository; each operation uses its own connection."""

    def __init__(self, database_path: str | Path) -> None:
        raw_path = str(database_path)
        if raw_path.startswith("sqlite:///"):
            raw_path = raw_path.removeprefix("sqlite:///")
        self.database_path = Path(raw_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        """Create metadata tables and indexes if they do not exist."""
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    faiss_id INTEGER,
                    page INTEGER,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_faiss_id
                    ON chunks(faiss_id) WHERE faiss_id IS NOT NULL;
                """
            )

    def upsert_document(
        self,
        *,
        document_id: str,
        filename: str,
        file_path: str,
        status: str = "processed",
        page_count: int = 0,
        chunk_count: int = 0,
    ) -> None:
        """Insert or update document metadata."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (document_id, filename, file_path, status, page_count, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename=excluded.filename,
                    file_path=excluded.file_path,
                    status=excluded.status,
                    page_count=excluded.page_count,
                    chunk_count=excluded.chunk_count,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (document_id, filename, file_path, status, page_count, chunk_count),
            )

    def replace_document_chunks(
        self, document_id: str, chunks: list[dict[str, Any]]
    ) -> None:
        """Replace a document's chunk rows in one SQLite transaction."""
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown document_id: {document_id}")

            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            for position, chunk in enumerate(chunks):
                metadata = chunk.get("metadata", {})
                chunk_id = str(metadata.get("chunk_id") or f"{document_id}:{position}")
                connection.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, document_id, faiss_id, page, text, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        chunk.get("faiss_id"),
                        metadata.get("page"),
                        str(chunk.get("text", "")),
                        json.dumps(metadata, ensure_ascii=False, default=str),
                    ),
                )
            connection.execute(
                """
                UPDATE documents
                SET chunk_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (len(chunks), document_id),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        """Return document metadata ordered by most recently updated."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, filename, file_path, status, page_count,
                       chunk_count, created_at, updated_at
                FROM documents ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch one chunk and decode its metadata."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT chunk_id, document_id, faiss_id, page, text, metadata_json
                FROM chunks WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result
