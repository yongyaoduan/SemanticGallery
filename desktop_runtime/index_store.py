from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


class IndexStore:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def connect(cls, db_path: Path) -> "IndexStore":
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(db_path))

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS image_assets (
              content_hash TEXT PRIMARY KEY,
              byte_size INTEGER NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS image_embeddings (
              content_hash TEXT NOT NULL,
              encoder_signature TEXT NOT NULL,
              embedding_blob BLOB NOT NULL,
              embedding_dim INTEGER NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (content_hash, encoder_signature)
            );
            CREATE TABLE IF NOT EXISTS image_paths (
              path_id INTEGER PRIMARY KEY AUTOINCREMENT,
              absolute_path TEXT NOT NULL UNIQUE,
              folder_path TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              byte_size INTEGER NOT NULL,
              mtime_ns INTEGER NOT NULL,
              is_present INTEGER NOT NULL,
              last_scanned_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_image_paths_folder_present_order_content
              ON image_paths(folder_path, is_present, absolute_path, content_hash);
            CREATE INDEX IF NOT EXISTS idx_image_embeddings_signature_content
              ON image_embeddings(encoder_signature, content_hash);
            CREATE TABLE IF NOT EXISTS folder_states (
              folder_path TEXT PRIMARY KEY,
              active_encoder_signature TEXT NOT NULL,
              file_count INTEGER NOT NULL DEFAULT 0,
              total_bytes INTEGER NOT NULL DEFAULT 0,
              scan_signature TEXT NOT NULL DEFAULT '',
              last_scanned_at TEXT,
              last_synced_at TEXT
            );
            """
        )
        self.connection.commit()

    def upsert_asset(self, content_hash: str, byte_size: int) -> None:
        self.connection.execute(
            """
            INSERT INTO image_assets (content_hash, byte_size)
            VALUES (?, ?)
            ON CONFLICT(content_hash) DO UPDATE SET
              byte_size = excluded.byte_size,
              updated_at = CURRENT_TIMESTAMP
            """,
            (content_hash, byte_size),
        )

    def upsert_embedding(self, content_hash: str, encoder_signature: str, embedding: np.ndarray) -> None:
        embedding_array = np.asarray(embedding, dtype=np.float32)
        if embedding_array.ndim != 1:
            raise ValueError("embedding must be a 1-D vector")
        self.connection.execute(
            """
            INSERT INTO image_embeddings (content_hash, encoder_signature, embedding_blob, embedding_dim)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(content_hash, encoder_signature) DO UPDATE SET
              embedding_blob = excluded.embedding_blob,
              embedding_dim = excluded.embedding_dim
            """,
            (content_hash, encoder_signature, embedding_array.tobytes(), int(embedding_array.shape[0])),
        )

    def upsert_path(
        self,
        absolute_path: str,
        folder_path: str,
        content_hash: str,
        byte_size: int,
        mtime_ns: int,
        is_present: bool,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO image_paths (absolute_path, folder_path, content_hash, byte_size, mtime_ns, is_present)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(absolute_path) DO UPDATE SET
              folder_path = excluded.folder_path,
              content_hash = excluded.content_hash,
              byte_size = excluded.byte_size,
              mtime_ns = excluded.mtime_ns,
              is_present = excluded.is_present,
              last_scanned_at = CURRENT_TIMESTAMP
            """,
            (absolute_path, folder_path, content_hash, byte_size, mtime_ns, int(is_present)),
        )

    def get_folder_rows(self, folder_path: str, encoder_signature: str):
        return self.connection.execute(
            """
            SELECT image_paths.absolute_path, image_paths.content_hash, image_embeddings.embedding_blob, image_embeddings.embedding_dim
            FROM image_paths
            JOIN image_embeddings
              ON image_paths.content_hash = image_embeddings.content_hash
            WHERE image_paths.folder_path = ?
              AND image_paths.is_present = 1
              AND image_embeddings.encoder_signature = ?
            ORDER BY image_paths.absolute_path
            """,
            (folder_path, encoder_signature),
        ).fetchall()

    def get_known_paths(self, folder_path: str):
        return self.connection.execute(
            """
            SELECT absolute_path, folder_path, content_hash, byte_size, mtime_ns, is_present
            FROM image_paths
            WHERE folder_path = ?
            ORDER BY absolute_path
            """,
            (folder_path,),
        ).fetchall()

    def get_embedding_row(self, content_hash: str, encoder_signature: str):
        return self.connection.execute(
            """
            SELECT content_hash, encoder_signature, embedding_blob, embedding_dim
            FROM image_embeddings
            WHERE content_hash = ? AND encoder_signature = ?
            """,
            (content_hash, encoder_signature),
        ).fetchone()

    def mark_path_missing(self, absolute_path: str) -> None:
        self.connection.execute(
            """
            UPDATE image_paths
            SET is_present = 0,
                last_scanned_at = CURRENT_TIMESTAMP
            WHERE absolute_path = ?
            """,
            (absolute_path,),
        )

    def upsert_folder_state(
        self,
        folder_path: str,
        active_encoder_signature: str,
        file_count: int,
        total_bytes: int,
        scan_signature: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO folder_states (
              folder_path,
              active_encoder_signature,
              file_count,
              total_bytes,
              scan_signature,
              last_scanned_at,
              last_synced_at
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(folder_path) DO UPDATE SET
              active_encoder_signature = excluded.active_encoder_signature,
              file_count = excluded.file_count,
              total_bytes = excluded.total_bytes,
              scan_signature = excluded.scan_signature,
              last_scanned_at = CURRENT_TIMESTAMP,
              last_synced_at = CURRENT_TIMESTAMP
            """,
            (folder_path, active_encoder_signature, file_count, total_bytes, scan_signature),
        )

    @staticmethod
    def decode_embedding_blob(embedding_blob: bytes, embedding_dim: int) -> np.ndarray:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32, count=embedding_dim)
        return embedding.copy()

    def count_assets(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM image_assets").fetchone()[0])

    def count_embeddings(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM image_embeddings").fetchone()[0])

    def count_paths(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM image_paths").fetchone()[0])

    def count_folder_states(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM folder_states").fetchone()[0])
