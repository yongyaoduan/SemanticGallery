from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from desktop_runtime.index_store import IndexStore


class IndexStoreTests(unittest.TestCase):
    def test_duplicate_content_reuses_embedding_but_keeps_two_paths(self):
        with tempfile.TemporaryDirectory(prefix="sg-index-store-") as tmp_dir:
            db_path = Path(tmp_dir) / "index.sqlite3"
            store = IndexStore.connect(db_path)
            store.migrate()

            with store.transaction():
                embedding = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
                store.upsert_asset(content_hash="abc", byte_size=3)
                store.upsert_embedding(content_hash="abc", encoder_signature="stage1", embedding=embedding)
                store.upsert_path(
                    absolute_path="/tmp/a.jpg",
                    folder_path="/tmp/folder-a",
                    content_hash="abc",
                    byte_size=3,
                    mtime_ns=10,
                    is_present=True,
                )
                store.upsert_path(
                    absolute_path="/tmp/b.jpg",
                    folder_path="/tmp/folder-b",
                    content_hash="abc",
                    byte_size=3,
                    mtime_ns=11,
                    is_present=True,
                )

            self.assertEqual(store.count_assets(), 1)
            self.assertEqual(store.count_embeddings(), 1)
            self.assertEqual(store.count_paths(), 2)

    def test_get_folder_rows_returns_present_paths_for_encoder_signature(self):
        with tempfile.TemporaryDirectory(prefix="sg-index-store-") as tmp_dir:
            db_path = Path(tmp_dir) / "index.sqlite3"
            store = IndexStore.connect(db_path)
            store.migrate()

            with store.transaction():
                store.upsert_asset(content_hash="abc", byte_size=3)
                store.upsert_asset(content_hash="def", byte_size=4)
                store.upsert_embedding(
                    content_hash="abc",
                    encoder_signature="stage1",
                    embedding=np.asarray([0.1, 0.2], dtype=np.float32),
                )
                store.upsert_embedding(
                    content_hash="def",
                    encoder_signature="stage1",
                    embedding=np.asarray([0.3, 0.4], dtype=np.float32),
                )
                store.upsert_embedding(
                    content_hash="abc",
                    encoder_signature="stage2",
                    embedding=np.asarray([0.5, 0.6], dtype=np.float32),
                )
                store.upsert_path(
                    absolute_path="/tmp/b.jpg",
                    folder_path="/tmp/folder",
                    content_hash="abc",
                    byte_size=3,
                    mtime_ns=10,
                    is_present=True,
                )
                store.upsert_path(
                    absolute_path="/tmp/a.jpg",
                    folder_path="/tmp/folder",
                    content_hash="def",
                    byte_size=4,
                    mtime_ns=11,
                    is_present=False,
                )

            rows = store.get_folder_rows(folder_path="/tmp/folder", encoder_signature="stage1")

            self.assertEqual([row["absolute_path"] for row in rows], ["/tmp/b.jpg"])
            self.assertEqual([row["content_hash"] for row in rows], ["abc"])
            self.assertEqual(rows[0]["embedding_dim"], 2)
            self.assertEqual(
                IndexStore.decode_embedding_blob(rows[0]["embedding_blob"], rows[0]["embedding_dim"]).tolist(),
                [0.10000000149011612, 0.20000000298023224],
            )

    def test_upserts_update_existing_asset_path_and_embedding_rows(self):
        with tempfile.TemporaryDirectory(prefix="sg-index-store-") as tmp_dir:
            db_path = Path(tmp_dir) / "index.sqlite3"
            store = IndexStore.connect(db_path)
            store.migrate()

            with store.transaction():
                store.upsert_asset(content_hash="abc", byte_size=3)
                store.upsert_path(
                    absolute_path="/tmp/a.jpg",
                    folder_path="/tmp/folder-a",
                    content_hash="abc",
                    byte_size=3,
                    mtime_ns=10,
                    is_present=True,
                )
                store.upsert_embedding(
                    content_hash="abc",
                    encoder_signature="stage1",
                    embedding=np.asarray([0.1, 0.2], dtype=np.float32),
                )

            with store.transaction():
                store.upsert_asset(content_hash="abc", byte_size=7)
                store.upsert_path(
                    absolute_path="/tmp/a.jpg",
                    folder_path="/tmp/folder-b",
                    content_hash="abc",
                    byte_size=7,
                    mtime_ns=99,
                    is_present=False,
                )
                store.upsert_embedding(
                    content_hash="abc",
                    encoder_signature="stage1",
                    embedding=np.asarray([0.9, 1.1], dtype=np.float32),
                )

            asset_row = store.connection.execute(
                "SELECT content_hash, byte_size, updated_at FROM image_assets WHERE content_hash = ?",
                ("abc",),
            ).fetchone()
            path_row = store.connection.execute(
                """
                SELECT absolute_path, folder_path, content_hash, byte_size, mtime_ns, is_present
                FROM image_paths
                WHERE absolute_path = ?
                """,
                ("/tmp/a.jpg",),
            ).fetchone()
            embedding_row = store.connection.execute(
                """
                SELECT content_hash, encoder_signature, embedding_blob, embedding_dim
                FROM image_embeddings
                WHERE content_hash = ? AND encoder_signature = ?
                """,
                ("abc", "stage1"),
            ).fetchone()

            self.assertEqual(asset_row["byte_size"], 7)
            self.assertEqual(path_row["folder_path"], "/tmp/folder-b")
            self.assertEqual(path_row["byte_size"], 7)
            self.assertEqual(path_row["mtime_ns"], 99)
            self.assertEqual(path_row["is_present"], 0)
            self.assertEqual(embedding_row["embedding_dim"], 2)
            self.assertEqual(
                IndexStore.decode_embedding_blob(embedding_row["embedding_blob"], embedding_row["embedding_dim"]).tolist(),
                [0.8999999761581421, 1.100000023841858],
            )

    def test_upsert_embedding_rejects_invalid_shapes(self):
        with tempfile.TemporaryDirectory(prefix="sg-index-store-") as tmp_dir:
            db_path = Path(tmp_dir) / "index.sqlite3"
            store = IndexStore.connect(db_path)
            store.migrate()

            with self.assertRaises(ValueError):
                store.upsert_embedding(
                    content_hash="abc",
                    encoder_signature="stage1",
                    embedding=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
