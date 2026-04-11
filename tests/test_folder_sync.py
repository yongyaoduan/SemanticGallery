from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from desktop_runtime.folder_sync import reconcile_folder
from desktop_runtime.index_store import IndexStore
from desktop_runtime.search_view import ActiveSearchView


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode_image(self, path: Path) -> np.ndarray:
        self.calls.append(path.name)
        seed = float(len(path.name))
        return np.asarray([seed, seed + 1.0, seed + 2.0], dtype=np.float32)


class FolderSyncTests(unittest.TestCase):
    def test_reconcile_reuses_existing_embedding_for_duplicate_content(self):
        with tempfile.TemporaryDirectory(prefix="sg-folder-sync-") as tmp_dir:
            root = Path(tmp_dir)
            folder_a = root / "folder-a"
            folder_b = root / "folder-b"
            folder_a.mkdir()
            folder_b.mkdir()
            image_a = folder_a / "a.jpg"
            image_b = folder_b / "b.jpg"
            image_a.write_bytes(b"same")
            image_b.write_bytes(b"same")

            store = IndexStore.connect(root / "index.sqlite3")
            store.migrate()
            encoder = FakeEncoder()

            reconcile_folder(store, folder_a, "stage1", encoder)
            reconcile_folder(store, folder_b, "stage1", encoder)

            self.assertEqual(encoder.calls, ["a.jpg"])
            self.assertEqual(store.count_embeddings(), 1)
            self.assertEqual(store.count_paths(), 2)
            self.assertEqual(store.count_folder_states(), 2)

    def test_reconcile_updates_changed_files_and_marks_missing_paths(self):
        with tempfile.TemporaryDirectory(prefix="sg-folder-sync-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "folder"
            folder.mkdir()
            keep = folder / "keep.jpg"
            change = folder / "change.jpg"
            remove = folder / "remove.jpg"
            keep.write_bytes(b"keep-content")
            change.write_bytes(b"change-content")
            remove.write_bytes(b"remove-content")

            store = IndexStore.connect(root / "index.sqlite3")
            store.migrate()
            encoder = FakeEncoder()

            reconcile_folder(store, folder, "stage1", encoder)

            change.write_bytes(b"changed-content")
            remove.unlink()

            reconcile_folder(store, folder, "stage1", encoder)

            self.assertEqual(encoder.calls.count("keep.jpg"), 1)
            self.assertEqual(encoder.calls.count("change.jpg"), 2)
            self.assertEqual(encoder.calls.count("remove.jpg"), 1)
            self.assertEqual(store.count_embeddings(), 4)
            self.assertEqual(store.count_paths(), 3)
            self.assertEqual(store.count_folder_states(), 1)

            rows = {
                row["absolute_path"]: row
                for row in store.get_known_paths(folder.resolve().as_posix())
            }
            self.assertEqual(rows[keep.resolve().as_posix()]["is_present"], 1)
            self.assertEqual(rows[change.resolve().as_posix()]["is_present"], 1)
            self.assertEqual(rows[remove.resolve().as_posix()]["is_present"], 0)

    def test_reconcile_backfills_missing_embedding_for_unchanged_file(self):
        with tempfile.TemporaryDirectory(prefix="sg-folder-sync-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "folder"
            folder.mkdir()
            image = folder / "photo.jpg"
            image.write_bytes(b"stable-content")

            store = IndexStore.connect(root / "index.sqlite3")
            store.migrate()
            encoder = FakeEncoder()

            content_hash = "hash-photo"
            stat = image.stat()
            store.upsert_asset(content_hash=content_hash, byte_size=stat.st_size)
            store.upsert_path(
                absolute_path=image.resolve().as_posix(),
                folder_path=folder.resolve().as_posix(),
                content_hash=content_hash,
                byte_size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                is_present=True,
            )

            reconcile_folder(store, folder, "stage1", encoder)

            self.assertEqual(encoder.calls, ["photo.jpg"])
            self.assertEqual(store.count_embeddings(), 1)
            view = ActiveSearchView.from_store(store, folder.resolve().as_posix(), "stage1")
            self.assertEqual(view.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), limit=1), [image.resolve().as_posix()])


if __name__ == "__main__":
    unittest.main()
