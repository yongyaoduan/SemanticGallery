from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from desktop_runtime.index_store import IndexStore
from desktop_runtime.search_view import ActiveSearchView


class SearchViewTests(unittest.TestCase):
    def test_active_search_view_returns_only_current_folder_paths(self):
        with tempfile.TemporaryDirectory(prefix="sg-search-view-") as tmp_dir:
            root = Path(tmp_dir)
            store = IndexStore.connect(root / "index.sqlite3")
            store.migrate()
            store.upsert_asset("hash-a", 4)
            store.upsert_asset("hash-b", 4)
            store.upsert_embedding("hash-a", "stage1", np.asarray([1.0, 0.0], dtype=np.float32))
            store.upsert_embedding("hash-b", "stage1", np.asarray([0.0, 1.0], dtype=np.float32))
            store.upsert_path("/tmp/folder-a/one.jpg", "/tmp/folder-a", "hash-a", 4, 1, True)
            store.upsert_path("/tmp/folder-b/two.jpg", "/tmp/folder-b", "hash-b", 4, 1, True)

            view = ActiveSearchView.from_store(store, "/tmp/folder-a", "stage1")
            matches = view.search(np.asarray([1.0, 0.0], dtype=np.float32), limit=5)

            self.assertEqual(matches, ["/tmp/folder-a/one.jpg"])

    def test_active_search_view_uses_exact_cosine_ranking(self):
        with tempfile.TemporaryDirectory(prefix="sg-search-view-") as tmp_dir:
            root = Path(tmp_dir)
            store = IndexStore.connect(root / "index.sqlite3")
            store.migrate()
            store.upsert_asset("hash-a", 8)
            store.upsert_asset("hash-b", 8)
            store.upsert_embedding("hash-a", "stage1", np.asarray([10.0, 0.0], dtype=np.float32))
            store.upsert_embedding("hash-b", "stage1", np.asarray([1.0, 1.0], dtype=np.float32))
            store.upsert_path("/tmp/folder-a/strong-x.jpg", "/tmp/folder-a", "hash-a", 8, 1, True)
            store.upsert_path("/tmp/folder-a/diagonal.jpg", "/tmp/folder-a", "hash-b", 8, 1, True)

            view = ActiveSearchView.from_store(store, "/tmp/folder-a", "stage1")
            matches = view.search(np.asarray([1.0, 1.0], dtype=np.float32), limit=2)

            self.assertEqual(matches, ["/tmp/folder-a/diagonal.jpg", "/tmp/folder-a/strong-x.jpg"])

    def test_active_search_view_can_drop_deleted_paths_without_rebuilding_from_store(self):
        view = ActiveSearchView(
            paths=["/tmp/folder-a/one.jpg", "/tmp/folder-a/two.jpg", "/tmp/folder-a/three.jpg"],
            matrix=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        reduced = view.without_paths(["/tmp/folder-a/two.jpg"])

        self.assertEqual(reduced.paths, ["/tmp/folder-a/one.jpg", "/tmp/folder-a/three.jpg"])
        self.assertEqual(reduced.matrix.shape, (2, 2))
        self.assertEqual(reduced.search(np.asarray([0.0, 1.0], dtype=np.float32), limit=2), ["/tmp/folder-a/three.jpg", "/tmp/folder-a/one.jpg"])


if __name__ == "__main__":
    unittest.main()
