from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

fake_mlx = types.ModuleType("mlx")
fake_mlx_core = types.ModuleType("mlx.core")
fake_mlx.core = fake_mlx_core
sys.modules.setdefault("mlx", fake_mlx)
sys.modules.setdefault("mlx.core", fake_mlx_core)

from deployment.search_engine import BaseSearchEngine


class DummySearchEngine(BaseSearchEngine):
    def __init__(self, gallery_path: Path, config: dict, image_paths: list[str], embeddings: np.ndarray, metadata_texts: list[str] | None):
        self.gallery_path = gallery_path
        self.config = config
        self.image_paths = image_paths
        self.embeddings = embeddings
        self.metadata_texts = metadata_texts

    def search(self, query_text: str, k: int = 20):
        raise NotImplementedError

    def search_by_image(self, image, k: int = 20):
        raise NotImplementedError

    def search_similar(self, image_path: str | Path, k: int = 20):
        raise NotImplementedError


class SearchEngineTransactionTests(unittest.TestCase):
    def test_delete_images_restores_index_files_after_failure(self):
        with tempfile.TemporaryDirectory(prefix="sg-search-engine-") as tmp_dir:
            root = Path(tmp_dir)
            gallery_dir = root / "gallery"
            gallery_dir.mkdir()
            image_paths = []
            for index in range(3):
                image_path = gallery_dir / f"image_{index}.jpg"
                image_path.write_bytes(f"image-{index}".encode("utf-8"))
                image_paths.append(image_path.as_posix())

            embeddings_path = root / "embeddings.npy"
            indexed_paths_path = root / "paths.txt"
            metadata_manifest_path = root / "manifest.jsonl"
            skipped_images_path = root / "skipped.json"
            file_state_path = root / "file_state.json"

            np.save(embeddings_path, np.arange(12, dtype=np.float32).reshape(3, 4))
            indexed_paths_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
            metadata_manifest_path.write_text(
                "".join(json.dumps({"image_path": path, "captions": [Path(path).stem]}) + "\n" for path in image_paths),
                encoding="utf-8",
            )
            skipped_images_path.write_text("[]", encoding="utf-8")
            file_state_path.write_text(
                json.dumps(
                    [{"relative_path": Path(path).name, "size": Path(path).stat().st_size, "mtime_ns": Path(path).stat().st_mtime_ns} for path in image_paths],
                    indent=2,
                ),
                encoding="utf-8",
            )

            engine = DummySearchEngine(
                gallery_path=gallery_dir,
                config={
                    "indexed_paths_file": indexed_paths_path.as_posix(),
                    "embeddings_file": embeddings_path.as_posix(),
                    "metadata_manifest": metadata_manifest_path.as_posix(),
                    "skipped_images_file": skipped_images_path.as_posix(),
                    "file_state_file": file_state_path.as_posix(),
                },
                image_paths=list(image_paths),
                embeddings=np.load(embeddings_path),
                metadata_texts=["a", "b", "c"],
            )

            original_paths_text = indexed_paths_path.read_text(encoding="utf-8")
            original_manifest_text = metadata_manifest_path.read_text(encoding="utf-8")
            original_state_text = file_state_path.read_text(encoding="utf-8")
            original_embeddings = np.load(embeddings_path).copy()

            def fail_after_partial_write(_image_paths):
                raise RuntimeError("forced failure after partial delete")

            engine._remove_from_metadata_manifest_many = fail_after_partial_write  # type: ignore[method-assign]

            with self.assertRaises(RuntimeError):
                engine.delete_images([image_paths[0]])

            self.assertEqual(engine.image_paths, image_paths)
            self.assertTrue(np.array_equal(np.load(embeddings_path), original_embeddings))
            self.assertEqual(indexed_paths_path.read_text(encoding="utf-8"), original_paths_text)
            self.assertEqual(metadata_manifest_path.read_text(encoding="utf-8"), original_manifest_text)
            self.assertEqual(file_state_path.read_text(encoding="utf-8"), original_state_text)


if __name__ == "__main__":
    unittest.main()
