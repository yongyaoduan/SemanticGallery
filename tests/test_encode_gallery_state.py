from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from deployment.encode_gallery import load_previous_embeddings


class EncodeGalleryStateTests(unittest.TestCase):
    def test_load_previous_embeddings_bootstraps_missing_file_state(self):
        with tempfile.TemporaryDirectory(prefix="sg-encode-gallery-") as tmp_dir:
            root = Path(tmp_dir)
            gallery_dir = root / "gallery"
            gallery_dir.mkdir()

            image_paths = []
            for index in range(2):
                image_path = gallery_dir / f"image_{index}.jpg"
                image_path.write_bytes(f"image-{index}".encode("utf-8"))
                image_paths.append(image_path)

            embeddings_path = root / "embeddings.npy"
            paths_path = root / "paths.txt"
            file_state_path = root / "file_state.json"

            np.save(embeddings_path, np.arange(8, dtype=np.float32).reshape(2, 4))
            paths_path.write_text("\n".join(path.as_posix() for path in image_paths) + "\n", encoding="utf-8")

            resolved_gallery_dir = gallery_dir.resolve()
            embedding_map, state_map, previous_paths, bootstrapped_state = load_previous_embeddings(
                gallery_path=resolved_gallery_dir,
                embeddings_path=embeddings_path,
                paths_path=paths_path,
                file_state_path=file_state_path,
            )

            resolved_paths = {path.resolve().as_posix() for path in image_paths}
            self.assertTrue(bootstrapped_state)
            self.assertEqual({Path(path).resolve().as_posix() for path in previous_paths}, resolved_paths)
            self.assertEqual(set(embedding_map), resolved_paths)
            self.assertEqual(set(state_map), resolved_paths)
            for path in image_paths:
                state_row = state_map[path.resolve().as_posix()]
                self.assertEqual(state_row["relative_path"], path.name)
                self.assertEqual(state_row["size"], path.stat().st_size)


if __name__ == "__main__":
    unittest.main()
