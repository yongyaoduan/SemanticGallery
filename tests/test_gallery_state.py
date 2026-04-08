from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deployment.gallery_state import build_gallery_bank_state_payload, iter_gallery_paths


class GalleryStateTests(unittest.TestCase):
    def test_iter_gallery_paths_ignores_hidden_segments(self):
        with tempfile.TemporaryDirectory(prefix="sg-gallery-state-") as tmp_dir:
            root = Path(tmp_dir)
            gallery_dir = root / "gallery"
            gallery_dir.mkdir()

            visible = gallery_dir / "visible.jpg"
            hidden_dir = gallery_dir / ".hidden"
            hidden_dir.mkdir()
            hidden = hidden_dir / "hidden.jpg"
            dotted_child = gallery_dir / "nested" / ".cache"
            dotted_child.mkdir(parents=True)
            dotted = dotted_child / "ignored.png"

            visible.write_bytes(b"visible")
            hidden.write_bytes(b"hidden")
            dotted.write_bytes(b"dotted")

            paths = iter_gallery_paths(gallery_dir)
            self.assertEqual(paths, [visible.resolve()])

    def test_hidden_files_do_not_change_bank_state(self):
        with tempfile.TemporaryDirectory(prefix="sg-gallery-state-") as tmp_dir:
            root = Path(tmp_dir)
            gallery_dir = root / "gallery"
            gallery_dir.mkdir()
            model_dir = root / "model"
            model_dir.mkdir()

            visible = gallery_dir / "visible.jpg"
            visible.write_bytes(b"visible")

            before = build_gallery_bank_state_payload(
                gallery_dir,
                model_dir=model_dir,
                precision="bfloat16",
                weights_file_path=None,
            )

            hidden_dir = gallery_dir / ".hidden"
            hidden_dir.mkdir()
            (hidden_dir / "ignored.jpg").write_bytes(b"ignored")

            after = build_gallery_bank_state_payload(
                gallery_dir,
                model_dir=model_dir,
                precision="bfloat16",
                weights_file_path=None,
            )

            self.assertEqual(before["gallery_file_count"], 1)
            self.assertEqual(after["gallery_file_count"], 1)
            self.assertEqual(before["gallery_state_sha256"], after["gallery_state_sha256"])


if __name__ == "__main__":
    unittest.main()
