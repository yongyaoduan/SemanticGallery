from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from desktop_runtime.mlx_encoder import build_stage1_weights_path, resolve_encoder_weights_file


class MLXEncoderTests(unittest.TestCase):
    def test_resolve_stage1_weights_file_prefers_published_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="sg-mlx-encoder-") as tmp_dir:
            root = Path(tmp_dir)
            stage1_weights = build_stage1_weights_path(root)
            stage1_weights.parent.mkdir(parents=True, exist_ok=True)
            stage1_weights.write_bytes(b"stage1")

            resolved = resolve_encoder_weights_file(root, None, "stage1")

            self.assertEqual(resolved, stage1_weights)

    def test_resolve_stage2_weights_file_validates_signature(self):
        with tempfile.TemporaryDirectory(prefix="sg-mlx-encoder-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            weights_path = root / "logs" / "semanticgallery_private_data_adapted" / "gallery-a287a17b14f2" / "weights.safetensors"
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_bytes(b"stage2")
            signature = hashlib.sha256(b"stage2").hexdigest()

            with unittest.mock.patch("desktop_runtime.mlx_encoder.gallery_artifact_key", return_value="gallery-a287a17b14f2"):
                resolved = resolve_encoder_weights_file(root, folder, signature)

            self.assertEqual(resolved, weights_path.resolve())

    def test_resolve_stage2_weights_file_rejects_mismatched_signature(self):
        with tempfile.TemporaryDirectory(prefix="sg-mlx-encoder-") as tmp_dir:
            root = Path(tmp_dir)
            folder = root / "gallery"
            folder.mkdir()
            weights_path = root / "logs" / "semanticgallery_private_data_adapted" / "gallery-a287a17b14f2" / "weights.safetensors"
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_bytes(b"stage2")

            with unittest.mock.patch("desktop_runtime.mlx_encoder.gallery_artifact_key", return_value="gallery-a287a17b14f2"):
                with self.assertRaises(ValueError):
                    resolve_encoder_weights_file(root, folder, "not-the-real-signature")


if __name__ == "__main__":
    unittest.main()
