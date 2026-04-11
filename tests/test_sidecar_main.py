from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from desktop_runtime.sidecar_main import build_service, default_index_db_path


class SidecarMainTests(unittest.TestCase):
    def test_default_index_db_path_uses_support_root_when_workspace_is_runtime(self):
        runtime_root = Path("/tmp/SemanticGallery/runtime")
        self.assertEqual(default_index_db_path(runtime_root), Path("/tmp/SemanticGallery/index.sqlite3").resolve())

    @mock.patch("desktop_runtime.sidecar_main.MLXEmbeddingEncoder")
    def test_build_service_wires_stage2_runner_and_stage1_loader(self, encoder_cls):
        with tempfile.TemporaryDirectory(prefix="sg-sidecar-") as tmp_dir:
            workspace_root = Path(tmp_dir) / "runtime"
            workspace_root.mkdir()

            service = build_service(workspace_root)

        self.assertEqual(service.setup_status, "ready")
        self.assertIsNotNone(service.stage2_job)
        self.assertEqual(service.last_task_message, "Choose a folder to build the local index.")
        encoder_loader = service.encoder_loader
        self.assertIsNotNone(encoder_loader)
        encoder_loader(Path("/tmp/gallery"), "stage1")
        encoder_cls.assert_called_once_with(workspace_root.resolve(), Path("/tmp/gallery"), "stage1")


if __name__ == "__main__":
    unittest.main()
