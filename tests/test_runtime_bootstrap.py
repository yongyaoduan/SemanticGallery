from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from desktop_runtime.progress import ProgressEvent
from desktop_runtime.runtime_bootstrap import (
    SETUP_PREFIX,
    build_runtime_paths,
    build_sidecar_command,
    emit_progress_line,
)


class RuntimeBootstrapTests(unittest.TestCase):
    def test_build_runtime_paths_uses_support_dir_parent_of_runtime_root(self):
        with tempfile.TemporaryDirectory(prefix="sg-bootstrap-") as tmp_dir:
            support_dir = Path(tmp_dir) / "SemanticGallery"
            runtime_dir = support_dir / "runtime"
            runtime_dir.mkdir(parents=True)

            paths = build_runtime_paths(runtime_dir)

        self.assertEqual(paths.support_dir, support_dir.resolve())
        self.assertEqual(paths.runtime_dir, runtime_dir.resolve())
        self.assertEqual(paths.index_db_path, support_dir.resolve() / "index.sqlite3")

    def test_build_runtime_paths_uses_bundled_resources_override(self):
        with tempfile.TemporaryDirectory(prefix="sg-bootstrap-") as tmp_dir:
            support_dir = Path(tmp_dir) / "SemanticGallery"
            runtime_dir = support_dir / "runtime"
            runtime_dir.mkdir(parents=True)
            bundled_resources_dir = Path(tmp_dir) / "bundled-resources"
            bundled_resources_dir.mkdir()

            paths = build_runtime_paths(
                runtime_dir,
                bundled_resources_dir=bundled_resources_dir,
            )

        self.assertEqual(paths.bundled_resources_dir, bundled_resources_dir.resolve())

    def test_emit_progress_line_serializes_json_payload(self):
        event = ProgressEvent(
            task="prepare-base-model",
            phase="finish",
            message="Base model is ready",
            current=3,
            total=5,
        )

        with mock.patch("builtins.print") as print_mock:
            emit_progress_line(event)

        message = print_mock.call_args.args[0]
        self.assertTrue(message.startswith(SETUP_PREFIX))
        self.assertEqual(
            json.loads(message.removeprefix(SETUP_PREFIX)),
            {
                "task": "prepare-base-model",
                "phase": "finish",
                "message": "Base model is ready",
                "current": 3,
                "total": 5,
            },
        )

    def test_build_sidecar_command_uses_runtime_venv_and_index_db(self):
        with tempfile.TemporaryDirectory(prefix="sg-bootstrap-") as tmp_dir:
            support_dir = Path(tmp_dir) / "SemanticGallery"
            runtime_dir = support_dir / "runtime"
            runtime_dir.mkdir(parents=True)
            paths = build_runtime_paths(runtime_dir)
            custom_index_db = support_dir / "custom-index.sqlite3"

            command = build_sidecar_command(
                paths,
                host="127.0.0.1",
                port=38291,
                index_db_path=custom_index_db,
            )

        self.assertEqual(command[0], (runtime_dir.resolve() / ".venv" / "bin" / "python").as_posix())
        self.assertEqual(
            command[1:],
            [
                "-m",
                "desktop_runtime.sidecar_main",
                "--host",
                "127.0.0.1",
                "--port",
                "38291",
                "--workspace-root",
                runtime_dir.resolve().as_posix(),
                "--index-db",
                custom_index_db.resolve().as_posix(),
            ],
        )


if __name__ == "__main__":
    unittest.main()
