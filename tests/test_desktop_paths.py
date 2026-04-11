from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from desktop_runtime.paths import AppPaths, BUNDLED_RESOURCES_DIR_ENV_VAR, build_app_paths, resolve_uv_binary
from desktop_runtime.progress import ProgressEvent


class DesktopPathsTests(unittest.TestCase):
    def test_build_app_paths_uses_application_support_layout(self):
        home = Path("/Users/tester")
        expected_bundled = Path(__file__).resolve().parent.parent / "desktop_runtime" / "resources"
        with patch.dict(os.environ, {}, clear=True):
            paths = build_app_paths(home_dir=home, app_name="SemanticGallery")

        self.assertEqual(paths.app_name, "SemanticGallery")
        self.assertEqual(paths.support_dir, home / "Library" / "Application Support" / "SemanticGallery")
        self.assertEqual(paths.runtime_dir, paths.support_dir / "runtime")
        self.assertEqual(paths.cache_dir, paths.support_dir / "cache")
        self.assertEqual(paths.index_db_path, paths.support_dir / "index.sqlite3")
        self.assertEqual(paths.logs_dir, paths.support_dir / "logs")
        self.assertEqual(paths.stage2_dir, paths.support_dir / "stage2")
        self.assertEqual(paths.thumbnails_dir, paths.support_dir / "thumbs")
        self.assertEqual(paths.config_dir, paths.support_dir / "config")
        self.assertEqual(paths.bundled_resources_dir, expected_bundled)

    def test_build_app_paths_honors_bundled_resources_env_override(self):
        home = Path("/Users/tester")
        override = Path("/tmp/semanticgallery-resources")

        with patch.dict(os.environ, {BUNDLED_RESOURCES_DIR_ENV_VAR: override.as_posix()}, clear=False):
            paths = build_app_paths(home_dir=home)

        self.assertEqual(paths.bundled_resources_dir, override)

    def test_resolve_uv_binary_prefers_env_override_then_bundled_resource(self):
        with tempfile.TemporaryDirectory(prefix="sg-paths-") as tmp_dir:
            root = Path(tmp_dir)
            override = root / "custom-uv"
            bundled = root / "resources" / "uv"
            override.write_text("override", encoding="utf-8")
            bundled.parent.mkdir(parents=True)
            bundled.write_text("bundled", encoding="utf-8")

            paths = AppPaths(
                app_name="SemanticGallery",
                support_dir=root / "support",
                runtime_dir=root / "support" / "runtime",
                cache_dir=root / "support" / "cache",
                logs_dir=root / "support" / "logs",
                stage2_dir=root / "support" / "stage2",
                thumbnails_dir=root / "support" / "thumbs",
                index_db_path=root / "support" / "index.sqlite3",
                config_dir=root / "support" / "config",
                bundled_resources_dir=root / "resources",
            )

            resolved = resolve_uv_binary(paths, override.as_posix())
            self.assertEqual(resolved, override)

            resolved = resolve_uv_binary(paths, "")
            self.assertEqual(resolved, bundled)

    def test_progress_event_serializes_to_payload(self):
        event = ProgressEvent(task="setup", phase="download", message="Fetching model", current=3, total=5)

        self.assertEqual(
            event.to_payload(),
            {
                "task": "setup",
                "phase": "download",
                "message": "Fetching model",
                "current": 3,
                "total": 5,
            },
        )


if __name__ == "__main__":
    unittest.main()
