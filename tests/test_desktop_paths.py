from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from desktop_runtime.paths import AppPaths, build_app_paths, resolve_uv_binary
from desktop_runtime.progress import ProgressEvent


class DesktopPathsTests(unittest.TestCase):
    def test_build_app_paths_uses_application_support_layout(self):
        home = Path("/Users/tester")
        paths = build_app_paths(home_dir=home, app_name="SemanticGallery")

        self.assertEqual(paths.support_dir, home / "Library" / "Application Support" / "SemanticGallery")
        self.assertEqual(paths.runtime_dir, paths.support_dir / "runtime")
        self.assertEqual(paths.index_db_path, paths.support_dir / "index.sqlite3")
        self.assertEqual(paths.logs_dir, paths.support_dir / "logs")
        self.assertEqual(paths.stage2_dir, paths.support_dir / "stage2")

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
