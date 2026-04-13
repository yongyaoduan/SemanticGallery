from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class BuildDesktopReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module(Path("scripts/build_desktop_release.py"), "build_desktop_release")

    def test_find_signing_identities_extracts_identity_names(self):
        output = """
  1) 1234567890ABCDEF1234567890ABCDEF12345678 "Developer ID Application: Example Corp (ABCDE12345)"
  2) ABCDEF1234567890ABCDEF1234567890ABCDEF12 "Apple Development: dev@example.com (ABCDE12345)"
"""
        self.assertEqual(
            self.module.find_signing_identities(output),
            [
                "Developer ID Application: Example Corp (ABCDE12345)",
                "Apple Development: dev@example.com (ABCDE12345)",
            ],
        )

    def test_select_signing_identity_prefers_developer_id(self):
        identity = self.module.select_signing_identity(
            [
                "Apple Development: dev@example.com (ABCDE12345)",
                "Developer ID Application: Example Corp (ABCDE12345)",
            ]
        )
        self.assertEqual(identity, "Developer ID Application: Example Corp (ABCDE12345)")

    def test_build_notary_credentials_accepts_apple_id_flow(self):
        credentials = self.module.build_notary_credentials(
            {
                "APPLE_ID": "dev@example.com",
                "APPLE_PASSWORD": "app-password",
                "APPLE_TEAM_ID": "ABCDE12345",
            }
        )
        self.assertIsNotNone(credentials)
        self.assertEqual(
            credentials.args,
            ["--apple-id", "dev@example.com", "--password", "app-password", "--team-id", "ABCDE12345"],
        )

    def test_discover_single_prefers_expected_app_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(temp_dir)
            macos_dir = bundle_dir / "macos"
            macos_dir.mkdir()
            fallback = macos_dir / "SemanticGallery 2.app"
            preferred = macos_dir / "SemanticGallery.app"
            fallback.mkdir()
            preferred.mkdir()
            with patch.object(self.module, "BUILD_BUNDLE_DIR", bundle_dir):
                selected = self.module.discover_single("macos/*.app", preferred_name="SemanticGallery.app")
            self.assertEqual(selected, preferred)

    def test_render_uninstaller_applescript_mentions_runtime_cleanup_targets(self):
        script = self.module.render_uninstaller_applescript()

        self.assertIn("Library/Application Support/com.semanticgallery.desktop", script)
        self.assertIn("Library/Caches/com.semanticgallery.desktop", script)
        self.assertIn("Library/WebKit/com.semanticgallery.desktop", script)
        self.assertIn("Library/HTTPStorages/com.semanticgallery.desktop", script)
        self.assertIn("pkill -f 'desktop_runtime.sidecar_main'", script)
        self.assertIn("pkill -f 'semanticgallery_desktop'", script)
        self.assertIn("SemanticGallery.app", script)
        self.assertIn("Uninstall", script)

    def test_prepare_release_stage_copies_app_and_uninstaller(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_path = root / "SemanticGallery.app"
            uninstall_path = root / "Uninstall SemanticGallery.app"
            stage_root = root / "stage"

            (app_path / "Contents").mkdir(parents=True)
            (app_path / "Contents" / "Info.plist").write_text("app")
            (uninstall_path / "Contents").mkdir(parents=True)
            (uninstall_path / "Contents" / "Info.plist").write_text("uninstall")

            volume_root = self.module.prepare_release_stage(
                app_path,
                stage_root,
                uninstall_helper_path=uninstall_path,
            )

            self.assertTrue((volume_root / "SemanticGallery.app" / "Contents" / "Info.plist").is_file())
            self.assertTrue((volume_root / "Uninstall SemanticGallery.app" / "Contents" / "Info.plist").is_file())
            self.assertTrue((volume_root / "Applications").is_symlink())

    def test_ensure_icons_reuses_existing_icon_set_when_regeneration_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tauri_dir = Path(temp_dir) / "src-tauri"
            icons_dir = tauri_dir / "icons"
            icons_dir.mkdir(parents=True)
            for name in self.module.ICON_FILENAMES:
                (icons_dir / name).write_text("icon")

            with patch.object(self.module, "TAURI_DIR", tauri_dir):
                with patch.object(
                    self.module,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, ["python", "generate_icons.py"]),
                ):
                    self.module.ensure_icons()


if __name__ == "__main__":
    unittest.main()
