from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

from desktop_runtime.paths import AppPaths, build_app_paths
from desktop_runtime.runtime_setup import RuntimeDownloader, RuntimeSetup, SetupStatus


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def ensure_python_runtime(self, *args, **kwargs):
        self.calls.append("python")

    def ensure_dependencies(self, *args, **kwargs):
        self.calls.append("deps")

    def ensure_base_model(self, *args, **kwargs):
        self.calls.append("model")

    def ensure_public_anchor(self, *args, **kwargs):
        self.calls.append("anchor")


class RuntimeSetupTests(unittest.TestCase):
    def test_prepare_runtime_reports_expected_steps_in_order(self):
        events = []
        downloader = FakeDownloader()
        with tempfile.TemporaryDirectory(prefix="sg-setup-") as tmp_dir:
            paths = build_app_paths(Path(tmp_dir))
            setup = RuntimeSetup(paths=paths, downloader=downloader, emit=events.append)

            status = setup.prepare()

        self.assertEqual(status, SetupStatus.READY)
        self.assertEqual(downloader.calls, ["python", "deps", "model", "anchor"])
        self.assertEqual(
            [
                (event.task, event.phase, event.message, event.current, event.total)
                for event in events
            ],
            [
                ("check-runtime", "start", "Checking the local runtime", 0, 5),
                ("check-runtime", "finish", "Local runtime is ready", 1, 5),
                ("prepare-dependencies", "start", "Preparing Python dependencies", 1, 5),
                ("prepare-dependencies", "finish", "Python dependencies are ready", 2, 5),
                ("prepare-base-model", "start", "Preparing the base model", 2, 5),
                ("prepare-base-model", "finish", "Base model is ready", 3, 5),
                ("prepare-public-anchor", "start", "Preparing the public adaptation set", 3, 5),
                ("prepare-public-anchor", "finish", "Public adaptation set is ready", 4, 5),
                ("finish-setup", "start", "Finishing setup", 4, 5),
                ("finish-setup", "finish", "Setup is complete", 5, 5),
            ],
        )

    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_installs_python_and_venv_via_uv(self, run_mock):
        with tempfile.TemporaryDirectory(prefix="sg-runtime-") as tmp_dir:
            root = Path(tmp_dir)
            paths = AppPaths(
                app_name="SemanticGallery",
                support_dir=root / "support",
                runtime_dir=root / "runtime",
                cache_dir=root / "cache",
                logs_dir=root / "logs",
                stage2_dir=root / "stage2",
                thumbnails_dir=root / "thumbs",
                index_db_path=root / "index.sqlite3",
                config_dir=root / "config",
                bundled_resources_dir=root / "resources",
            )

            RuntimeDownloader().ensure_python_runtime(paths)

        uv_path = (paths.bundled_resources_dir / "uv").as_posix()
        self.assertEqual(
            run_mock.call_args_list,
            [
                mock.call([uv_path, "python", "install", "3.12"], check=True),
                mock.call(
                    [
                        uv_path,
                        "venv",
                        (paths.runtime_dir / ".venv").as_posix(),
                        "--python",
                        "3.12",
                    ],
                    check=True,
                ),
            ],
        )

    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_installs_dependencies_with_uv_pip(self, run_mock):
        with tempfile.TemporaryDirectory(prefix="sg-runtime-") as tmp_dir:
            root = Path(tmp_dir)
            paths = AppPaths(
                app_name="SemanticGallery",
                support_dir=root / "support",
                runtime_dir=root / "runtime",
                cache_dir=root / "cache",
                logs_dir=root / "logs",
                stage2_dir=root / "stage2",
                thumbnails_dir=root / "thumbs",
                index_db_path=root / "index.sqlite3",
                config_dir=root / "config",
                bundled_resources_dir=root / "resources",
            )

            RuntimeDownloader().ensure_dependencies(paths)

        uv_path = (paths.bundled_resources_dir / "uv").as_posix()
        requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
        self.assertEqual(
            run_mock.call_args_list,
            [
                mock.call(
                    [
                        uv_path,
                        "pip",
                        "install",
                        "--python",
                        (paths.runtime_dir / ".venv" / "bin" / "python").as_posix(),
                        "-r",
                        requirements_path.as_posix(),
                    ],
                    check=True,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
