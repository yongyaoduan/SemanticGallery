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
    @staticmethod
    def _expected_process_env(paths: AppPaths) -> dict[str, str]:
        return {
            "UV_CACHE_DIR": (paths.cache_dir / "uv").as_posix(),
            "UV_PYTHON_INSTALL_DIR": (paths.support_dir / "python").as_posix(),
            "XDG_CACHE_HOME": paths.cache_dir.as_posix(),
            "HF_HOME": (paths.cache_dir / "huggingface").as_posix(),
            "HUGGINGFACE_HUB_CACHE": (paths.cache_dir / "huggingface" / "hub").as_posix(),
            "TRANSFORMERS_CACHE": (paths.cache_dir / "transformers").as_posix(),
        }

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
                ("check-runtime", "start", "Checking the local runtime", 1, 6),
                ("check-runtime", "finish", "Local runtime is ready", 2, 6),
                ("prepare-dependencies", "start", "Preparing Python dependencies", 2, 6),
                ("prepare-dependencies", "finish", "Python dependencies are ready", 3, 6),
                ("prepare-base-model", "start", "Preparing the base model", 3, 6),
                ("prepare-base-model", "finish", "Base model is ready", 4, 6),
                ("prepare-public-anchor", "start", "Preparing the public adaptation set", 4, 6),
                ("prepare-public-anchor", "finish", "Public adaptation set is ready", 5, 6),
                ("finish-setup", "start", "Finishing setup", 5, 6),
                ("finish-setup", "finish", "Setup is complete", 6, 6),
            ],
        )

    @mock.patch("desktop_runtime.runtime_setup.resolve_uv_binary")
    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_installs_python_and_venv_via_uv(self, run_mock, resolve_uv_binary_mock):
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
            resolve_uv_binary_mock.return_value = paths.bundled_resources_dir / "uv"

            RuntimeDownloader().ensure_python_runtime(paths)

        uv_path = (paths.bundled_resources_dir / "uv").as_posix()
        expected_env = self._expected_process_env(paths)
        self.assertEqual(
            run_mock.call_args_list,
            [
                mock.call(
                    [
                        uv_path,
                        "python",
                        "install",
                        "3.12",
                        "--install-dir",
                        (paths.support_dir / "python").as_posix(),
                    ],
                    check=True,
                    env=mock.ANY,
                ),
                mock.call(
                    [
                        uv_path,
                        "venv",
                        (paths.runtime_dir / ".venv").as_posix(),
                        "--python",
                        "3.12",
                    ],
                    check=True,
                    env=mock.ANY,
                ),
            ],
        )
        for call in run_mock.call_args_list:
            for key, value in expected_env.items():
                self.assertEqual(call.kwargs["env"][key], value)

    @mock.patch("desktop_runtime.runtime_setup.resolve_uv_binary")
    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_skips_python_setup_when_runtime_venv_exists(self, run_mock, resolve_uv_binary_mock):
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
            (paths.runtime_dir / ".venv" / "bin").mkdir(parents=True)
            (paths.runtime_dir / ".venv" / "bin" / "python").write_text("")
            resolve_uv_binary_mock.return_value = paths.bundled_resources_dir / "uv"

            RuntimeDownloader().ensure_python_runtime(paths)

        run_mock.assert_not_called()

    @mock.patch("desktop_runtime.runtime_setup.resolve_uv_binary")
    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_installs_dependencies_with_uv_pip(self, run_mock, resolve_uv_binary_mock):
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
            (paths.runtime_dir / ".venv" / "bin").mkdir(parents=True)
            (paths.runtime_dir / ".venv" / "bin" / "python").write_text("")
            resolve_uv_binary_mock.return_value = paths.bundled_resources_dir / "uv"

            run_mock.side_effect = [
                mock.Mock(returncode=1),
                None,
            ]
            RuntimeDownloader().ensure_dependencies(paths)

        uv_path = (paths.bundled_resources_dir / "uv").as_posix()
        requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
        self.assertEqual(
            run_mock.call_args_list,
            [
                mock.call(
                    [
                        (paths.runtime_dir / ".venv" / "bin" / "python").as_posix(),
                        "-c",
                        mock.ANY,
                    ],
                    check=False,
                    env=mock.ANY,
                    capture_output=True,
                    text=True,
                ),
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
                    env=mock.ANY,
                ),
            ],
        )
        for key, value in self._expected_process_env(paths).items():
            self.assertEqual(run_mock.call_args_list[1].kwargs["env"][key], value)

    @mock.patch("desktop_runtime.runtime_setup.resolve_uv_binary")
    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_skips_dependency_install_when_runtime_imports_are_ready(
        self,
        run_mock,
        resolve_uv_binary_mock,
    ):
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
            (paths.runtime_dir / ".venv" / "bin").mkdir(parents=True)
            (paths.runtime_dir / ".venv" / "bin" / "python").write_text("")
            resolve_uv_binary_mock.return_value = paths.bundled_resources_dir / "uv"
            run_mock.return_value = mock.Mock(returncode=0)

            RuntimeDownloader().ensure_dependencies(paths)

        resolve_uv_binary_mock.assert_not_called()
        self.assertEqual(
            run_mock.call_args_list,
            [
                mock.call(
                    [
                        (paths.runtime_dir / ".venv" / "bin" / "python").as_posix(),
                        "-c",
                        mock.ANY,
                    ],
                    check=False,
                    env=mock.ANY,
                    capture_output=True,
                    text=True,
                )
            ],
        )

    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_prepares_base_model_with_runtime_python(self, run_mock):
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

            RuntimeDownloader().ensure_base_model(paths)

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], (paths.runtime_dir / ".venv" / "bin" / "python").as_posix())
        self.assertEqual(command[1], "-c")
        self.assertIn("from mlx_embeddings.convert import convert", command[2])
        self.assertIn('"google/siglip2-base-patch16-224"', command[2])
        self.assertEqual(command[3], paths.runtime_dir.as_posix())
        self.assertEqual(run_mock.call_args.kwargs["check"], True)
        for key, value in self._expected_process_env(paths).items():
            self.assertEqual(run_mock.call_args.kwargs["env"][key], value)

    @mock.patch("desktop_runtime.runtime_setup.subprocess.run")
    def test_runtime_downloader_prepares_public_anchor_with_runtime_python(self, run_mock):
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

            RuntimeDownloader().ensure_public_anchor(paths)

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], (paths.runtime_dir / ".venv" / "bin" / "python").as_posix())
        self.assertEqual(command[1], "-c")
        self.assertIn("from huggingface_hub import hf_hub_download", command[2])
        self.assertIn("from deployment.public_anchor import normalize_public_anchor_extract", command[2])
        self.assertIn("normalize_public_anchor_extract(extract_root)", command[2])
        self.assertIn("Lucas20250626/semanticgallery-stage2-public-anchor", command[2])
        self.assertIn("tarfile.open", command[2])
        self.assertEqual(command[3], paths.runtime_dir.as_posix())
        self.assertEqual(run_mock.call_args.kwargs["check"], True)
        for key, value in self._expected_process_env(paths).items():
            self.assertEqual(run_mock.call_args.kwargs["env"][key], value)


if __name__ == "__main__":
    unittest.main()
