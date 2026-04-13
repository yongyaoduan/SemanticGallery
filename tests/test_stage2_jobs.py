from __future__ import annotations
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deployment.public_anchor import normalize_public_anchor_extract
from deployment.gallery_state import sha256_file
from deployment.gallery_keys import gallery_artifact_key
from desktop_runtime.stage2_jobs import ScriptStage2Runner, Stage2Error, Stage2Job


class FakeRunner:
    def __init__(self, result: str = "stage2-signature"):
        self.result = result
        self.calls: list[Path] = []

    def run(self, folder_path: Path) -> str:
        self.calls.append(folder_path)
        return self.result


class FakeProcess:
    def __init__(self, *, stdout_text: str | None = "", returncode: int = 0):
        self.stdout = None if stdout_text is None else io.StringIO(stdout_text)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


class Stage2JobTests(unittest.TestCase):
    def test_validate_folder_rejects_fewer_than_100_supported_images(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-job-") as tmp_dir:
            folder = Path(tmp_dir) / "folder"
            folder.mkdir()
            for index in range(99):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            job = Stage2Job(FakeRunner())

            with self.assertRaisesRegex(
                Stage2Error,
                "This folder currently has 99 supported images. Stage 2 adaptation is only useful for folders with at least 100 images, so the app will skip adaptation for now.",
            ):
                job.run(folder)

    def test_validate_folder_allows_100_supported_images(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-job-") as tmp_dir:
            folder = Path(tmp_dir) / "folder"
            folder.mkdir()
            for index in range(100):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            runner = FakeRunner(result="sig-123")
            job = Stage2Job(runner)

            self.assertEqual(job.run(folder), "sig-123")
            self.assertEqual(runner.calls, [folder])

    def test_hidden_files_do_not_count_toward_threshold(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-job-") as tmp_dir:
            folder = Path(tmp_dir) / "folder"
            hidden = folder / ".hidden"
            nested_hidden = folder / ".cache" / "image.jpg"
            folder.mkdir()
            hidden.write_bytes(b"x")
            nested_hidden.parent.mkdir()
            nested_hidden.write_bytes(b"x")
            for index in range(99):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            job = Stage2Job(FakeRunner())

            self.assertEqual(job.count_supported_images(folder), 99)
            with self.assertRaises(Stage2Error):
                job.validate_folder(folder)

    def test_script_runner_invokes_prepare_before_adapt_with_active_folder_env(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-runner-") as tmp_dir:
            root_dir = Path(tmp_dir)
            folder = root_dir / "gallery"
            folder.mkdir()
            for index in range(100):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            calls: list[tuple[list[str], dict[str, str]]] = []
            gallery_key = gallery_artifact_key(folder)
            resolved_root = root_dir.expanduser().resolve()
            weights_path = resolved_root / "logs" / "semanticgallery_private_data_adapted" / gallery_key / "weights.safetensors"

            def fake_popen(command, *, cwd, env, stdout, stderr, text, bufsize):
                del cwd, stdout, stderr, text, bufsize
                calls.append((list(command), dict(env)))
                if command[-1].endswith("adapt_best.sh"):
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    weights_path.write_bytes(b"weights")
                    return FakeProcess(stdout_text="[adapt]\n")
                return FakeProcess(stdout_text="[prepare]\n")

            with patch("subprocess.Popen", side_effect=fake_popen):
                runner = ScriptStage2Runner(root_dir, emit=lambda _line: None)
                runner.run(folder)

            self.assertEqual(
                [Path(command[-1]).name for command, _ in calls],
                ["prepare_data.sh", "adapt_best.sh"],
            )
            self.assertEqual(calls[0][1]["PRIVATE_GALLERY_DIR"], folder.resolve().as_posix())
            self.assertEqual(calls[1][1]["PRIVATE_GALLERY_DIR"], folder.resolve().as_posix())
            self.assertEqual(calls[0][1]["PREPARE_PUBLIC_DATA"], "0")
            self.assertEqual(calls[1][1]["FINAL_RUN_DIR"], weights_path.parent.as_posix())

    def test_script_runner_emits_log_lines_and_returns_weights_signature(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-runner-") as tmp_dir:
            root_dir = Path(tmp_dir)
            folder = root_dir / "gallery"
            folder.mkdir()
            for index in range(100):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            emitted: list[str] = []
            gallery_key = gallery_artifact_key(folder)
            weights_path = root_dir.expanduser().resolve() / "logs" / "semanticgallery_private_data_adapted" / gallery_key / "weights.safetensors"

            def fake_popen(command, *, cwd, env, stdout, stderr, text, bufsize):
                del cwd, env, stdout, stderr, text, bufsize
                if command[-1].endswith("adapt_best.sh"):
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    weights_path.write_bytes(b"stage2-weights")
                    return FakeProcess(stdout_text="adapt line 1\nadapt line 2\n")
                return FakeProcess(stdout_text="prepare line 1\nprepare line 2\n")

            with patch("subprocess.Popen", side_effect=fake_popen):
                runner = ScriptStage2Runner(root_dir, emit=emitted.append)
                signature = runner.run(folder)

            self.assertEqual(emitted, ["prepare line 1", "prepare line 2", "adapt line 1", "adapt line 2"])
            self.assertEqual(signature, sha256_file(weights_path))

    def test_script_runner_prepends_bundled_uv_directory_to_path(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-runner-") as tmp_dir:
            root_dir = Path(tmp_dir)
            folder = root_dir / "gallery"
            resources_dir = root_dir / "resources"
            resources_dir.mkdir()
            (resources_dir / "uv").write_text("uv", encoding="utf-8")
            folder.mkdir()
            for index in range(100):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            calls: list[tuple[list[str], dict[str, str]]] = []
            gallery_key = gallery_artifact_key(folder)
            weights_path = root_dir.expanduser().resolve() / "logs" / "semanticgallery_private_data_adapted" / gallery_key / "weights.safetensors"

            def fake_popen(command, *, cwd, env, stdout, stderr, text, bufsize):
                del cwd, stdout, stderr, text, bufsize
                calls.append((list(command), dict(env)))
                if command[-1].endswith("adapt_best.sh"):
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    weights_path.write_bytes(b"weights")
                return FakeProcess(stdout_text="ok\n")

            with patch.dict(
                os.environ,
                {
                    "SEMANTICGALLERY_BUNDLED_RESOURCES_DIR": resources_dir.as_posix(),
                    "PATH": "/usr/bin",
                },
                clear=False,
            ):
                with patch("subprocess.Popen", side_effect=fake_popen):
                    runner = ScriptStage2Runner(root_dir, emit=lambda _line: None)
                    runner.run(folder)

            self.assertEqual(calls[0][1]["PATH"].split(os.pathsep)[0], resources_dir.resolve().as_posix())
            self.assertEqual(
                calls[0][1]["SEMANTICGALLERY_UV_BINARY"],
                (resources_dir.resolve() / "uv").as_posix(),
            )

    def test_script_runner_raises_on_subprocess_failure(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-runner-") as tmp_dir:
            root_dir = Path(tmp_dir)
            folder = root_dir / "gallery"
            folder.mkdir()
            for index in range(100):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            def fake_popen(command, *, cwd, env, stdout, stderr, text, bufsize):
                del command, cwd, env, stdout, stderr, text, bufsize
                return FakeProcess(stdout_text="bad line\n", returncode=1)

            with patch("subprocess.Popen", side_effect=fake_popen):
                runner = ScriptStage2Runner(root_dir, emit=lambda _line: None)

                with self.assertRaisesRegex(
                    Stage2Error,
                    "Stage 2 adaptation failed in prepare_data.sh with exit code 1. Last log: bad line",
                ):
                    runner.run(folder)

    def test_script_runner_raises_when_logs_are_unavailable(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-runner-") as tmp_dir:
            root_dir = Path(tmp_dir)
            folder = root_dir / "gallery"
            folder.mkdir()
            for index in range(100):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            def fake_popen(command, *, cwd, env, stdout, stderr, text, bufsize):
                del command, cwd, env, stdout, stderr, text, bufsize
                return FakeProcess(stdout_text=None)

            with patch("subprocess.Popen", side_effect=fake_popen):
                runner = ScriptStage2Runner(root_dir, emit=lambda _line: None)

                with self.assertRaises(Stage2Error):
                    runner.run(folder)

    def test_script_runner_repairs_cached_public_anchor_manifest_before_training(self):
        with tempfile.TemporaryDirectory(prefix="sg-stage2-runner-") as tmp_dir:
            root_dir = Path(tmp_dir)
            folder = root_dir / "gallery"
            folder.mkdir()
            for index in range(100):
                (folder / f"image-{index}.jpg").write_bytes(b"x")

            extract_root = root_dir / ".cache" / "semanticgallery" / "stage2_public_anchor" / "extracted"
            manifest_path = extract_root / "screen2words" / "manifest.jsonl"
            image_path = extract_root / "screen2words" / "images" / "rico" / "66545.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            manifest_path.write_text(
                '{"image_path":"/private/tmp/sg_public_anchor_seeded/screen2words/images/rico/66545.jpg","captions":["recipe app"],"split":"train","source":"screen2words"}\n',
                encoding="utf-8",
            )

            gallery_key = gallery_artifact_key(folder)
            weights_path = root_dir.expanduser().resolve() / "logs" / "semanticgallery_private_data_adapted" / gallery_key / "weights.safetensors"

            def fake_popen(command, *, cwd, env, stdout, stderr, text, bufsize):
                del cwd, env, stdout, stderr, text, bufsize
                if command[-1].endswith("adapt_best.sh"):
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    weights_path.write_bytes(b"weights")
                return FakeProcess(stdout_text="ok\n")

            with patch("subprocess.Popen", side_effect=fake_popen):
                runner = ScriptStage2Runner(root_dir, emit=lambda _line: None)
                runner.run(folder)

            repaired_row = normalize_public_anchor_extract(extract_root)
            self.assertEqual(repaired_row, 0)
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8").splitlines()[0],
                '{"image_path": "images/rico/66545.jpg", "captions": ["recipe app"], "split": "train", "source": "screen2words"}',
            )


if __name__ == "__main__":
    unittest.main()
