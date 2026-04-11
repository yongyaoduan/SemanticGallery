from __future__ import annotations

import subprocess
from enum import StrEnum
from pathlib import Path
from textwrap import dedent
from typing import Protocol

from desktop_runtime.paths import AppPaths, resolve_uv_binary
from desktop_runtime.progress import ProgressEmitter, ProgressEvent


class SetupStatus(StrEnum):
    READY = "ready"


class RuntimeDownloaderProtocol(Protocol):
    def ensure_python_runtime(self, paths: AppPaths) -> None: ...

    def ensure_dependencies(self, paths: AppPaths) -> None: ...

    def ensure_base_model(self, paths: AppPaths) -> None: ...

    def ensure_public_anchor(self, paths: AppPaths) -> None: ...


class RuntimeDownloader:
    @staticmethod
    def _runtime_python(paths: AppPaths) -> Path:
        return paths.runtime_dir / ".venv" / "bin" / "python"

    @classmethod
    def _run_runtime_code(cls, paths: AppPaths, code: str, *args: str) -> None:
        python_bin = cls._runtime_python(paths)
        subprocess.run([python_bin.as_posix(), "-c", dedent(code), *args], check=True)

    def ensure_python_runtime(self, paths: AppPaths) -> None:
        uv_path = resolve_uv_binary(paths, override=None)
        subprocess.run([uv_path.as_posix(), "python", "install", "3.12"], check=True)
        subprocess.run(
            [
                uv_path.as_posix(),
                "venv",
                (paths.runtime_dir / ".venv").as_posix(),
                "--python",
                "3.12",
            ],
            check=True,
        )

    def ensure_dependencies(self, paths: AppPaths) -> None:
        uv_path = resolve_uv_binary(paths, override=None)
        python_bin = self._runtime_python(paths)
        requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
        subprocess.run(
            [
                uv_path.as_posix(),
                "pip",
                "install",
                "--python",
                python_bin.as_posix(),
                "-r",
                requirements_path.as_posix(),
            ],
            check=True,
        )

    def ensure_base_model(self, paths: AppPaths) -> None:
        self._run_runtime_code(
            paths,
            """
            import sys
            from pathlib import Path

            from mlx_embeddings.convert import convert

            runtime_root = Path(sys.argv[1]).resolve()
            model_dir = runtime_root / ".cache" / "mlx" / "siglip2-base-patch16-224-f32"
            if not (model_dir / "config.json").is_file():
                model_dir.parent.mkdir(parents=True, exist_ok=True)
                convert(
                    "google/siglip2-base-patch16-224",
                    mlx_path=model_dir.as_posix(),
                    dtype="float32",
                    skip_vision=False,
                )
            """,
            paths.runtime_dir.as_posix(),
        )

    def ensure_public_anchor(self, paths: AppPaths) -> None:
        self._run_runtime_code(
            paths,
            """
            import shutil
            import sys
            import tarfile
            from pathlib import Path

            from huggingface_hub import hf_hub_download

            runtime_root = Path(sys.argv[1]).resolve()
            cache_dir = runtime_root / ".cache" / "semanticgallery" / "stage2_public_anchor"
            archive_path = cache_dir / "semanticgallery-stage2-public-anchor.tar.gz"
            metadata_path = cache_dir / "sample_info.json"
            extract_root = cache_dir / "extracted"
            flickr_captions = extract_root / "flickr30k" / "captions.txt"
            screen2words_manifest = extract_root / "screen2words" / "manifest.jsonl"

            if flickr_captions.is_file() and screen2words_manifest.is_file():
                raise SystemExit(0)

            cache_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("semanticgallery-stage2-public-anchor.tar.gz", "sample_info.json"):
                hf_hub_download(
                    repo_id="Lucas20250626/semanticgallery-stage2-public-anchor",
                    repo_type="dataset",
                    revision="main",
                    filename=filename,
                    local_dir=cache_dir.as_posix(),
                )

            tmp_root = cache_dir / "extracting"
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                try:
                    tar.extractall(tmp_root, filter="data")
                except TypeError:
                    tar.extractall(tmp_root)

            if extract_root.exists():
                shutil.rmtree(extract_root)
            tmp_root.rename(extract_root)

            if metadata_path.exists():
                shutil.copy2(metadata_path, extract_root / "sample_info.json")
            """,
            paths.runtime_dir.as_posix(),
        )


class RuntimeSetup:
    def __init__(
        self,
        paths: AppPaths,
        downloader: RuntimeDownloaderProtocol,
        emit: ProgressEmitter,
    ):
        self.paths = paths
        self.downloader = downloader
        self.emit = emit

    def _start(self, task: str, message: str, current: int, total: int) -> None:
        self.emit(
            ProgressEvent(
                task=task,
                phase="start",
                message=message,
                current=current,
                total=total,
            )
        )

    def _finish(self, task: str, message: str, current: int, total: int) -> None:
        self.emit(
            ProgressEvent(
                task=task,
                phase="finish",
                message=message,
                current=current,
                total=total,
            )
        )

    def prepare(self) -> SetupStatus:
        self.paths.support_dir.mkdir(parents=True, exist_ok=True)
        self._start("check-runtime", "Checking the local runtime", 0, 5)
        self.downloader.ensure_python_runtime(self.paths)
        self._finish("check-runtime", "Local runtime is ready", 1, 5)

        self._start("prepare-dependencies", "Preparing Python dependencies", 1, 5)
        self.downloader.ensure_dependencies(self.paths)
        self._finish("prepare-dependencies", "Python dependencies are ready", 2, 5)

        self._start("prepare-base-model", "Preparing the base model", 2, 5)
        self.downloader.ensure_base_model(self.paths)
        self._finish("prepare-base-model", "Base model is ready", 3, 5)

        self._start("prepare-public-anchor", "Preparing the public adaptation set", 3, 5)
        self.downloader.ensure_public_anchor(self.paths)
        self._finish("prepare-public-anchor", "Public adaptation set is ready", 4, 5)

        self._start("finish-setup", "Finishing setup", 4, 5)
        self._finish("finish-setup", "Setup is complete", 5, 5)
        return SetupStatus.READY
