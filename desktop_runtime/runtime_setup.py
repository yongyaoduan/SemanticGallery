from __future__ import annotations

import os
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
    def _dependency_check_code() -> str:
        return """
        import datasets, fastapi, huggingface_hub, jinja2, mlx, mlx_embeddings, multipart, pillow_heif, tqdm, uvicorn
        """

    @staticmethod
    def _runtime_python(paths: AppPaths) -> Path:
        return paths.runtime_dir / ".venv" / "bin" / "python"

    @staticmethod
    def _process_env(paths: AppPaths) -> dict[str, str]:
        cache_root = paths.cache_dir
        python_install_dir = paths.support_dir / "python"
        huggingface_root = cache_root / "huggingface"

        cache_root.mkdir(parents=True, exist_ok=True)
        python_install_dir.mkdir(parents=True, exist_ok=True)
        huggingface_root.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["UV_CACHE_DIR"] = (cache_root / "uv").as_posix()
        env["UV_PYTHON_INSTALL_DIR"] = python_install_dir.as_posix()
        env["XDG_CACHE_HOME"] = cache_root.as_posix()
        env["HF_HOME"] = huggingface_root.as_posix()
        env["HUGGINGFACE_HUB_CACHE"] = (huggingface_root / "hub").as_posix()
        env["TRANSFORMERS_CACHE"] = (cache_root / "transformers").as_posix()
        return env

    @classmethod
    def _run_runtime_code(cls, paths: AppPaths, code: str, *args: str) -> None:
        python_bin = cls._runtime_python(paths)
        subprocess.run(
            [python_bin.as_posix(), "-c", dedent(code), *args],
            check=True,
            env=cls._process_env(paths),
        )

    @classmethod
    def _dependencies_ready(cls, paths: AppPaths) -> bool:
        python_bin = cls._runtime_python(paths)
        if not python_bin.is_file():
            return False

        result = subprocess.run(
            [python_bin.as_posix(), "-c", dedent(cls._dependency_check_code())],
            check=False,
            env=cls._process_env(paths),
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def ensure_python_runtime(self, paths: AppPaths) -> None:
        if self._runtime_python(paths).is_file():
            return

        uv_path = resolve_uv_binary(paths, override=None)
        env = self._process_env(paths)
        subprocess.run(
            [
                uv_path.as_posix(),
                "python",
                "install",
                "3.12",
                "--install-dir",
                (paths.support_dir / "python").as_posix(),
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                uv_path.as_posix(),
                "venv",
                (paths.runtime_dir / ".venv").as_posix(),
                "--python",
                "3.12",
            ],
            check=True,
            env=env,
        )

    def ensure_dependencies(self, paths: AppPaths) -> None:
        if self._dependencies_ready(paths):
            return

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
            env=self._process_env(paths),
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
            if runtime_root.as_posix() not in sys.path:
                sys.path.insert(0, runtime_root.as_posix())

            from deployment.public_anchor import normalize_public_anchor_extract

            cache_dir = runtime_root / ".cache" / "semanticgallery" / "stage2_public_anchor"
            archive_path = cache_dir / "semanticgallery-stage2-public-anchor.tar.gz"
            metadata_path = cache_dir / "sample_info.json"
            extract_root = cache_dir / "extracted"
            flickr_captions = extract_root / "flickr30k" / "captions.txt"
            screen2words_manifest = extract_root / "screen2words" / "manifest.jsonl"

            if flickr_captions.is_file() and screen2words_manifest.is_file():
                normalize_public_anchor_extract(extract_root)
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
            normalize_public_anchor_extract(extract_root)
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
        self._start("check-runtime", "Checking the local runtime", 1, 6)
        self.downloader.ensure_python_runtime(self.paths)
        self._finish("check-runtime", "Local runtime is ready", 2, 6)

        self._start("prepare-dependencies", "Preparing Python dependencies", 2, 6)
        self.downloader.ensure_dependencies(self.paths)
        self._finish("prepare-dependencies", "Python dependencies are ready", 3, 6)

        self._start("prepare-base-model", "Preparing the base model", 3, 6)
        self.downloader.ensure_base_model(self.paths)
        self._finish("prepare-base-model", "Base model is ready", 4, 6)

        self._start("prepare-public-anchor", "Preparing the public adaptation set", 4, 6)
        self.downloader.ensure_public_anchor(self.paths)
        self._finish("prepare-public-anchor", "Public adaptation set is ready", 5, 6)

        self._start("finish-setup", "Finishing setup", 5, 6)
        self._finish("finish-setup", "Setup is complete", 6, 6)
        return SetupStatus.READY
