from __future__ import annotations

import subprocess
from enum import StrEnum
from pathlib import Path
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
        python_bin = paths.runtime_dir / ".venv" / "bin" / "python"
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
        _ = paths

    def ensure_public_anchor(self, paths: AppPaths) -> None:
        _ = paths


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
