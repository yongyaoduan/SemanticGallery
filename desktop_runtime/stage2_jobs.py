from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Protocol

from deployment.gallery_keys import gallery_artifact_key
from deployment.public_anchor import normalize_public_anchor_extract
from deployment.gallery_state import iter_gallery_paths, sha256_file
from desktop_runtime.paths import BUNDLED_RESOURCES_DIR_ENV_VAR


class Stage2Error(RuntimeError):
    pass


class Stage2RunnerProtocol(Protocol):
    def run(self, folder_path: Path) -> str: ...


class Stage2Job:
    def __init__(self, runner: Stage2RunnerProtocol):
        self.runner = runner

    def count_supported_images(self, folder_path: Path) -> int:
        return len(iter_gallery_paths(folder_path))

    def validate_folder(self, folder_path: Path) -> None:
        image_count = self.count_supported_images(folder_path)
        if image_count < 100:
            raise Stage2Error(
                f"This folder currently has {image_count} supported images. "
                "Stage 2 adaptation is only useful for folders with at least 100 images, so the app will skip adaptation for now."
            )

    def run(self, folder_path: Path) -> str:
        self.validate_folder(folder_path)
        return self.runner.run(folder_path)


class ScriptStage2Runner:
    def __init__(self, root_dir: Path, emit: Callable[[str], None]):
        self.root_dir = root_dir.expanduser().resolve()
        self.emit = emit

    def _script_path(self, name: str) -> Path:
        return self.root_dir / "scripts" / name

    @staticmethod
    def _prepare_runtime_env(env: dict[str, str]) -> dict[str, str]:
        resources_dir = env.get(BUNDLED_RESOURCES_DIR_ENV_VAR, "").strip()
        if not resources_dir:
            return env

        uv_path = (Path(resources_dir).expanduser().resolve() / "uv")
        if not uv_path.is_file():
            return env

        existing_parts = [part for part in env.get("PATH", "").split(os.pathsep) if part]
        uv_dir = uv_path.parent.as_posix()
        env["PATH"] = os.pathsep.join([uv_dir, *[part for part in existing_parts if part != uv_dir]])
        env["SEMANTICGALLERY_UV_BINARY"] = uv_path.as_posix()
        return env

    def _run_script(self, script_name: str, env: dict[str, str]) -> None:
        command = ["/bin/bash", self._script_path(script_name).as_posix()]
        try:
            proc = subprocess.Popen(
                command,
                cwd=self.root_dir.as_posix(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise Stage2Error(f"Stage 2 adaptation failed to start {script_name}: {exc}") from exc

        stdout = proc.stdout
        if stdout is None:
            proc.wait()
            raise Stage2Error(f"Stage 2 adaptation did not expose logs for {script_name}")

        last_log_line = ""
        for line in iter(stdout.readline, ""):
            normalized = line.rstrip("\r\n")
            if normalized:
                last_log_line = normalized
            self.emit(normalized)

        returncode = proc.wait()
        if returncode != 0:
            message = f"Stage 2 adaptation failed in {script_name} with exit code {returncode}"
            if last_log_line:
                message = f"{message}. Last log: {last_log_line}"
            raise Stage2Error(message)

    def run(self, folder_path: Path) -> str:
        folder_root = folder_path.expanduser().resolve()
        gallery_key = gallery_artifact_key(folder_root)
        private_data_dir = self.root_dir / "datasets" / "private_gallery_local" / gallery_key
        final_run_dir = self.root_dir / "logs" / "semanticgallery_private_data_adapted" / gallery_key
        normalize_public_anchor_extract(
            self.root_dir / ".cache" / "semanticgallery" / "stage2_public_anchor" / "extracted"
        )

        env = os.environ.copy()
        env.update(
            {
                "PRIVATE_GALLERY_DIR": folder_root.as_posix(),
                "PRIVATE_DATA_DIR": private_data_dir.as_posix(),
                "FINAL_RUN_DIR": final_run_dir.as_posix(),
                "PREPARE_PUBLIC_DATA": "0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        env = self._prepare_runtime_env(env)

        self._run_script("prepare_data.sh", env)
        self._run_script("adapt_best.sh", env)

        weights_path = final_run_dir / "weights.safetensors"
        if not weights_path.is_file():
            raise Stage2Error(f"Stage 2 adaptation did not produce weights at {weights_path.as_posix()}")

        signature = sha256_file(weights_path)
        if signature is None:
            raise Stage2Error(f"Stage 2 adaptation did not produce readable weights at {weights_path.as_posix()}")
        return signature
