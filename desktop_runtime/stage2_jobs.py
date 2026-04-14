from __future__ import annotations

import os
import re
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
    _EPOCHS_RE = re.compile(r"^epochs=(?P<epochs>\d+)$")
    _TOTAL_STEPS_RE = re.compile(r"^epoch=(?P<epoch>\d+)\s+total_steps=(?P<total>\d+)\s+mode=stage2$")
    _STEP_RE = re.compile(r"^epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<total>\d+)\b")
    _VALIDATION_RE = re.compile(r"^epoch=(?P<epoch>\d+)\s+validation_start=true$")

    def __init__(
        self,
        root_dir: Path,
        emit: Callable[[str], None],
        emit_progress: Callable[[dict[str, object]], None] | None = None,
    ):
        self.root_dir = root_dir.expanduser().resolve()
        self.emit = emit
        self.emit_progress = emit_progress
        self._epochs = 1
        self._epoch_totals: dict[int, int] = {}
        self._stage2_total_units = 0

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

    def _publish_progress(self, payload: dict[str, object]) -> None:
        if self.emit_progress is None:
            return
        self.emit_progress(payload)

    def _reset_training_progress(self) -> None:
        self._epochs = 1
        self._epoch_totals = {}
        self._stage2_total_units = 0

    def _estimated_total_train_steps(self, fallback_total: int | None = None) -> int:
        if not self._epoch_totals and not fallback_total:
            return 0

        representative_total = fallback_total or max(self._epoch_totals.values(), default=0)
        estimated_steps = 0
        for epoch in range(1, self._epochs + 1):
            estimated_steps += self._epoch_totals.get(epoch, representative_total)
        return estimated_steps

    def _stage2_units_total(self, fallback_total: int | None = None) -> int:
        estimated_train_steps = self._estimated_total_train_steps(fallback_total)
        if estimated_train_steps <= 0:
            return self._stage2_total_units
        self._stage2_total_units = 1 + estimated_train_steps + self._epochs + 1
        return self._stage2_total_units

    def _completed_train_steps_before(self, epoch: int) -> int:
        return sum(self._epoch_totals.get(index, 0) for index in range(1, epoch))

    def _handle_adapt_progress_line(self, line: str) -> None:
        if match := self._EPOCHS_RE.match(line):
            self._epochs = max(1, int(match.group("epochs")))
            return

        if match := self._TOTAL_STEPS_RE.match(line):
            epoch = int(match.group("epoch"))
            total_steps = int(match.group("total"))
            self._epoch_totals[epoch] = total_steps
            total_units = self._stage2_units_total(total_steps)
            current_units = 1 + self._completed_train_steps_before(epoch) + max(0, epoch - 1)
            self._publish_progress(
                {
                    "status": "running",
                    "phase": "adapt",
                    "current": current_units,
                    "total": total_units,
                    "phaseCurrent": 0,
                    "phaseTotal": total_steps,
                    "message": f"Training epoch {epoch} of {self._epochs} · step 0 of {total_steps}",
                }
            )
            return

        if match := self._STEP_RE.match(line):
            epoch = int(match.group("epoch"))
            step = int(match.group("step"))
            total_steps = int(match.group("total"))
            self._epoch_totals.setdefault(epoch, total_steps)
            total_units = self._stage2_units_total(total_steps)
            current_units = 1 + self._completed_train_steps_before(epoch) + max(0, epoch - 1) + step
            self._publish_progress(
                {
                    "status": "running",
                    "phase": "adapt",
                    "current": current_units,
                    "total": total_units,
                    "phaseCurrent": step,
                    "phaseTotal": total_steps,
                    "message": f"Training epoch {epoch} of {self._epochs} · step {step} of {total_steps}",
                }
            )
            return

        if match := self._VALIDATION_RE.match(line):
            epoch = int(match.group("epoch"))
            epoch_total = self._epoch_totals.get(epoch, 0)
            total_units = self._stage2_units_total(epoch_total or None)
            current_units = 1 + self._completed_train_steps_before(epoch) + epoch_total + epoch
            self._publish_progress(
                {
                    "status": "running",
                    "phase": "validate",
                    "current": current_units,
                    "total": total_units,
                    "phaseCurrent": epoch,
                    "phaseTotal": self._epochs,
                    "message": f"Validating epoch {epoch} of {self._epochs}",
                }
            )

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
            if script_name == "adapt_best.sh" and normalized:
                self._handle_adapt_progress_line(normalized)

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

        self._reset_training_progress()
        self._publish_progress(
            {
                "status": "running",
                "phase": "prepare",
                "current": 0,
                "total": 0,
                "phaseCurrent": 0,
                "phaseTotal": 1,
                "message": "Preparing private adaptation data.",
            }
        )
        self._run_script("prepare_data.sh", env)
        self._publish_progress(
            {
                "status": "running",
                "phase": "prepare",
                "current": 1,
                "total": 1,
                "phaseCurrent": 1,
                "phaseTotal": 1,
                "message": "Private adaptation data is ready.",
            }
        )
        self._run_script("adapt_best.sh", env)

        total_units = self._stage2_total_units or 3
        self._publish_progress(
            {
                "status": "running",
                "phase": "finalize",
                "current": max(1, total_units - 1),
                "total": total_units,
                "phaseCurrent": 0,
                "phaseTotal": 1,
                "message": "Loading the adapted encoder.",
            }
        )

        weights_path = final_run_dir / "weights.safetensors"
        if not weights_path.is_file():
            raise Stage2Error(f"Stage 2 adaptation did not produce weights at {weights_path.as_posix()}")

        signature = sha256_file(weights_path)
        if signature is None:
            raise Stage2Error(f"Stage 2 adaptation did not produce readable weights at {weights_path.as_posix()}")
        return signature
